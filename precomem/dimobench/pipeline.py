"""The six-stage DiMoBench construction pipeline (Appendix D.6).

Generator–verifier separation is enforced structurally: the *generator*
client writes personas, skeletons, dialogue and questions; the *verifier*
client only scores questions. Every stage checkpoints its output to
``work_dir/<persona_id>/stageN_*.json`` so a build can resume.

Stage 1  Persona construction        generator   PersonaCard
Stage 2  Life skeleton & cue scripts generator   List[SessionScript]
Stage 3  Dialogue generation         generator   List[Turn] (sessions in order, rolling context)
Stage 4  QA generation               generator   List[QARecord] (one per QACell of the spec)
Stage 5  Consistency verification    verifier    six criteria, REVISE loop, REJECT discard
Stage 6  Human–machine review        offline     20% stratified audit shard (``audit.py``)
"""

from __future__ import annotations

import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..llm.client import LLMClient
from ..utils.logging import get_logger
from . import prompts as P
from .audit import mark_audit_shard
from .schema import (
    Character,
    CueScript,
    PersonaCard,
    QARecord,
    SessionScript,
    SlotSpec,
    Turn,
    Verification,
    state_timeline,
)
from .spec import CueKind, DiMoBenchSpec, QACell, SessionPlan, base_spec, xl_spec

_LOGGER = get_logger("dimobench")

ACCEPT, REVISE, REJECT = "ACCEPT", "REVISE", "REJECT"
UNRESOLVED = "UNRESOLVED"  # still REVISE after the last round
CRITERIA = (
    "answer_grounding",
    "evidence_complete",
    "temporal_consistency",
    "entity_accuracy",
    "interference_valid",
    "drift_coherent",
)


@dataclass
class BuildOptions:
    rounds: int = 3  # verifier rounds per question (paper: max 3)
    target_accept: float = 0.95  # reported, not enforced
    dialogue_temperature: float = 0.7
    qa_temperature: float = 0.4
    verifier_temperature: float = 0.0
    skeleton_chunk: int = 20  # sessions per skeleton call
    qa_batch: int = 10  # questions per QA call
    workers: int = 4  # parallel QA / verification calls
    max_sessions: Optional[int] = None  # smoke runs: truncate the dialogue
    max_qa: Optional[int] = None  # smoke runs: cap the number of questions
    audit_fraction: float = 0.20
    keep_unresolved: bool = False  # keep questions still marked REVISE after the last round
    seed: int = 1234


@dataclass
class BuildStats:
    persona_id: str
    n_turns: int = 0
    n_cues_planned: int = 0
    n_cues_realized: int = 0
    n_qa_requested: int = 0
    n_qa_generated: int = 0
    n_qa_final: int = 0
    accept_by_round: Dict[int, int] = field(default_factory=dict)  # round at which a question was accepted
    n_accept: int = 0
    n_rejected: int = 0  # REJECT verdicts, discarded
    n_unresolved: int = 0  # still REVISE after the last round
    n_dropped_invalid: int = 0  # the generator never produced a valid question for the cell
    generator_model: str = ""
    verifier_model: str = ""
    seconds: float = 0.0

    @property
    def accept_rate(self) -> float:
        return self.n_accept / self.n_qa_generated if self.n_qa_generated else 0.0

    def to_dict(self) -> Dict[str, Any]:
        d = dict(self.__dict__)
        d["accept_rate"] = round(self.accept_rate, 4)
        return d


# ----------------------------------------------------------------------
# Seeds
# ----------------------------------------------------------------------


def spec_for_seed(seed: Mapping[str, Any], variant: str, rng_seed: int = 1234) -> DiMoBenchSpec:
    """Build the structural spec implied by a persona seed."""
    slots = [SlotSpec.from_dict(s) for s in seed["slots"]]
    if variant == "base":
        if len(slots) != 1:
            raise ValueError("a base-track seed has exactly one attribute slot")
        return base_spec(slots[0].slot, seed=rng_seed)
    if variant == "xl":
        drift_slots: List[List[str]] = [[], [], []]
        for s in slots:
            for k in s.transitions:
                if not 1 <= k <= 3:
                    raise ValueError(f"slot {s.slot}: transition index {k} outside 1..3")
                drift_slots[k - 1].append(s.slot)
        if any(not ds for ds in drift_slots):
            raise ValueError("an XL seed must shift at least one slot at each of the three drifts")
        return xl_spec(drift_slots=drift_slots, seed=rng_seed)
    raise ValueError(f"unknown variant {variant!r} (expected 'base' or 'xl')")


# ----------------------------------------------------------------------
# Builder
# ----------------------------------------------------------------------


class DiMoBenchBuilder:
    def __init__(
        self,
        spec: DiMoBenchSpec,
        generator: LLMClient,
        verifier: LLMClient,
        options: Optional[BuildOptions] = None,
        work_dir: Optional[Path] = None,
        resume: bool = True,
    ) -> None:
        if generator is verifier or (generator.model == verifier.model):
            _LOGGER.warning(
                "Generator and verifier are the same model (%s); the paper uses different models "
                "to avoid self-reinforcing bias.",
                generator.model,
            )
        self.spec = spec
        self.generator = generator
        self.verifier = verifier
        self.opt = options or BuildOptions()
        self.work_dir = work_dir
        self.resume = resume

    # -- checkpoints ----------------------------------------------------
    def _ckpt(self, persona_id: str, name: str) -> Optional[Path]:
        if self.work_dir is None:
            return None
        path = self.work_dir / persona_id / f"{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _load(self, path: Optional[Path]) -> Optional[Any]:
        if path is not None and self.resume and path.exists():
            _LOGGER.info("resume: %s", path)
            return json.loads(path.read_text(encoding="utf-8"))
        return None

    @staticmethod
    def _save(path: Optional[Path], payload: Any) -> None:
        if path is not None:
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # ==================================================================
    def build(self, seed: Mapping[str, Any]) -> Dict[str, Any]:
        t0 = time.perf_counter()
        persona_id = str(seed["persona_id"])
        stats = BuildStats(
            persona_id=persona_id, generator_model=self.generator.model, verifier_model=self.verifier.model
        )

        persona = self.stage1_persona(seed)
        scripts = self.stage2_skeleton(persona)
        turns = self.stage3_dialogue(persona, scripts, stats)
        qas = self.stage4_qa(persona, scripts, turns, stats)
        qas = self.stage5_verify(persona, scripts, turns, qas, stats)
        qas = mark_audit_shard(qas, fraction=self.opt.audit_fraction, seed=self.opt.seed)
        stats.n_qa_final = len(qas)
        stats.seconds = round(time.perf_counter() - t0, 1)

        return self.assemble(persona, scripts, turns, qas, stats)

    # ==================================================================
    # Stage 1
    # ==================================================================
    def stage1_persona(self, seed: Mapping[str, Any]) -> PersonaCard:
        persona_id = str(seed["persona_id"])
        ckpt = self._ckpt(persona_id, "stage1_persona")
        cached = self._load(ckpt)
        if cached:
            return PersonaCard.from_dict(cached)

        slots = [SlotSpec.from_dict(s) for s in seed["slots"]]
        plan = {
            "persona_id": persona_id,
            "life_stage": seed.get("life_stage", ""),
            "big_five": seed.get("big_five", {}),
            "slots": [s.to_dict() for s in slots],
            "drift_turns": self.spec.drift_turns,
            "drift_hints": seed.get("drift_hints", {}),
            "social_circle_hint": seed.get("social_circle_hint", []),
        }
        payload = self.generator.complete_json(
            P.PERSONA_PROMPT.format(plan_json=json.dumps(plan, ensure_ascii=False, indent=2)),
            temperature=self.opt.dialogue_temperature,
        )
        circle = [
            Character(
                name=str(c.get("name", "")).strip(),
                relation=str(c.get("relation", "")),
                confusable_with=c.get("confusable_with") or None,
            )
            for c in payload.get("social_circle") or []
            if isinstance(c, dict) and str(c.get("name", "")).strip()
        ]
        if len(circle) < 2:
            circle = _fallback_circle(seed.get("social_circle_hint") or [])
        triggers: Dict[int, str] = {}
        raw_triggers = payload.get("drift_triggers") or {}
        for k in range(1, len(self.spec.drifts) + 1):
            trig = raw_triggers.get(str(k)) if isinstance(raw_triggers, dict) else None
            triggers[k] = str(
                trig
                or (seed.get("drift_hints") or {}).get(str(k))
                or (seed.get("drift_hints") or {}).get(k)
                or "an unspecified life event"
            )

        persona = PersonaCard(
            persona_id=persona_id,
            name=str(payload.get("name") or seed.get("name") or persona_id.replace("_", " ").title()),
            summary=str(payload.get("summary", "")),
            life_stage=str(seed.get("life_stage", "")),
            big_five={str(k): float(v) for k, v in (seed.get("big_five") or {}).items()},
            social_circle=circle,
            routine=[str(r) for r in payload.get("routine") or []],
            speaking_style=str(payload.get("speaking_style", "")),
            slots=slots,
            drift_triggers=triggers,
            seed=dict(seed),
        )
        self._save(ckpt, persona.to_dict())
        return persona

    # ==================================================================
    # Stage 2
    # ==================================================================
    def stage2_skeleton(self, persona: PersonaCard) -> List[SessionScript]:
        ckpt = self._ckpt(persona.persona_id, "stage2_skeleton")
        cached = self._load(ckpt)
        if cached:
            return [SessionScript.from_dict(s) for s in cached]

        plans = self.spec.session_plans()
        if self.opt.max_sessions:
            plans = plans[: self.opt.max_sessions]
        timeline = state_timeline(persona, self.spec.drift_turns, self.spec.n_turns)
        scripts: List[SessionScript] = []
        previous_summary = "(this is the first chunk)"
        for start in range(0, len(plans), self.opt.skeleton_chunk):
            chunk = plans[start : start + self.opt.skeleton_chunk]
            plan_json = [self._plan_for_skeleton(p, persona) for p in chunk]
            payload = self.generator.complete_json(
                P.SKELETON_PROMPT.format(
                    first_session=chunk[0].index,
                    last_session=chunk[-1].index,
                    turns_per_session=self.spec.turns_per_session,
                    n_sessions=self.spec.n_sessions,
                    n_turns=self.spec.n_turns,
                    persona_json=json.dumps(persona.brief(), ensure_ascii=False, indent=2),
                    timeline_json=json.dumps(timeline, ensure_ascii=False),
                    triggers_json=json.dumps(persona.drift_triggers, ensure_ascii=False),
                    previous_summary=previous_summary,
                    plan_json=json.dumps(plan_json, ensure_ascii=False, indent=2),
                ),
                temperature=self.opt.dialogue_temperature,
                max_tokens=8192,
            )
            by_index = {int(s.get("index", -1)): s for s in payload.get("sessions") or [] if isinstance(s, dict)}
            for p in chunk:
                scripts.append(self._script_from_payload(p, persona, by_index.get(p.index) or {}))
            previous_summary = str(payload.get("chunk_summary") or previous_summary)
            _LOGGER.info("[%s] skeleton sessions %d-%d", persona.persona_id, chunk[0].index, chunk[-1].index)
        self._save(ckpt, [s.to_dict() for s in scripts])
        return scripts

    def _plan_for_skeleton(self, p: SessionPlan, persona: PersonaCard) -> Dict[str, Any]:
        return {
            "index": p.index,
            "turns": [p.first_turn, p.last_turn],
            "phase": p.phase.value,
            "drift_index": p.drift_index,
            "cues": [
                {
                    "cue_id": c.cue_id,
                    "slot": c.slot,
                    "kind": c.kind.value,
                    "state_to_express": persona.slot(c.slot).state_at_drift(
                        c.drift_index if c.kind is not CueKind.BASELINE else p.drift_index
                    ),
                }
                for c in p.cues
            ],
            "confounder": p.confounder.value if p.confounder else None,
            "mention_confusable_entity": p.mention_confusable_entity,
        }

    def _script_from_payload(self, p: SessionPlan, persona: PersonaCard, raw: Dict[str, Any]) -> SessionScript:
        raw_cues = {str(c.get("cue_id")): c for c in raw.get("cues") or [] if isinstance(c, dict)}
        cues: List[CueScript] = []
        for c in p.cues:
            state = persona.slot(c.slot).state_at_drift(
                c.drift_index if c.kind is not CueKind.BASELINE else p.drift_index
            )
            rc = raw_cues.get(c.cue_id, {})
            description = str(rc.get("description") or "").strip()
            if not description:
                description = f"A small everyday choice or reaction that reveals, implicitly: {state}."
            cues.append(
                CueScript(
                    cue_id=c.cue_id,
                    session=p.index,
                    slot=c.slot,
                    kind=c.kind,
                    drift_index=c.drift_index,
                    state_to_express=state,
                    description=description,
                    surface_hint=str(rc.get("surface_hint") or ""),
                )
            )
        return SessionScript(
            index=p.index,
            first_turn=p.first_turn,
            last_turn=p.last_turn,
            phase=p.phase,
            drift_index=p.drift_index,
            summary=str(raw.get("summary") or f"Session {p.index}: everyday conversation."),
            setting=str(raw.get("setting") or ""),
            cues=cues,
            confounder=p.confounder,
            confounder_script=str(raw.get("confounder_script") or ""),
            mention_confusable_entity=p.mention_confusable_entity,
        )

    # ==================================================================
    # Stage 3
    # ==================================================================
    def stage3_dialogue(self, persona: PersonaCard, scripts: Sequence[SessionScript], stats: BuildStats) -> List[Turn]:
        ckpt = self._ckpt(persona.persona_id, "stage3_dialogue")
        cached = self._load(ckpt)
        if cached:
            turns = [Turn.from_dict(t) for t in cached]
            stats.n_turns = len(turns)
            stats.n_cues_planned = sum(len(s.cues) for s in scripts)
            stats.n_cues_realized = sum(len(t.cue_ids) for t in turns)
            return turns

        turns: List[Turn] = []
        rolling_summary = "(start of the conversation)"
        for script in scripts:
            session_turns, rolling_summary = self._generate_session(persona, script, turns, rolling_summary)
            turns.extend(session_turns)
            realized = {c for t in session_turns for c in t.cue_ids}
            missing = [c.cue_id for c in script.cues if c.cue_id not in realized]
            if missing:
                _LOGGER.warning("[%s] session %d: cues not localized %s", persona.persona_id, script.index, missing)
            _LOGGER.info("[%s] session %d/%d written", persona.persona_id, script.index, len(scripts))
            self._save(ckpt, [t.to_dict() for t in turns])  # partial progress is resumable per session
        stats.n_turns = len(turns)
        stats.n_cues_planned = sum(len(s.cues) for s in scripts)
        stats.n_cues_realized = sum(len(t.cue_ids) for t in turns)
        return turns

    def _generate_session(
        self, persona: PersonaCard, script: SessionScript, previous: Sequence[Turn], rolling_summary: str
    ) -> Tuple[List[Turn], str]:
        n = self.spec.turns_per_session
        tail = "\n".join(f"[{t.speaker}] {t.text}" for t in previous[-4:]) or "(none)"
        plan = {
            "session_index": script.index,
            "phase": script.phase.value,
            "setting": script.setting,
            "summary": script.summary,
            "cues": [
                {
                    "cue_id": c.cue_id,
                    "kind": c.kind.value,
                    "description": c.description,
                    "surface_hint": c.surface_hint,
                    "state_to_express": c.state_to_express,
                }
                for c in script.cues
            ],
            "confounder": script.confounder.value if script.confounder else None,
            "confounder_script": script.confounder_script,
            "mention_confusable_entity": script.mention_confusable_entity,
            "confusable_characters": [c.to_dict() for c in persona.social_circle],
        }
        prompt = P.DIALOGUE_PROMPT.format(
            session_index=script.index,
            user_name=persona.name,
            turns_per_session=n,
            persona_json=json.dumps(persona.brief(), ensure_ascii=False, indent=2),
            rolling_summary=rolling_summary,
            previous_tail=tail,
            plan_json=json.dumps(plan, ensure_ascii=False, indent=2),
        )
        best: Optional[Tuple[List[Turn], str]] = None
        for _ in range(2):
            payload = self.generator.complete_json(prompt, temperature=self.opt.dialogue_temperature, max_tokens=4096)
            parsed = self._parse_session(payload, script, previous)
            if parsed is None:
                continue
            session_turns, summary = parsed
            realized = {c for t in session_turns for c in t.cue_ids}
            if all(c.cue_id in realized for c in script.cues):
                return session_turns, summary
            best = best or (session_turns, summary)
            prompt += "\n\nNOTE: the previous draft failed to realize every cue id; realize ALL of them in user turns."
        if best is None:
            raise RuntimeError(f"session {script.index}: generator returned no usable dialogue")
        return best

    def _parse_session(
        self, payload: Dict[str, Any], script: SessionScript, previous: Sequence[Turn]
    ) -> Optional[Tuple[List[Turn], str]]:
        n = self.spec.turns_per_session
        raw_turns = [t for t in payload.get("turns") or [] if isinstance(t, dict) and str(t.get("text", "")).strip()]
        if len(raw_turns) < n:
            return None
        raw_turns = raw_turns[:n]
        offset = len(previous)
        turns: List[Turn] = []
        for k, rt in enumerate(raw_turns, start=1):
            turns.append(
                Turn(
                    turn=offset + k,
                    session=script.index,
                    turn_in_session=k,
                    speaker="user" if k % 2 == 1 else "assistant",  # enforce alternation
                    text=str(rt["text"]).strip(),
                    phase=script.phase,
                )
            )
        valid_ids = {c.cue_id for c in script.cues}
        for rc in payload.get("cue_realizations") or []:
            if not isinstance(rc, dict) or str(rc.get("cue_id")) not in valid_ids:
                continue
            k = _as_index(rc.get("turn"), n)
            if k is None:
                continue
            if k % 2 == 0:  # cues must live in user turns; move to the preceding user turn
                k -= 1
            turns[k - 1].cue_ids.append(str(rc["cue_id"]))
        if script.confounder is not None:
            for k in payload.get("confounder_turns") or []:
                idx = _as_index(k, n)
                if idx is not None:
                    turns[idx - 1].tags.append(script.confounder.value)
        for k in payload.get("entity_decoy_turns") or []:
            idx = _as_index(k, n)
            if idx is not None:
                turns[idx - 1].tags.append(CueKind.ENTITY_DECOY.value)
        summary = str(payload.get("rolling_summary") or "").strip() or script.summary
        return turns, summary

    # ==================================================================
    # Stage 4
    # ==================================================================
    def stage4_qa(
        self, persona: PersonaCard, scripts: Sequence[SessionScript], turns: Sequence[Turn], stats: BuildStats
    ) -> List[QARecord]:
        ckpt = self._ckpt(persona.persona_id, "stage4_qa")
        cells = [c for c in self.spec.qa_cells() if c.anchor_turn <= len(turns)]
        if self.opt.max_qa:
            cells = cells[: self.opt.max_qa]
        stats.n_qa_requested = len(cells)
        cached = self._load(ckpt)
        if cached:
            records = [QARecord.from_dict(q) for q in cached]
            stats.n_qa_generated = len(records)
            return records

        cue_slot = {c.cue_id: c.slot for sc in scripts for c in sc.cues}
        groups: Dict[int, List[QACell]] = {}
        for c in cells:
            groups.setdefault(c.anchor_turn, []).append(c)
        batches: List[List[QACell]] = []
        for anchor in sorted(groups):
            group = groups[anchor]
            for i in range(0, len(group), self.opt.qa_batch):
                batches.append(group[i : i + self.opt.qa_batch])

        def run(batch: List[QACell]) -> List[QARecord]:
            return self._generate_batch(persona, turns, batch, cue_slot)

        records: List[QARecord] = []
        with ThreadPoolExecutor(max_workers=max(1, self.opt.workers)) as pool:
            for out in pool.map(run, batches):
                records.extend(out)
        records.sort(key=lambda q: q.qid)
        stats.n_qa_generated = len(records)
        stats.n_dropped_invalid = len(cells) - len(records)
        _LOGGER.info("[%s] QA generated: %d / %d requested", persona.persona_id, len(records), len(cells))
        self._save(ckpt, [q.to_dict() for q in records])
        return records

    def _generate_batch(
        self, persona: PersonaCard, turns: Sequence[Turn], batch: List[QACell], cue_slot: Mapping[str, str]
    ) -> List[QARecord]:
        anchor = batch[0].anchor_turn
        excerpts, allowed = self._excerpts_for(turns, anchor, {c.focus_slot for c in batch}, cue_slot)
        prompt = self._qa_prompt(persona, batch, anchor, excerpts)
        payload = self.generator.complete_json(prompt, temperature=self.opt.qa_temperature, max_tokens=4096)
        by_qid = {str(q.get("qid")): q for q in payload.get("questions") or [] if isinstance(q, dict)}

        records: List[QARecord] = []
        for cell in batch:
            rec = self._record_from_payload(cell, by_qid.get(cell.qid), allowed)
            for _ in range(2):
                if rec is not None:
                    break
                single = self.generator.complete_json(
                    self._qa_prompt(persona, [cell], anchor, excerpts),
                    temperature=self.opt.qa_temperature,
                    max_tokens=1024,
                )
                items = [q for q in single.get("questions") or [] if isinstance(q, dict)]
                rec = self._record_from_payload(cell, items[0] if items else None, allowed)
            if rec is None:
                _LOGGER.warning("[%s] %s: no valid question after retries; dropped", persona.persona_id, cell.qid)
                continue
            records.append(rec)
        return records

    def _qa_prompt(
        self,
        persona: PersonaCard,
        cells: Sequence[QACell],
        anchor: int,
        excerpts: str,
        revise: Optional[Tuple[str, str, str]] = None,
    ) -> str:
        states, previous_block = self._state_blocks(persona, anchor)
        plan = [
            {
                "qid": c.qid,
                "category": c.category.value,
                "interference": c.interference.value,
                "focus_slot": c.focus_slot,
                "qa_phase": c.qa_phase,
            }
            for c in cells
        ]
        prompt = P.QA_PROMPT.format(
            anchor_turn=anchor,
            anchor_session=self.spec.session_of_turn(anchor),
            persona_json=json.dumps(persona.brief(), ensure_ascii=False, indent=2),
            states_json=json.dumps(states, ensure_ascii=False),
            previous_states_block=previous_block,
            excerpts=excerpts,
            plan_json=json.dumps(plan, ensure_ascii=False, indent=2),
        )
        if revise is not None:
            qid, decision, feedback = revise
            prompt += P.REVISE_SUFFIX.format(qid=qid, decision=decision, feedback=feedback)
        return prompt

    def _state_blocks(self, persona: PersonaCard, anchor: int) -> Tuple[Dict[str, str], str]:
        k = sum(1 for d in self.spec.drifts if d.turn < anchor)
        states = {s.slot: s.state_at_drift(k or None) for s in persona.slots}
        if k == 0:
            return states, "No drift has happened yet.\n"
        lines = ["Earlier states that are now outdated (slot: state, valid until turn):"]
        for s in persona.slots:
            for j in sorted(s.transitions):
                if j <= k:
                    lines.append(
                        f"- {s.slot}: {s.state_at_drift(j - 1 or None)} (until turn {self.spec.drifts[j - 1].turn})"
                    )
        return states, "\n".join(lines) + "\n"

    def _excerpts_for(
        self, turns: Sequence[Turn], anchor: int, focus_slots: Iterable[str], cue_slot: Mapping[str, str]
    ) -> Tuple[str, set]:
        """Evidence candidates up to ``anchor``: cue turns of the focus slots, decoy turns, the anchor session."""
        focus = set(focus_slots)
        anchor_session = self.spec.session_of_turn(anchor)
        selected: Dict[int, Turn] = {}
        for t in turns:
            if t.turn > anchor:
                break
            on_focus = any(cue_slot.get(cid) in focus for cid in t.cue_ids)
            if t.session == anchor_session or t.tags or on_focus:
                selected[t.turn] = t
        # Keep prompts bounded: decoys and the anchor session always stay; older cue turns are thinned first.
        max_lines = 140
        if len(selected) > max_lines:
            keep = [t for t in selected.values() if t.tags or t.session == anchor_session]
            cue_turns = [t for t in selected.values() if t.turn not in {k.turn for k in keep}]
            keep.extend(cue_turns[-(max_lines - len(keep)) :])
            selected = {t.turn: t for t in keep}
        lines = []
        for t in sorted(selected.values(), key=lambda x: x.turn):
            flag = f" {{{', '.join(t.tags)}}}" if t.tags else ""
            lines.append(f"{t.dialogue_id} [{t.speaker}]{flag} {t.text}")
        return "\n".join(lines), {t.dialogue_id for t in selected.values()}

    def _record_from_payload(self, cell: QACell, raw: Optional[Dict[str, Any]], allowed: set) -> Optional[QARecord]:
        if not raw:
            return None
        question = str(raw.get("question") or "").strip()
        answer = str(raw.get("answer") or "").strip()
        evidence = [str(e).strip() for e in raw.get("evidence") or [] if str(e).strip()]
        distractors = [str(e).strip() for e in raw.get("distractor_ids") or [] if str(e).strip()]
        if not question or not answer or not evidence or len(evidence) > 4:
            return None
        if any(e not in allowed for e in evidence):
            return None
        distractors = [d for d in distractors if d in allowed]
        return QARecord(
            qid=cell.qid,
            question=question,
            answer=answer,
            phase=cell.qa_phase,
            category=cell.category,
            interference=cell.interference,
            evidence=evidence,
            anchor_turn=cell.anchor_turn,
            window=cell.window,
            focus_slot=cell.focus_slot,
            drift_index=cell.drift_index,
            rationale=str(raw.get("rationale") or ""),
            distractor_ids=distractors,
        )

    # ==================================================================
    # Stage 5
    # ==================================================================
    def stage5_verify(
        self,
        persona: PersonaCard,
        scripts: Sequence[SessionScript],
        turns: Sequence[Turn],
        qas: List[QARecord],
        stats: BuildStats,
    ) -> List[QARecord]:
        """Verify every question; REVISE regenerates with feedback, REJECT discards (Appendix D.6)."""
        ckpt = self._ckpt(persona.persona_id, "stage5_verified")
        cached = self._load(ckpt)
        if cached:
            kept = [QARecord.from_dict(q) for q in cached["qa_pairs"]]
            self._tally(kept, cached["decisions"], stats)
            return kept

        by_id = {t.dialogue_id: t for t in turns}
        cells = {c.qid: c for c in self.spec.qa_cells()}
        cue_slot = {c.cue_id: c.slot for sc in scripts for c in sc.cues}

        def verify_one(qa: QARecord) -> Tuple[Optional[QARecord], str]:
            cell = cells[qa.qid]
            current = qa
            for rnd in range(1, self.opt.rounds + 1):
                verdict = self._verify(persona, by_id, current)
                verdict.rounds = rnd
                current.verification = verdict
                if verdict.decision == ACCEPT:
                    return current, ACCEPT
                if verdict.decision == REJECT:
                    return None, REJECT
                if rnd == self.opt.rounds:
                    break
                revised = self._revise(persona, turns, cell, current, verdict, cue_slot)
                if revised is None:
                    break
                current = revised
            return (current if self.opt.keep_unresolved else None), UNRESOLVED

        with ThreadPoolExecutor(max_workers=max(1, self.opt.workers)) as pool:
            outcomes = list(pool.map(verify_one, qas))
        kept = sorted((rec for rec, _ in outcomes if rec is not None), key=lambda q: q.qid)
        decisions = {qa.qid: decision for qa, (_, decision) in zip(qas, outcomes)}
        self._tally(kept, decisions, stats)
        _LOGGER.info(
            "[%s] verification: %d accepted (%.1f%% of %d generated), %d rejected, %d unresolved after %d rounds",
            persona.persona_id,
            stats.n_accept,
            100 * stats.accept_rate,
            stats.n_qa_generated,
            stats.n_rejected,
            stats.n_unresolved,
            self.opt.rounds,
        )
        self._save(ckpt, {"qa_pairs": [q.to_dict() for q in kept], "decisions": decisions})
        return kept

    def _verify(self, persona: PersonaCard, by_id: Mapping[str, Turn], qa: QARecord) -> Verification:
        issues = self._mechanical_issues(qa, by_id)
        if issues:
            return Verification(
                decision=REVISE, scores={}, feedback="; ".join(issues), verifier="mechanical", mechanical_issues=issues
            )
        states, previous_block = self._state_blocks(persona, qa.anchor_turn)
        payload = self.verifier.complete_json(
            P.VERIFIER_PROMPT.format(
                persona_summary=persona.summary,
                anchor_turn=qa.anchor_turn,
                states_json=json.dumps(states, ensure_ascii=False),
                previous_states_block=previous_block,
                plan_json=json.dumps(
                    {
                        "category": qa.category.value,
                        "interference": qa.interference.value,
                        "focus_slot": qa.focus_slot,
                        "qa_phase": qa.phase,
                    },
                    ensure_ascii=False,
                ),
                excerpts=self._context_excerpts(by_id, qa),
                qa_json=json.dumps(
                    {
                        "question": qa.question,
                        "answer": qa.answer,
                        "evidence": qa.evidence,
                        "distractor_ids": qa.distractor_ids,
                        "rationale": qa.rationale,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
            ),
            temperature=self.opt.verifier_temperature,
            max_tokens=512,
        )
        scores = (
            {k: int(bool(payload.get("scores", {}).get(k, 0))) for k in CRITERIA}
            if isinstance(payload.get("scores"), dict)
            else {}
        )
        decision = str(payload.get("decision") or "").upper().strip()
        if decision not in {ACCEPT, REVISE, REJECT}:
            decision = ACCEPT if scores and all(scores.values()) else REVISE
        return Verification(
            decision=decision, scores=scores, feedback=str(payload.get("feedback") or ""), verifier=self.verifier.model
        )

    @staticmethod
    def _mechanical_issues(qa: QARecord, by_id: Mapping[str, Turn]) -> List[str]:
        issues = []
        for e in qa.evidence:
            t = by_id.get(e)
            if t is None:
                issues.append(f"evidence {e} does not exist")
            elif t.turn > qa.anchor_turn:
                issues.append(f"evidence {e} (turn {t.turn}) is after the anchor turn {qa.anchor_turn}")
        if not qa.answer.strip():
            issues.append("empty answer")
        return issues

    @staticmethod
    def _context_excerpts(by_id: Mapping[str, Turn], qa: QARecord) -> str:
        wanted: Dict[int, Turn] = {}
        ordered = sorted(by_id.values(), key=lambda t: t.turn)
        pos = {t.dialogue_id: i for i, t in enumerate(ordered)}
        for did in [*qa.evidence, *qa.distractor_ids]:
            i = pos.get(did)
            if i is None:
                continue
            for j in (i - 1, i, i + 1):
                if 0 <= j < len(ordered) and ordered[j].turn <= qa.anchor_turn:
                    wanted[ordered[j].turn] = ordered[j]
        return "\n".join(
            f"{t.dialogue_id} [{t.speaker}] {t.text}" for t in sorted(wanted.values(), key=lambda x: x.turn)
        )

    def _revise(
        self,
        persona: PersonaCard,
        turns: Sequence[Turn],
        cell: QACell,
        qa: QARecord,
        verdict: Verification,
        cue_slot: Mapping[str, str],
    ) -> Optional[QARecord]:
        """Ask the generator to rewrite one question using the verifier's feedback."""
        excerpts, allowed = self._excerpts_for(turns, cell.anchor_turn, {cell.focus_slot}, cue_slot)
        payload = self.generator.complete_json(
            self._qa_prompt(
                persona, [cell], cell.anchor_turn, excerpts, revise=(cell.qid, verdict.decision, verdict.feedback)
            ),
            temperature=self.opt.qa_temperature,
            max_tokens=1024,
        )
        items = [q for q in payload.get("questions") or [] if isinstance(q, dict)]
        return self._record_from_payload(cell, items[0] if items else None, allowed)

    @staticmethod
    def _tally(kept: Sequence[QARecord], decisions: Mapping[str, str], stats: BuildStats) -> None:
        counts = Counter(decisions.values())
        stats.n_accept = counts.get(ACCEPT, 0)
        stats.n_rejected = counts.get(REJECT, 0)
        stats.n_unresolved = counts.get(UNRESOLVED, 0)
        rounds = Counter(q.verification.rounds for q in kept if q.verification.decision == ACCEPT)
        stats.accept_by_round = dict(sorted(rounds.items()))

    # ==================================================================
    # Output
    # ==================================================================
    def assemble(
        self,
        persona: PersonaCard,
        scripts: Sequence[SessionScript],
        turns: Sequence[Turn],
        qas: Sequence[QARecord],
        stats: BuildStats,
    ) -> Dict[str, Any]:
        realized: Dict[str, str] = {}
        for t in turns:
            for cid in t.cue_ids:
                realized[cid] = t.dialogue_id
        cues = []
        for s in scripts:
            for c in s.cues:
                d = c.to_dict()
                d["dialogue_id"] = realized.get(c.cue_id)
                cues.append(d)
        return {
            "benchmark": self.spec.name,
            "persona_id": persona.persona_id,
            "persona": persona.to_dict(),
            "spec": self.spec.describe(),
            "drift_turns": self.spec.drift_turns,
            "slots": state_timeline(persona, self.spec.drift_turns, self.spec.n_turns),
            "sessions": [s.to_dict() for s in scripts],
            "cues": cues,
            "dialogue": [t.to_dict() for t in turns],
            "qa_pairs": [q.to_dict() for q in qas],
            "stats": stats.to_dict(),
            "metadata": {
                "generator_model": self.generator.model,
                "verifier_model": self.verifier.model,
                "options": dict(self.opt.__dict__),
                "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
        }


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------


def _as_index(value: Any, n: int) -> Optional[int]:
    try:
        k = int(value)
    except (TypeError, ValueError):
        return None
    return k if 1 <= k <= n else None


def _fallback_circle(hints: Sequence[str]) -> List[Character]:
    names = [str(h) for h in hints][:3] or ["Leo", "Leon", "Mia"]
    circle = [Character(name=n, relation="friend") for n in names]
    if len(circle) >= 2:
        circle[0].confusable_with = circle[1].name
        circle[1].confusable_with = circle[0].name
    return circle
