"""Data records exchanged between the DiMoBench pipeline stages.

All records are plain dataclasses with ``to_dict``/``from_dict`` so that each
stage can be checkpointed to JSON and the build resumed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .spec import Category, CueKind, Interference, Phase


@dataclass
class SlotSpec:
    """One attribute slot of a persona and how it changes across drifts."""

    slot: str
    initial_state: str
    # drift index (1-based) -> new state; empty for slots that never change
    transitions: Dict[int, str] = field(default_factory=dict)

    def state_at_drift(self, drift_index: Optional[int]) -> str:
        """State after the given drift (None / 0 = before any drift)."""
        state = self.initial_state
        for k in sorted(self.transitions):
            if drift_index is not None and k <= drift_index:
                state = self.transitions[k]
        return state

    def to_dict(self) -> Dict[str, Any]:
        return {
            "slot": self.slot,
            "initial_state": self.initial_state,
            "transitions": {str(k): v for k, v in self.transitions.items()},
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> SlotSpec:
        return cls(
            slot=str(d["slot"]),
            initial_state=str(d["initial_state"]),
            transitions={int(k): str(v) for k, v in (d.get("transitions") or {}).items()},
        )


@dataclass
class Character:
    name: str
    relation: str
    confusable_with: Optional[str] = None  # name this character is designed to be confused with

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> Character:
        return cls(name=str(d["name"]), relation=str(d.get("relation", "")), confusable_with=d.get("confusable_with"))


@dataclass
class PersonaCard:
    """Stage-1 output: a fully specified user persona."""

    persona_id: str
    name: str
    summary: str
    life_stage: str
    big_five: Dict[str, float]
    social_circle: List[Character]
    routine: List[str]
    speaking_style: str
    slots: List[SlotSpec]
    drift_triggers: Dict[int, str]  # drift index -> concrete trigger event
    seed: Dict[str, Any] = field(default_factory=dict)

    def slot(self, name: str) -> SlotSpec:
        for s in self.slots:
            if s.slot == name:
                return s
        raise KeyError(name)

    def confusable_names(self) -> List[str]:
        return [c.name for c in self.social_circle]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "persona_id": self.persona_id,
            "name": self.name,
            "summary": self.summary,
            "life_stage": self.life_stage,
            "big_five": dict(self.big_five),
            "social_circle": [c.to_dict() for c in self.social_circle],
            "routine": list(self.routine),
            "speaking_style": self.speaking_style,
            "slots": [s.to_dict() for s in self.slots],
            "drift_triggers": {str(k): v for k, v in self.drift_triggers.items()},
            "seed": dict(self.seed),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> PersonaCard:
        return cls(
            persona_id=str(d["persona_id"]),
            name=str(d.get("name", d["persona_id"])),
            summary=str(d.get("summary", "")),
            life_stage=str(d.get("life_stage", "")),
            big_five={str(k): float(v) for k, v in (d.get("big_five") or {}).items()},
            social_circle=[Character.from_dict(c) for c in d.get("social_circle", [])],
            routine=[str(r) for r in d.get("routine", [])],
            speaking_style=str(d.get("speaking_style", "")),
            slots=[SlotSpec.from_dict(s) for s in d.get("slots", [])],
            drift_triggers={int(k): str(v) for k, v in (d.get("drift_triggers") or {}).items()},
            seed=dict(d.get("seed") or {}),
        )

    def brief(self) -> Dict[str, Any]:
        """Compact view for prompts (no seed blob)."""
        return {
            "name": self.name,
            "summary": self.summary,
            "life_stage": self.life_stage,
            "big_five": self.big_five,
            "social_circle": [c.to_dict() for c in self.social_circle],
            "routine": self.routine,
            "speaking_style": self.speaking_style,
        }


@dataclass
class CueScript:
    """Stage-2 output: a concrete, implicit event the dialogue must express."""

    cue_id: str
    session: int
    slot: str
    kind: CueKind
    drift_index: Optional[int]
    state_to_express: str  # the (current) slot state the cue should reveal implicitly
    description: str  # what happens, in behavioral terms
    surface_hint: str = ""  # e.g. "declines an invitation citing work"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["kind"] = self.kind.value
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> CueScript:
        return cls(
            cue_id=str(d["cue_id"]),
            session=int(d["session"]),
            slot=str(d["slot"]),
            kind=CueKind(d["kind"]),
            drift_index=d.get("drift_index"),
            state_to_express=str(d.get("state_to_express", "")),
            description=str(d.get("description", "")),
            surface_hint=str(d.get("surface_hint", "")),
        )


@dataclass
class SessionScript:
    """Stage-2 output for one session: summary, events and the cue scripts."""

    index: int
    first_turn: int
    last_turn: int
    phase: Phase
    drift_index: Optional[int]
    summary: str
    setting: str  # time/place framing ("a Tuesday evening, two weeks later")
    cues: List[CueScript]
    confounder: Optional[CueKind]
    confounder_script: str = ""  # what the confounder chit-chat / hypothetical is about
    mention_confusable_entity: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "first_turn": self.first_turn,
            "last_turn": self.last_turn,
            "phase": self.phase.value,
            "drift_index": self.drift_index,
            "summary": self.summary,
            "setting": self.setting,
            "cues": [c.to_dict() for c in self.cues],
            "confounder": self.confounder.value if self.confounder else None,
            "confounder_script": self.confounder_script,
            "mention_confusable_entity": self.mention_confusable_entity,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> SessionScript:
        return cls(
            index=int(d["index"]),
            first_turn=int(d["first_turn"]),
            last_turn=int(d["last_turn"]),
            phase=Phase(d["phase"]),
            drift_index=d.get("drift_index"),
            summary=str(d.get("summary", "")),
            setting=str(d.get("setting", "")),
            cues=[CueScript.from_dict(c) for c in d.get("cues", [])],
            confounder=CueKind(d["confounder"]) if d.get("confounder") else None,
            confounder_script=str(d.get("confounder_script", "")),
            mention_confusable_entity=bool(d.get("mention_confusable_entity", False)),
        )


@dataclass
class Turn:
    """Stage-3 output: one dialogue turn."""

    turn: int  # global 1-based index
    session: int
    turn_in_session: int  # 1-based within the session
    speaker: str  # "user" | "assistant"
    text: str
    phase: Phase
    cue_ids: List[str] = field(default_factory=list)  # cues realized in this turn
    tags: List[str] = field(default_factory=list)  # "keyword_trap", "hypothetical", "entity_decoy"

    @property
    def dialogue_id(self) -> str:
        return f"D{self.session}:{self.turn_in_session}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn": self.turn,
            "session": self.session,
            "turn_in_session": self.turn_in_session,
            "dialogue_id": self.dialogue_id,
            "speaker": self.speaker,
            "text": self.text,
            "phase": self.phase.value,
            "cue_ids": list(self.cue_ids),
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> Turn:
        return cls(
            turn=int(d["turn"]),
            session=int(d["session"]),
            turn_in_session=int(d["turn_in_session"]),
            speaker=str(d["speaker"]),
            text=str(d["text"]),
            phase=Phase(d["phase"]),
            cue_ids=[str(c) for c in d.get("cue_ids", [])],
            tags=[str(t) for t in d.get("tags", [])],
        )


@dataclass
class Verification:
    decision: str = "PENDING"  # ACCEPT | REVISE | REJECT | PENDING
    scores: Dict[str, int] = field(default_factory=dict)
    feedback: str = ""
    rounds: int = 0
    verifier: str = ""
    mechanical_issues: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> Verification:
        return cls(
            decision=str(d.get("decision", "PENDING")),
            scores={str(k): int(v) for k, v in (d.get("scores") or {}).items()},
            feedback=str(d.get("feedback", "")),
            rounds=int(d.get("rounds", 0)),
            verifier=str(d.get("verifier", "")),
            mechanical_issues=[str(x) for x in d.get("mechanical_issues", [])],
        )


@dataclass
class QARecord:
    """Stage-4/5 output: one question with its gold answer and provenance."""

    qid: str
    question: str
    answer: str
    phase: str  # pre-change | change-point | post-change
    category: Category
    interference: Interference
    evidence: List[str]  # dialogue ids D{session}:{turn}
    anchor_turn: int
    window: int
    focus_slot: str
    drift_index: Optional[int]
    rationale: str = ""
    distractor_ids: List[str] = field(default_factory=list)  # decoy turns the question is designed around
    verification: Verification = field(default_factory=Verification)
    audit: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "qid": self.qid,
            "question": self.question,
            "answer": self.answer,
            "phase": self.phase,
            "category": self.category.value,
            "interference": self.interference.value,
            "evidence": list(self.evidence),
            "anchor_turn": self.anchor_turn,
            "window": self.window,
            "focus_slot": self.focus_slot,
            "drift_index": self.drift_index,
            "rationale": self.rationale,
            "distractor_ids": list(self.distractor_ids),
            "verification": self.verification.to_dict(),
            "audit": self.audit,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> QARecord:
        return cls(
            qid=str(d["qid"]),
            question=str(d["question"]),
            answer=str(d["answer"]),
            phase=str(d["phase"]),
            category=Category(d["category"]),
            interference=Interference(d["interference"]),
            evidence=[str(e) for e in d.get("evidence", [])],
            anchor_turn=int(d["anchor_turn"]),
            window=int(d.get("window", 0)),
            focus_slot=str(d.get("focus_slot", "")),
            drift_index=d.get("drift_index"),
            rationale=str(d.get("rationale", "")),
            distractor_ids=[str(x) for x in d.get("distractor_ids", [])],
            verification=Verification.from_dict(d.get("verification") or {}),
            audit=bool(d.get("audit", False)),
        )


def state_timeline(persona: PersonaCard, drift_turns: Sequence[int], n_turns: int) -> List[Dict[str, Any]]:
    """Gold state of every slot over turn ranges (change points are the drift turns)."""
    out: List[Dict[str, Any]] = []
    boundaries = [0, *drift_turns, n_turns]
    for slot in persona.slots:
        segments = []
        for k in range(len(boundaries) - 1):
            segments.append(
                {"from_turn": boundaries[k] + 1, "to_turn": boundaries[k + 1], "state": slot.state_at_drift(k or None)}
            )
        out.append({"slot": slot.slot, "segments": segments})
    return out
