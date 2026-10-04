"""DiMoBench / DiMoBench-XL loader.

Reads the persona files produced by ``precomem-build-dimobench``::

    {
        "benchmark": "dimobench",
        "persona_id": "persona_a",
        "drift_turns": [200],
        "dialogue": [
            {"turn": 1, "session": 1, "turn_in_session": 1, "dialogue_id": "D1:1",
             "speaker": "user", "text": "...", "phase": "pre"}, ...
        ],
        "qa_pairs": [
            {"qid": "...", "question": "...", "answer": "...", "phase": "post-change",
             "category": "current-state inference", "interference": "temporal_decoy",
             "evidence": ["D7:5", "D13:9"], "anchor_turn": 260, ...}
        ]
    }

``anchor_turn`` is the dialogue position after which a question is asked;
the runner evaluates anchored questions mid-dialogue so that answers reflect
the belief state *at that time*. Questions without an anchor are asked after
the whole dialogue, and a missing ``dialogue_id`` is reconstructed from
``session`` / ``turn_in_session``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .base import BenchmarkSample, DialogueTurn, QAPair, infer_role, load_json, unwrap_records


class DiMoBenchLoader:
    name = "dimobench"

    @staticmethod
    def load(path: Union[str, Path]) -> List[BenchmarkSample]:
        """Load one persona file, a list of personas, or a ``suite.json`` listing persona files."""
        base_dir = Path(path).parent
        samples: List[BenchmarkSample] = []
        for idx, persona in enumerate(unwrap_records(load_json(path), "personas", "data")):
            if isinstance(persona, str):
                ref = Path(persona)
                persona = load_json(ref if ref.is_absolute() or ref.exists() else base_dir / ref.name)
            samples.append(parse_persona(persona, fallback_id=str(idx)))
        return samples


def parse_persona(persona: Dict[str, Any], fallback_id: str = "0") -> BenchmarkSample:
    sample_id = str(persona.get("persona_id") or persona.get("id") or fallback_id)

    turns: List[DialogueTurn] = []
    within_session: Dict[str, int] = {}
    for entry in persona.get("dialogue", []):
        turn = int(entry.get("turn", len(turns) + 1))
        session = str(entry["session"]) if entry.get("session") is not None else None
        speaker = str(entry.get("speaker") or "user")
        turns.append(
            DialogueTurn(
                index=turn,
                speaker=speaker,
                text=str(entry.get("text", "")),
                role=infer_role(speaker),
                session=session,
                timestamp=entry.get("timestamp"),
                dialogue_id=_dialogue_id(entry, session, within_session),
                meta={"phase": entry.get("phase")},
            )
        )

    qa_pairs: List[QAPair] = []
    for j, q in enumerate(persona.get("qa_pairs") or []):
        qa_pairs.append(
            QAPair(
                qid=str(q.get("qid") or f"{sample_id}::Q{j:04d}"),
                question=str(q.get("question", "")),
                answer=str(q.get("answer", "")),
                category=q.get("category"),
                evidence=[str(e) for e in q.get("evidence") or []],
                meta={
                    "phase": q.get("phase"),
                    "interference": q.get("interference"),
                    "anchor_turn": q.get("anchor_turn"),
                    "window": q.get("window"),
                    "audit": bool(q.get("audit", False)),
                },
            )
        )

    return BenchmarkSample(
        sample_id=sample_id,
        dialogue=turns,
        qa_pairs=qa_pairs,
        meta={
            "source": persona.get("benchmark", "dimobench"),
            "drift_turns": list(persona.get("drift_turns", [])),
            "slots": persona.get("slots", []),
        },
    )


def _dialogue_id(entry: Dict[str, Any], session: Optional[str], within_session: Dict[str, int]) -> str:
    if entry.get("dialogue_id"):
        return str(entry["dialogue_id"])
    key = session or "0"
    within_session[key] = within_session.get(key, 0) + 1
    return f"D{key}:{entry.get('turn_in_session') or within_session[key]}"
