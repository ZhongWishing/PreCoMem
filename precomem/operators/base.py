"""Shared types for the three memory-evolution operators.

Every operator returns a typed outcome rather than a loose dictionary. The
agent uses :meth:`OperatorOutcome.supported_ids` to know which items received
supporting evidence this turn (they skip hypothesis decay) and
:meth:`OperatorOutcome.written_ids` to attach provenance pointers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..memory.items import HypothesisClass, MemoryItem
from ..retrieval.retriever import RetrievalResult


@dataclass
class WeightUpdate:
    """One closed-form update applied to an existing item."""

    item_id: str
    old_weight: float
    new_weight: float
    scalar: float  # η_t(m) for MAINTAIN, ζ_t(m) for CORRECT
    reason: str = ""
    operation: str = ""  # CORRECT only: decrease_confidence | archive_and_replace

    def to_dict(self) -> Dict[str, Any]:
        out = {
            "id": self.item_id,
            "old_weight": round(self.old_weight, 4),
            "new_weight": round(self.new_weight, 4),
            "scalar": round(self.scalar, 4),
            "reason": self.reason,
        }
        if self.operation:
            out["operation"] = self.operation
        return out


@dataclass
class Refinement:
    parent_id: str
    kind: str
    detail: str

    def to_dict(self) -> Dict[str, Any]:
        return {"parent": self.parent_id, "type": self.kind, "detail": self.detail}


class OperatorOutcome:
    """Base class; subclasses are plain dataclasses."""

    def supported_ids(self) -> List[str]:
        return []

    def written_ids(self) -> List[str]:
        return []

    def retired(self) -> List[MemoryItem]:
        """Items archived this turn (feeds the DT lifespan statistic)."""
        return []

    def to_dict(self) -> Dict[str, Any]:
        return {}


@dataclass
class MaintainOutcome(OperatorOutcome):
    updates: List[WeightUpdate] = field(default_factory=list)
    refinements: List[Refinement] = field(default_factory=list)
    agent_payload: Dict[str, Any] = field(default_factory=dict)

    def supported_ids(self) -> List[str]:
        return [u.item_id for u in self.updates]

    def written_ids(self) -> List[str]:
        return self.supported_ids()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "updates": [u.to_dict() for u in self.updates],
            "refinements": [r.to_dict() for r in self.refinements],
        }


@dataclass
class ProfileOutcome(OperatorOutcome):
    hypothesis: Optional[MemoryItem]
    classification: HypothesisClass
    prior: float  # b_t^{(c)} in Eq. (8)
    agent_confidence: float  # clip(q_t, I_c)
    initial_weight: float  # w_0(h)
    agent_payload: Dict[str, Any] = field(default_factory=dict)

    def written_ids(self) -> List[str]:
        return [self.hypothesis.item_id] if self.hypothesis is not None else []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hypothesis": self.hypothesis.to_dict() if self.hypothesis is not None else None,
            "classification": self.classification.value,
            "prior": round(self.prior, 4),
            "agent_confidence": round(self.agent_confidence, 4),
            "initial_weight": round(self.initial_weight, 4),
        }


@dataclass
class CorrectOutcome(OperatorOutcome):
    updates: List[WeightUpdate] = field(default_factory=list)
    archived: List[MemoryItem] = field(default_factory=list)
    new_fact: Optional[MemoryItem] = None
    diagnosis: str = ""
    conflict_type: str = ""
    agent_payload: Dict[str, Any] = field(default_factory=dict)

    def written_ids(self) -> List[str]:
        return [self.new_fact.item_id] if self.new_fact is not None else []

    def retired(self) -> List[MemoryItem]:
        return list(self.archived)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "updates": [u.to_dict() for u in self.updates],
            "archived": [m.item_id for m in self.archived],
            "new_fact": self.new_fact.to_dict() if self.new_fact is not None else None,
            "diagnosis": self.diagnosis,
            "conflict_type": self.conflict_type,
        }


@dataclass
class NoOpOutcome(OperatorOutcome):
    """Returned when an operator has nothing to act on (e.g. empty evidence)."""

    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"noop": True, "reason": self.reason}


def format_evidence(retrieval: RetrievalResult, with_scores: bool = False) -> str:
    """Render ``C_t`` for a sub-agent prompt (ids are needed so the agent can target items)."""
    if retrieval.is_empty:
        return "(no relevant prior beliefs retrieved)"
    lines = []
    for m, score in zip(retrieval.items, retrieval.scores):
        extra = f" score={score:.3f}" if with_scores else ""
        lines.append(f'- id={m.item_id} type={m.mtype.value} weight={m.weight:.2f}{extra} content="{m.content}"')
    return "\n".join(lines)


def records_with_ids(payload: Any, key: str) -> List[Dict[str, Any]]:
    """Return the list under ``payload[key]`` keeping only dict entries with a string id."""
    if not isinstance(payload, dict):
        return []
    entries = payload.get(key) or []
    if not isinstance(entries, list):
        return []
    return [e for e in entries if isinstance(e, dict) and isinstance(e.get("id"), str)]
