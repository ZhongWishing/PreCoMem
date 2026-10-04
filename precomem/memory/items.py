"""Memory primitives: belief items and episodic traces.

A :class:`MemoryItem` is a node of the belief store ``B_t`` (Section 2.2 of
the paper). It carries natural-language content ``c(m)``, a type
``tau(m) in {FACT, HYPOTHESIS}``, a confidence weight ``w_t(m) in [0, 1]`` and
the retrieval metadata described in Appendix C.2 ("Memory item fields"): keywords
proposed by the Profile-Agent (used only by the lexical term ``s_fact``), a
topic label assigned at write time (used only by the topic prior
``s_topic``), a dense embedding of the content alone, and provenance
pointers into the episodic cache ``E_t``.

An :class:`EpisodicTrace` is an immutable record of one dialogue exchange.
Traces are fixed at ``w(e) == 1`` and never enter confidence-weighted
Top-K ranking; they are attached to retrieved beliefs as provenance.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np


class MemoryType(str, Enum):
    FACT = "FACT"
    HYPOTHESIS = "HYPOTHESIS"


class HypothesisClass(str, Enum):
    """Class label ``c`` returned by the Profile-Agent (selects the band ``I_c``)."""

    EXPLICIT = "explicit_event"
    LATENT = "latent_inference"

    @classmethod
    def parse(cls, raw: Any) -> HypothesisClass:
        text = str(raw or "").strip().lower()
        if text.startswith("explicit") or text in {"a", "type a", "type_a"}:
            return cls.EXPLICIT
        return cls.LATENT


def _embedding_to_list(vec: Optional[np.ndarray]) -> Optional[List[float]]:
    if vec is None:
        return None
    return [float(x) for x in np.asarray(vec, dtype=np.float32).tolist()]


def _embedding_from_list(values: Any) -> Optional[np.ndarray]:
    if values is None:
        return None
    return np.asarray(values, dtype=np.float32)


@dataclass
class EpisodicTrace:
    """One (user, assistant) exchange written back to ``E_t``.

    ``session``/``dialogue_id``/``timestamp`` are optional benchmark metadata
    (e.g. LoCoMo's ``D1:3`` identifiers and session dates); they are
    surfaced verbatim in QA prompts so that temporal reasoning can use them.
    """

    trace_id: str
    turn: int
    user: str
    assistant: str = ""
    speaker: Optional[str] = None
    session: Optional[str] = None
    dialogue_id: Optional[str] = None
    timestamp: Optional[str] = None
    wall_time: float = field(default_factory=time.time)
    embedding: Optional[np.ndarray] = field(default=None, repr=False, compare=False)

    @property
    def text(self) -> str:
        """Plain text used for embedding and for prompt rendering."""
        user_line = f"{self.speaker}: {self.user}" if self.speaker else f"User: {self.user}"
        if self.assistant:
            return f"{user_line}\nAssistant: {self.assistant}"
        return user_line

    def render(self) -> str:
        """Compact one-line rendering with identifiers, for QA prompts."""
        tag = self.dialogue_id or f"T{self.turn}"
        stamp = f" ({self.timestamp})" if self.timestamp else ""
        head = f"[{tag}]{stamp}"
        who = self.speaker or "User"
        line = f"{head} {who}: {self.user}"
        if self.assistant:
            line += f"\n{head} Assistant: {self.assistant}"
        return line

    def to_dict(self, include_embedding: bool = False) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "trace_id": self.trace_id,
            "turn": self.turn,
            "user": self.user,
            "assistant": self.assistant,
            "speaker": self.speaker,
            "session": self.session,
            "dialogue_id": self.dialogue_id,
            "timestamp": self.timestamp,
            "wall_time": self.wall_time,
        }
        if include_embedding:
            out["embedding"] = _embedding_to_list(self.embedding)
        return out

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EpisodicTrace:
        return cls(
            trace_id=str(data["trace_id"]),
            turn=int(data["turn"]),
            user=str(data.get("user", "")),
            assistant=str(data.get("assistant", "") or ""),
            speaker=data.get("speaker"),
            session=data.get("session"),
            dialogue_id=data.get("dialogue_id"),
            timestamp=data.get("timestamp"),
            wall_time=float(data.get("wall_time", time.time())),
            embedding=_embedding_from_list(data.get("embedding")),
        )


@dataclass
class MemoryItem:
    """A typed belief node in ``B_t``.

    Attributes
    ----------
    content:
        Proposition ``c(m)`` shown to the response model.
    mtype:
        ``FACT`` for committed beliefs, ``HYPOTHESIS`` for the provisional
        beliefs written by PROFILE. Transitions are one-way (promotion).
    weight:
        Operational support score ``w_t(m) in [0, 1]``.
    created_turn:
        Turn at which the item was written.
    last_supported_turn:
        ``l(h)`` in Eq. (9): the turn of creation or of the last *supporting*
        evidence (a MAINTAIN reinforcement). CORRECT decays do not move it.
    last_updated_turn:
        Bookkeeping: the last turn any operator touched the item.
    decay_protection:
        Explicit-event hypotheses are exempt from the time-based decay of
        Eq. (9), as flagged by the Profile-Agent.
    classification:
        The Profile-Agent class label for hypotheses; ``None`` for facts
        written by CORRECT.
    keywords:
        Discriminative keywords proposed at write time; they enter only the
        BM25 term ``s_fact``.
    topic:
        Topic label ``z(m)`` assigned at write time; enters only ``s_topic``.
    sources:
        Trace ids in ``E_t`` that support this item (provenance).
    deprecated:
        Set when the Correct-Agent archives the item in favor of a fresh
        belief node. Deprecated items are kept at low confidence and are
        deprioritised by confidence-weighted retrieval rather than deleted.
    """

    content: str
    mtype: MemoryType = MemoryType.FACT
    weight: float = 0.5
    item_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_turn: int = 0
    last_supported_turn: int = 0
    last_updated_turn: int = 0
    decay_protection: bool = False
    classification: Optional[HypothesisClass] = None
    keywords: List[str] = field(default_factory=list)
    topic: Optional[str] = None
    sources: List[str] = field(default_factory=list)
    deprecated: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.weight = float(min(max(self.weight, 0.0), 1.0))

    # -- helpers ------------------------------------------------------
    @property
    def is_hypothesis(self) -> bool:
        return self.mtype is MemoryType.HYPOTHESIS

    @property
    def is_fact(self) -> bool:
        return self.mtype is MemoryType.FACT

    def set_weight(self, value: float, turn: int) -> None:
        self.weight = float(min(max(value, 0.0), 1.0))
        self.last_updated_turn = turn

    def mark_supported(self, turn: int) -> None:
        """Reset the decay anchor ``l(h)`` after supporting evidence."""
        self.last_supported_turn = turn
        self.last_updated_turn = turn

    def age_since_support(self, turn: int) -> int:
        """``a_t(h) = t - l(h)`` from Eq. (9)."""
        return max(0, turn - self.last_supported_turn)

    def lexical_text(self) -> str:
        """Content plus keywords, the document seen by the BM25 term."""
        if not self.keywords:
            return self.content
        return f"{self.content} {' '.join(self.keywords)}"

    def add_source(self, trace_id: str) -> None:
        if trace_id not in self.sources:
            self.sources.append(trace_id)

    # -- (de)serialization --------------------------------------------
    def to_dict(self, include_embedding: bool = False) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "item_id": self.item_id,
            "content": self.content,
            "type": self.mtype.value,
            "weight": round(float(self.weight), 4),
            "created_turn": self.created_turn,
            "last_supported_turn": self.last_supported_turn,
            "last_updated_turn": self.last_updated_turn,
            "decay_protection": self.decay_protection,
            "classification": self.classification.value if self.classification else None,
            "keywords": list(self.keywords),
            "topic": self.topic,
            "sources": list(self.sources),
            "deprecated": self.deprecated,
            "metadata": dict(self.metadata),
        }
        if include_embedding:
            out["embedding"] = _embedding_to_list(self.embedding)
        return out

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MemoryItem:
        classification = data.get("classification")
        return cls(
            content=str(data["content"]),
            mtype=MemoryType(data.get("type", MemoryType.FACT.value)),
            weight=float(data.get("weight", 0.5)),
            item_id=str(data.get("item_id") or uuid.uuid4().hex[:12]),
            created_turn=int(data.get("created_turn", 0)),
            last_supported_turn=int(data.get("last_supported_turn", data.get("created_turn", 0))),
            last_updated_turn=int(data.get("last_updated_turn", data.get("created_turn", 0))),
            decay_protection=bool(data.get("decay_protection", False)),
            classification=HypothesisClass(classification) if classification else None,
            keywords=[str(k) for k in data.get("keywords", [])],
            topic=data.get("topic"),
            sources=[str(s) for s in data.get("sources", [])],
            deprecated=bool(data.get("deprecated", False)),
            metadata=dict(data.get("metadata", {})),
            embedding=_embedding_from_list(data.get("embedding")),
        )
