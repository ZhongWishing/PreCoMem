"""Memory state ``M_t = (B_t, E_t, T_t)`` and its three containers.

* :class:`BeliefStore` — the editable belief store ``B_t``.
* :class:`EpisodicCache` — the immutable, fixed-capacity dialogue cache ``E_t``.
* :class:`TopicTable` — the optional topic table ``T_t`` (a retrieval prior).

All three serialize to plain JSON so that a memory snapshot can be dumped at
any turn for inspection or regression testing, and restored later.
"""

from __future__ import annotations

import json
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, Iterator, List, Optional, Union

import numpy as np

from ..utils.numeric import cosine_sim, softmax
from .items import EpisodicTrace, MemoryItem, MemoryType


class BeliefStore:
    """Editable belief store ``B_t``.

    The store keeps insertion order and exposes a monotone ``version``
    counter that changes whenever the set of items *or any item's text*
    changes. Retrieval uses it to decide when its lexical index is stale.
    Weight updates do not bump the version because BM25 statistics do not
    depend on weights.
    """

    def __init__(self) -> None:
        self._items: OrderedDict[str, MemoryItem] = OrderedDict()
        self.version: int = 0

    # -- mutation -----------------------------------------------------
    def add(self, item: MemoryItem) -> MemoryItem:
        self._items[item.item_id] = item
        self.version += 1
        return item

    def remove(self, item_id: str) -> Optional[MemoryItem]:
        removed = self._items.pop(item_id, None)
        if removed is not None:
            self.version += 1
        return removed

    def update_content(self, item_id: str, content: str, turn: int) -> Optional[MemoryItem]:
        """Replace an item's text and invalidate its cached embedding."""
        item = self._items.get(item_id)
        if item is None:
            return None
        item.content = content
        item.embedding = None
        item.last_updated_turn = turn
        self.version += 1
        return item

    # -- access -------------------------------------------------------
    def get(self, item_id: str) -> Optional[MemoryItem]:
        return self._items.get(item_id)

    def __contains__(self, item_id: object) -> bool:
        return item_id in self._items

    def __iter__(self) -> Iterator[MemoryItem]:
        return iter(list(self._items.values()))

    def __len__(self) -> int:
        return len(self._items)

    def items(self) -> List[MemoryItem]:
        return list(self._items.values())

    def facts(self) -> List[MemoryItem]:
        return [m for m in self._items.values() if m.mtype is MemoryType.FACT]

    def hypotheses(self) -> List[MemoryItem]:
        return [m for m in self._items.values() if m.mtype is MemoryType.HYPOTHESIS]

    def effective_size(self) -> float:
        """``sum_m w_t(m)``, the effective-memory measure used in the paper."""
        return float(sum(m.weight for m in self._items.values()))

    # -- (de)serialization --------------------------------------------
    def to_list(self, include_embeddings: bool = False) -> List[Dict[str, Any]]:
        return [m.to_dict(include_embedding=include_embeddings) for m in self._items.values()]

    @classmethod
    def from_list(cls, records: Iterable[Dict[str, Any]]) -> BeliefStore:
        store = cls()
        for rec in records:
            store.add(MemoryItem.from_dict(rec))
        return store


class EpisodicCache:
    """Fixed-capacity cache ``E_t`` of immutable dialogue traces.

    When the capacity is exceeded the oldest trace is evicted; beliefs that
    still point to it keep the dangling id, which simply resolves to no
    excerpt at prompt-rendering time.
    """

    def __init__(self, capacity: int = 20000) -> None:
        if capacity < 1:
            raise ValueError("episodic capacity must be >= 1")
        self.capacity = int(capacity)
        self._traces: Deque[EpisodicTrace] = deque()
        self._index: Dict[str, EpisodicTrace] = {}

    def append(self, trace: EpisodicTrace) -> EpisodicTrace:
        if trace.trace_id in self._index:
            raise ValueError(f"duplicate trace id {trace.trace_id!r}")
        if len(self._traces) >= self.capacity:
            evicted = self._traces.popleft()
            self._index.pop(evicted.trace_id, None)
        self._traces.append(trace)
        self._index[trace.trace_id] = trace
        return trace

    def get(self, trace_id: str) -> Optional[EpisodicTrace]:
        return self._index.get(trace_id)

    def resolve(self, trace_ids: Iterable[str]) -> List[EpisodicTrace]:
        """Look up ids, dropping those already evicted; preserves order, dedups."""
        seen: set = set()
        out: List[EpisodicTrace] = []
        for tid in trace_ids:
            trace = self._index.get(tid)
            if trace is not None and tid not in seen:
                seen.add(tid)
                out.append(trace)
        return out

    def recent(self, n: int) -> List[EpisodicTrace]:
        if n <= 0:
            return []
        return list(self._traces)[-n:]

    def __iter__(self) -> Iterator[EpisodicTrace]:
        return iter(list(self._traces))

    def __len__(self) -> int:
        return len(self._traces)

    def to_list(self, include_embeddings: bool = False) -> List[Dict[str, Any]]:
        return [t.to_dict(include_embedding=include_embeddings) for t in self._traces]

    @classmethod
    def from_list(cls, records: Iterable[Dict[str, Any]], capacity: int = 20000) -> EpisodicCache:
        cache = cls(capacity=capacity)
        for rec in records:
            cache.append(EpisodicTrace.from_dict(rec))
        return cache


class TopicTable:
    """Topic table ``T_t``: one running-mean centroid per topic.

    Following Appendix C.2 ("Memory item fields"), a topic label is assigned to a
    belief at write time as ``z(m) = argmax_z p(z | c(m))`` with
    ``p(z | x) = softmax_z cos(e(x), t_z)``. New topics are opened when the
    best cosine similarity to every existing prototype is below
    ``new_threshold`` (leader clustering); the prototype is then the item's
    own embedding and is updated as a running mean on later assignments.
    The table is only a retrieval prior and never affects which items exist.
    """

    def __init__(self, new_threshold: float = 0.5) -> None:
        self.new_threshold = float(new_threshold)
        self.labels: List[str] = []
        self._centroids: Optional[np.ndarray] = None  # (Z, d)
        self._counts: List[int] = []
        self._index: Dict[str, int] = {}

    def __len__(self) -> int:
        return len(self.labels)

    @property
    def centroids(self) -> Optional[np.ndarray]:
        return self._centroids

    def index_of(self, label: str) -> Optional[int]:
        return self._index.get(label)

    def similarities(self, embedding: np.ndarray) -> np.ndarray:
        if self._centroids is None:
            return np.zeros(0, dtype=np.float64)
        return cosine_sim(np.asarray(embedding, dtype=np.float64)[None, :], self._centroids)[0]

    def distribution(self, embedding: np.ndarray) -> np.ndarray:
        """``p(z | x) = softmax_z cos(e(x), t_z)`` (Eq. 18); empty if no topics."""
        return softmax(self.similarities(embedding))

    def assign(self, embedding: np.ndarray, label_hint: str) -> str:
        """Assign (and if needed create) the topic for a newly written item."""
        vec = np.asarray(embedding, dtype=np.float64)
        if self._centroids is not None and len(self.labels) > 0:
            sims = self.similarities(vec)
            best = int(np.argmax(sims))
            if sims[best] >= self.new_threshold:
                n = self._counts[best]
                self._centroids[best] = (self._centroids[best] * n + vec) / (n + 1)
                self._counts[best] = n + 1
                return self.labels[best]
        label = self._unique_label(label_hint)
        self.labels.append(label)
        self._counts.append(1)
        self._index[label] = len(self.labels) - 1
        self._centroids = vec[None, :] if self._centroids is None else np.vstack([self._centroids, vec[None, :]])
        return label

    def _unique_label(self, hint: str) -> str:
        base = (hint or "topic").strip().lower().replace(" ", "_")[:32] or "topic"
        label, k = base, 2
        while label in self._index:
            label = f"{base}_{k}"
            k += 1
        return label

    def to_dict(self) -> Dict[str, Any]:
        return {
            "new_threshold": self.new_threshold,
            "labels": list(self.labels),
            "counts": list(self._counts),
            "centroids": None if self._centroids is None else self._centroids.astype(np.float32).tolist(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TopicTable:
        table = cls(new_threshold=float(data.get("new_threshold", 0.5)))
        table.labels = [str(x) for x in data.get("labels", [])]
        table._counts = [int(x) for x in data.get("counts", [])]
        table._index = {label: i for i, label in enumerate(table.labels)}
        cents = data.get("centroids")
        table._centroids = None if cents is None else np.asarray(cents, dtype=np.float64)
        return table


@dataclass
class MemoryState:
    """``M_t`` plus the turn counter ``t``."""

    belief_store: BeliefStore = field(default_factory=BeliefStore)
    episodic_cache: EpisodicCache = field(default_factory=EpisodicCache)
    topic_table: TopicTable = field(default_factory=TopicTable)
    turn: int = 0

    def advance(self) -> int:
        self.turn += 1
        return self.turn

    def summary(self, active_threshold: float = 0.8, low_threshold: float = 0.1) -> Dict[str, Any]:
        """Per-turn memory counters (the N_raw / N_eff / N_active / N_low of the memory-dynamics analyses)."""
        store = self.belief_store
        weights = [m.weight for m in store.items()]
        return {
            "turn": self.turn,
            "n_raw": len(weights),
            "n_facts": len(store.facts()),
            "n_hypotheses": len(store.hypotheses()),
            "n_effective": round(store.effective_size(), 3),
            "n_active": sum(1 for w in weights if w >= active_threshold),
            "n_low": sum(1 for w in weights if w < low_threshold),
            "n_episodic": len(self.episodic_cache),
            "n_topics": len(self.topic_table),
        }

    def to_dict(self, include_embeddings: bool = False, include_episodic: bool = True) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "turn": self.turn,
            "belief_store": self.belief_store.to_list(include_embeddings=include_embeddings),
            "topic_table": self.topic_table.to_dict(),
            "episodic_capacity": self.episodic_cache.capacity,
        }
        if include_episodic:
            out["episodic_cache"] = self.episodic_cache.to_list(include_embeddings=include_embeddings)
        return out

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MemoryState:
        capacity = int(data.get("episodic_capacity", 20000))
        return cls(
            belief_store=BeliefStore.from_list(data.get("belief_store", [])),
            episodic_cache=EpisodicCache.from_list(data.get("episodic_cache", []), capacity=capacity),
            topic_table=TopicTable.from_dict(data.get("topic_table", {})),
            turn=int(data.get("turn", 0)),
        )

    def save(self, path: Union[str, Path], include_embeddings: bool = False) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(self.to_dict(include_embeddings=include_embeddings), fh, ensure_ascii=False, indent=2)
        return path

    @classmethod
    def load(cls, path: Union[str, Path]) -> MemoryState:
        with Path(path).open("r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    def render(self, max_items: Optional[int] = None) -> str:
        """Compact human-readable view (the format of Appendix Fig. "Belief Store snapshot")."""
        items = sorted(self.belief_store.items(), key=lambda m: -m.weight)
        if max_items is not None:
            items = items[:max_items]
        lines: List[str] = []
        for m in items:
            tag = "FACT" if m.is_fact else "HYPO"
            flag = " (deprecated)" if m.deprecated else ""
            lines.append(f"[{tag}]  {m.content}  (w={m.weight:.2f}){flag}")
            if m.sources:
                lines.append(f"        sources={m.sources}")
        return "\n".join(lines) if lines else "(empty belief store)"
