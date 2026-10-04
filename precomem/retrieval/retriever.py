"""Confidence-modulated tri-signal retrieval (Eq. 1 and Eqs. 16–18).

For every belief ``m`` in ``B_t`` the retriever computes three relevance
signals in ``[0, 1]``,

* ``s_sem``   = (1 + cos(e(u_t), e(c(m)))) / 2                 (Eq. 16)
* ``s_fact``  = BM25(u_t, c(m)) / (max BM25 + ε)               (Eq. 17)
* ``s_topic`` = p(z(m) | u_t) = softmax_z cos(e(u_t), t_z)     (Eq. 18)

and ranks by the confidence-modulated score

    s(u_t, m) = w_t(m) · (α₁ s_sem + α₂ s_fact + α₃ s_topic).

The Top-K beliefs ``C_t`` and their scores ``s_t`` are returned together with
the episodic traces they point to (provenance only; traces never enter the
ranking). The same scores feed the reliability coefficient ``ρ_t``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from ..config import RetrievalConfig
from ..memory.items import EpisodicTrace, MemoryItem
from ..memory.store import BeliefStore, EpisodicCache, TopicTable
from ..utils.numeric import cosine_sim
from .bm25 import BM25Index
from .embedder import Embedder


@dataclass
class RetrievalResult:
    """``(C_t, s_t)`` plus the per-signal breakdown and linked provenance."""

    query_embedding: Optional[np.ndarray] = field(default=None, repr=False)
    items: List[MemoryItem] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    semantic: List[float] = field(default_factory=list)
    lexical: List[float] = field(default_factory=list)
    topic: List[float] = field(default_factory=list)
    traces: List[EpisodicTrace] = field(default_factory=list)
    n_candidates: int = 0

    @property
    def n(self) -> int:
        return len(self.items)

    @property
    def is_empty(self) -> bool:
        return not self.items

    def item_embeddings(self) -> np.ndarray:
        return np.stack([np.asarray(m.embedding, dtype=np.float32) for m in self.items], axis=0)

    def summary(self) -> List[dict]:
        return [
            {
                "id": m.item_id,
                "type": m.mtype.value,
                "weight": round(m.weight, 4),
                "score": round(s, 4),
                "content": m.content,
            }
            for m, s in zip(self.items, self.scores)
        ]


class Retriever:
    """Implements the Memory-Leveraging read path (Section 2.3)."""

    def __init__(self, embedder: Embedder, cfg: RetrievalConfig, use_weights: bool = True) -> None:
        self.embedder = embedder
        alpha = np.asarray(cfg.alpha.as_tuple(), dtype=np.float64)
        self.alpha = alpha / alpha.sum()  # convex weights
        self.cfg = cfg
        self.use_weights = use_weights  # False = factorial control without read-time confidence weighting
        self._bm25: Optional[BM25Index] = None
        self._bm25_version: int = -1

    # -- embedding cache ---------------------------------------------
    def ensure_embeddings(self, items: Sequence[MemoryItem]) -> np.ndarray:
        """Embed items whose cached embedding is missing (after creation or a content edit)."""
        missing = [m for m in items if m.embedding is None]
        if missing:
            vecs = self.embedder.encode([m.content for m in missing])
            for m, v in zip(missing, vecs):
                m.embedding = np.asarray(v, dtype=np.float32)
        return np.stack([np.asarray(m.embedding, dtype=np.float32) for m in items], axis=0)

    def ensure_trace_embedding(self, trace: EpisodicTrace) -> np.ndarray:
        if trace.embedding is None:
            trace.embedding = np.asarray(self.embedder.encode_one(trace.text), dtype=np.float32)
        return trace.embedding

    def _lexical_index(self, store: BeliefStore, items: Sequence[MemoryItem]) -> BM25Index:
        if self._bm25 is None or self._bm25_version != store.version:
            self._bm25 = BM25Index([m.lexical_text() for m in items], k1=self.cfg.bm25_k1, b=self.cfg.bm25_b)
            self._bm25_version = store.version
        return self._bm25

    # -- main entry ---------------------------------------------------
    def retrieve(
        self,
        query: str,
        store: BeliefStore,
        episodic_cache: EpisodicCache,
        topic_table: Optional[TopicTable] = None,
        top_k: Optional[int] = None,
        query_embedding: Optional[np.ndarray] = None,
    ) -> RetrievalResult:
        k = int(top_k or self.cfg.top_k)
        q_emb = (
            np.asarray(query_embedding, dtype=np.float32)
            if query_embedding is not None
            else np.asarray(self.embedder.encode_one(query), dtype=np.float32)
        )
        items = store.items()
        if not items:
            return RetrievalResult(query_embedding=q_emb)

        item_embs = self.ensure_embeddings(items)
        s_sem = np.clip((cosine_sim(q_emb[None, :], item_embs)[0] + 1.0) / 2.0, 0.0, 1.0)
        s_fact = self._lexical_index(store, items).normalized_score(query)
        s_topic = self._topic_scores(q_emb, items, topic_table)

        relevance = self.alpha[0] * s_sem + self.alpha[1] * s_fact + self.alpha[2] * s_topic
        if self.use_weights:
            relevance = np.asarray([m.weight for m in items], dtype=np.float64) * relevance
        combined = relevance

        order = np.argsort(-combined, kind="stable")[: min(k, len(items))]
        result = RetrievalResult(
            query_embedding=q_emb,
            items=[items[i] for i in order],
            scores=[float(combined[i]) for i in order],
            semantic=[float(s_sem[i]) for i in order],
            lexical=[float(s_fact[i]) for i in order],
            topic=[float(s_topic[i]) for i in order],
            n_candidates=len(items),
        )
        result.traces = self._provenance(result.items, episodic_cache)
        return result

    def _topic_scores(
        self, q_emb: np.ndarray, items: Sequence[MemoryItem], topic_table: Optional[TopicTable]
    ) -> np.ndarray:
        scores = np.zeros(len(items), dtype=np.float64)
        if topic_table is None or len(topic_table) == 0 or self.alpha[2] <= 0:
            return scores
        dist = topic_table.distribution(q_emb)
        for i, m in enumerate(items):
            idx = topic_table.index_of(m.topic) if m.topic else None
            if idx is not None:
                scores[i] = float(dist[idx])
        return scores

    def _provenance(self, items: Sequence[MemoryItem], cache: EpisodicCache) -> List[EpisodicTrace]:
        """Episodic excerpts linked from the Top-K beliefs, oldest first, capped."""
        ids: List[str] = []
        for m in items:
            ids.extend(m.sources)
        traces = cache.resolve(ids)
        traces.sort(key=lambda t: t.turn)
        limit = self.cfg.max_provenance_traces
        if limit and len(traces) > limit:
            traces = traces[-limit:]  # keep the most recent excerpts
        return traces
