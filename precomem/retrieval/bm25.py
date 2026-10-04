"""Okapi BM25 for the lexical term ``s_fact`` of the retrieval score.

The belief store is small (hundreds to a few thousand items) and changes
every turn, so the index is simply rebuilt whenever the store's version
changes; the cost is negligible next to the embedding and NLI calls.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Sequence

import numpy as np

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


def tokenize(text: str) -> List[str]:
    """Lower-cased word tokens; CJK characters are emitted one per token."""
    if not text:
        return []
    return [t.lower() for t in _TOKEN_RE.findall(text)]


class BM25Index:
    """Okapi BM25 over a fixed list of documents."""

    def __init__(self, documents: Sequence[str], k1: float = 1.5, b: float = 0.75, idf_floor: float = 0.25) -> None:
        self.k1 = float(k1)
        self.b = float(b)
        self._tfs: List[Counter] = [Counter(tokenize(d)) for d in documents]
        self._lengths = np.asarray([sum(tf.values()) for tf in self._tfs], dtype=np.float64)
        self._avgdl = float(self._lengths.mean()) if len(self._lengths) else 0.0
        n = len(self._tfs)
        df: Counter = Counter()
        for tf in self._tfs:
            df.update(tf.keys())
        self._idf: Dict[str, float] = {
            tok: max(math.log((n - f + 0.5) / (f + 0.5) + 1.0), idf_floor) for tok, f in df.items()
        }

    def __len__(self) -> int:
        return len(self._tfs)

    def score(self, query: str) -> np.ndarray:
        """Raw BM25 scores for every document (zeros if the query is empty)."""
        scores = np.zeros(len(self._tfs), dtype=np.float64)
        q_tokens = [t for t in tokenize(query) if t in self._idf]
        if not q_tokens or not self._tfs:
            return scores
        avgdl = max(self._avgdl, 1.0)
        for i, tf in enumerate(self._tfs):
            norm = self.k1 * (1.0 - self.b + self.b * max(self._lengths[i], 1.0) / avgdl)
            s = 0.0
            for q in q_tokens:
                f = tf.get(q, 0)
                if f:
                    s += self._idf[q] * f * (self.k1 + 1.0) / (f + norm)
            scores[i] = s
        return scores

    def normalized_score(self, query: str, eps: float = 1e-9) -> np.ndarray:
        """``s_fact`` of Eq. (17): BM25 divided by the pool maximum (plus ε)."""
        raw = self.score(query)
        if raw.size == 0:
            return raw
        return raw / (raw.max() + eps)
