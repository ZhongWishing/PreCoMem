"""Reliability ``ρ_t``, raw surprisal ``S_raw`` and effective surprisal ``S_eff``.

These closed-form steps (Eqs. 3, 4, 5, 19, 20, 22) are the deterministic
core of Memory Evolution. They are exposed both as pure functions, which
the tests exercise directly, and as :func:`compute_surprisal`, which the
agent calls once per turn.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..retrieval.retriever import RetrievalResult
from ..utils.numeric import clip01, cosine_sim, smoothed_normalize
from .nli import ContradictionVerifier


@dataclass(frozen=True)
class SurprisalSignal:
    """Everything Phase B produces for one turn."""

    rho: float  # retrieval reliability ρ_t              (Eq. 3)
    d_sem: float  # semantic deviation                      (Eq. 19/20)
    d_conf: float  # max contradiction probability           (Eq. 22)
    s_raw: float  # λ d_sem + (1 - λ) d_conf                (Eq. 4)
    s_eff: float  # ρ_t · S_raw, the gate score x_t         (Eq. 5)
    per_item_contradiction: tuple = ()  # π_t(m_i) for each retrieved item

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["per_item_contradiction"] = [round(float(p), 4) for p in self.per_item_contradiction]
        for key in ("rho", "d_sem", "d_conf", "s_raw", "s_eff"):
            d[key] = round(float(d[key]), 4)
        return d


def retrieval_reliability(scores: Sequence[float], eps: float = 1e-9) -> float:
    """Eq. (3): ``ρ_t = 1 - H_t`` with ``H_t`` the normalized entropy of ``ŝ``.

    Returns 0 for an empty evidence set (no evidence, no reliability).
    """
    s = smoothed_normalize(scores, eps=eps)
    if s.size == 0:
        return 0.0
    entropy = -float(np.sum(s * np.log(np.maximum(s, 1e-300))))
    h_t = entropy / math.log(max(s.size, 2))
    return 1.0 - clip01(h_t)


def semantic_deviation(query_emb: np.ndarray, item_embs: np.ndarray, norm_scores: Sequence[float]) -> float:
    """Eqs. (19)+(20): ``d_sem = (1 - cos(e(u_t), Σ_i ŝ_i e(c(m_i)))) / 2``."""
    if item_embs.size == 0:
        return 0.0
    weights = np.asarray(norm_scores, dtype=np.float64).reshape(-1, 1)
    if weights.sum() <= 0:
        weights = np.full_like(weights, 1.0 / item_embs.shape[0])
    evidence = (np.asarray(item_embs, dtype=np.float64) * weights).sum(axis=0)
    cos = float(cosine_sim(np.asarray(query_emb, dtype=np.float64)[None, :], evidence[None, :])[0, 0])
    return clip01(0.5 * (1.0 - cos))


def contradiction_scores(user_input: str, premises: Sequence[str], verifier: ContradictionVerifier) -> List[float]:
    """Eq. (21): ``π_t(m) = Verif_conf(u_t, c(m))`` for every retrieved belief."""
    if not premises:
        return []
    return [float(p) for p in verifier.contradict_batch([(c, user_input) for c in premises])]


def raw_surprisal(d_sem: float, d_conf: float, lam: float) -> float:
    """Eq. (4): ``S_raw = λ d_sem + (1 - λ) d_conf``."""
    lam = clip01(lam)
    return clip01(lam * d_sem + (1.0 - lam) * d_conf)


def effective_surprisal(rho: float, s_raw: float) -> float:
    """Eq. (5): ``S_eff = ρ_t · S_raw``."""
    return clip01(rho * s_raw)


def compute_surprisal(
    user_input: str,
    retrieval: RetrievalResult,
    verifier: ContradictionVerifier,
    lam: float,
    eps: float = 1e-9,
    query_embedding: Optional[np.ndarray] = None,
    force_rho_one: bool = False,
) -> SurprisalSignal:
    """Phase B of Algorithm 1 for one turn.

    With an empty evidence set every component is defined as 0, which the
    gate interprets as the cold-start case (Appendix C.2, "Cold start").
    ``force_rho_one`` implements the RR ablation (``rho_t = 1``).
    """
    if retrieval.is_empty:
        return SurprisalSignal(rho=0.0, d_sem=0.0, d_conf=0.0, s_raw=0.0, s_eff=0.0)

    q_emb = query_embedding if query_embedding is not None else retrieval.query_embedding
    rho = 1.0 if force_rho_one else retrieval_reliability(retrieval.scores, eps=eps)
    norm_scores = smoothed_normalize(retrieval.scores, eps=eps)
    d_sem = semantic_deviation(q_emb, retrieval.item_embeddings(), norm_scores)
    per_item = contradiction_scores(user_input, [m.content for m in retrieval.items], verifier)
    d_conf = clip01(max(per_item)) if per_item else 0.0
    s_raw = raw_surprisal(d_sem, d_conf, lam)
    return SurprisalSignal(
        rho=rho,
        d_sem=d_sem,
        d_conf=d_conf,
        s_raw=s_raw,
        s_eff=effective_surprisal(rho, s_raw),
        per_item_contradiction=tuple(per_item),
    )
