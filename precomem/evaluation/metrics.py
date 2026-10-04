"""Aggregate metrics: per-category accuracy, cross-seed mean/std, bootstrap CIs."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


def mean_std(values: Sequence[float]) -> Tuple[float, float]:
    """Mean and sample standard deviation (``ddof=1``; 0 for fewer than two values)."""
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return 0.0, 0.0
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return float(arr.mean()), std


def bootstrap_ci(
    values: Sequence[float], confidence: float = 0.95, n_iter: int = 1000, seed: int = 1234
) -> Tuple[float, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_iter, arr.size))
    boot = arr[idx].mean(axis=1)
    lo = float(np.percentile(boot, (1 - confidence) / 2 * 100))
    hi = float(np.percentile(boot, (1 + confidence) / 2 * 100))
    return lo, hi


def summarize_records(records: Iterable[Mapping[str, Any]], score_key: str = "score") -> Dict[str, Any]:
    """Overall and per-category accuracy (in %) for one seed's prediction records."""
    scores: List[float] = []
    by_cat: Dict[str, List[float]] = {}
    for rec in records:
        s = float(rec.get(score_key, 0.0))
        scores.append(s)
        by_cat.setdefault(str(rec.get("category") or "uncategorized"), []).append(s)
    return {
        "n": len(scores),
        "overall": round(100.0 * float(np.mean(scores)), 4) if scores else 0.0,
        "per_category": {
            k: {"n": len(v), "accuracy": round(100.0 * float(np.mean(v)), 4)} for k, v in sorted(by_cat.items())
        },
    }


def aggregate_seeds(by_seed: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    """Cross-seed mean ± std of the summaries produced by :func:`summarize_records`."""
    if not by_seed:
        return {}
    overall_mean, overall_std = mean_std([m["overall"] for m in by_seed.values()])
    categories = sorted({c for m in by_seed.values() for c in m.get("per_category", {})})
    per_cat: Dict[str, Dict[str, float]] = {}
    for cat in categories:
        vals = [m["per_category"][cat]["accuracy"] for m in by_seed.values() if cat in m.get("per_category", {})]
        mu, sd = mean_std(vals)
        per_cat[cat] = {"mean": round(mu, 4), "std": round(sd, 4), "n_seeds": len(vals)}
    return {"overall_mean": round(overall_mean, 4), "overall_std": round(overall_std, 4), "per_category": per_cat}
