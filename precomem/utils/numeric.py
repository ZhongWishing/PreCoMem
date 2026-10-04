"""Small shared helpers."""

from __future__ import annotations

import math
from typing import Any, Optional, Sequence

import numpy as np


def clip01(x: float) -> float:
    return float(min(max(float(x), 0.0), 1.0))


def clip(x: float, lo: float, hi: float) -> float:
    return float(min(max(float(x), lo), hi))


def sigmoid(x: float) -> float:
    """Numerically stable logistic function."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def as_float(value: Any, default: float) -> float:
    """Parse an LLM-emitted scalar leniently; fall back to ``default``."""
    if isinstance(value, bool):
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(out) or math.isinf(out):
        return default
    return out


def as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    return default


def as_str_list(value: Any) -> list:
    if isinstance(value, list):
        return [str(v) for v in value if isinstance(v, (str, int, float))]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity matrix between ``a`` (n×d) and ``b`` (m×d)."""
    a = np.atleast_2d(np.asarray(a, dtype=np.float64))
    b = np.atleast_2d(np.asarray(b, dtype=np.float64))
    a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-9)
    b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-9)
    return a @ b.T


def softmax(x: Sequence[float] | np.ndarray) -> np.ndarray:
    z = np.asarray(x, dtype=np.float64)
    if z.size == 0:
        return z
    z = z - z.max()
    e = np.exp(z)
    return e / max(e.sum(), 1e-12)


def smoothed_normalize(scores: Sequence[float] | np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """The smoothed normalization of Eq. (3): ŝ_i = (s_i + ε) / Σ_j (s_j + ε)."""
    s = np.asarray(scores, dtype=np.float64)
    if s.size == 0:
        return s
    s = s + eps
    total = s.sum()
    if total <= 0:
        return np.full_like(s, 1.0 / s.size)
    return s / total


def truncate(text: Optional[str], limit: int) -> str:
    text = text or ""
    return text if len(text) <= limit else text[: max(limit - 3, 0)] + "..."
