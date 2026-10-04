"""Dense embedding back-ends.

The default :class:`SentenceTransformersEmbedder` loads
``Qwen/Qwen3-Embedding-0.6B``. A deterministic, dependency-free
:class:`HashingEmbedder` keeps the framework runnable (tests, CPU-only
machines) when the model cannot be loaded; set ``PRECOMEM_STRICT_EMBEDDER=1``
to turn the silent fallback into an error.
"""

from __future__ import annotations

import hashlib
import os
import threading
from typing import Optional, Sequence

import numpy as np

from ..config import EmbeddingConfig
from ..utils.logging import get_logger

_LOGGER = get_logger("retrieval.embedder")


def resolve_device(device: str) -> str:
    """Map ``auto`` to the best available torch device (cuda > mps > cpu)."""
    if device != "auto":
        return device
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


class Embedder:
    """Interface: ``encode(texts) -> (n, dim)`` unit-norm float32 array."""

    dim: int
    name: str = "embedder"

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


class HashingEmbedder(Embedder):
    """Deterministic hashed bag-of-tokens embedding (offline fallback).

    Not a substitute for a neural embedder; it only keeps the ranking
    coherent and reproducible when ``sentence-transformers`` is unavailable.
    """

    name = "hashing"

    def __init__(self, dim: int = 256) -> None:
        self.dim = int(dim)

    def _embed_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for tok in text.lower().split():
            digest = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
            h = int.from_bytes(digest, "little", signed=False)
            vec[h % self.dim] += 1.0 if (h >> 16) & 1 else -1.0
        return vec

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        out = np.stack([self._embed_one(t or "") for t in texts], axis=0)
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms < 1e-9] = 1.0
        return out / norms


class SentenceTransformersEmbedder(Embedder):
    """Wrapper around ``sentence_transformers`` (Qwen3-Embedding by default).

    ``encode`` is guarded by a lock so that one shared instance can serve
    several benchmark workers; inference is already batched per call.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        device: str = "auto",
        max_seq_len: int = 512,
        batch_size: int = 32,
        cache_dir: Optional[str] = None,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self.device = resolve_device(device)
        self.batch_size = int(batch_size)
        self.model = SentenceTransformer(model_name, device=self.device, cache_folder=cache_dir)
        self.model.max_seq_length = int(max_seq_len)
        self.dim = int(self.model.get_sentence_embedding_dimension())
        self._lock = threading.Lock()

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        with self._lock:
            out = self.model.encode(
                list(texts),
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        return np.asarray(out, dtype=np.float32)


def build_embedder(cfg: EmbeddingConfig) -> Embedder:
    """Factory: neural embedder when possible, hashing fallback otherwise."""
    provider = cfg.provider.lower()
    if provider == "hashing":
        return HashingEmbedder(dim=cfg.fallback_dim)
    if provider in {"huggingface", "sentence-transformers", "qwen3"}:
        try:
            return SentenceTransformersEmbedder(
                model_name=cfg.model,
                device=cfg.device,
                max_seq_len=cfg.max_seq_len,
                batch_size=cfg.batch_size,
            )
        except Exception as exc:  # pragma: no cover - depends on the environment
            if os.environ.get("PRECOMEM_STRICT_EMBEDDER") == "1":
                raise
            _LOGGER.warning(
                "Could not load embedding model %s (%s: %s); falling back to HashingEmbedder. "
                "Set PRECOMEM_STRICT_EMBEDDER=1 to make this an error.",
                cfg.model,
                type(exc).__name__,
                exc,
            )
            return HashingEmbedder(dim=cfg.fallback_dim)
    raise ValueError(f"Unknown embedding provider: {cfg.provider!r}")
