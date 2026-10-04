"""Calibrated NLI contradiction verifier ``Verif_conf`` (Eq. 21).

``π_t(m) = softmax(z(u_t, c(m)) / T)_contr`` where ``z`` are the logits of a
cross-encoder NLI model over {entailment, neutral, contradiction} and ``T`` is
a scalar temperature fitted on the MNLI matched-dev split
(``precomem-calibrate-nli``). The premise is the stored belief ``c(m)`` and
the hypothesis is the new user input ``u_t``.

A keyword/negation heuristic keeps the framework runnable without the model
(tests, CPU-only machines). Set ``PRECOMEM_STRICT_NLI=1`` to disable it.
"""

from __future__ import annotations

import os
import re
import threading
from typing import List, Optional, Sequence, Tuple

from ..config import NLIConfig
from ..retrieval.embedder import resolve_device
from ..utils.logging import get_logger

_LOGGER = get_logger("surprisal.nli")


class ContradictionVerifier:
    """Interface: ``contradict_batch([(premise, hypothesis), ...]) -> [π, ...]``."""

    name: str = "verifier"

    def contradict_batch(self, pairs: Sequence[Tuple[str, str]]) -> List[float]:
        raise NotImplementedError

    def contradict(self, premise: str, hypothesis: str) -> float:
        return self.contradict_batch([(premise, hypothesis)])[0]


class HeuristicVerifier(ContradictionVerifier):
    """Lexical-overlap + negation/shift cue heuristic (offline fallback only)."""

    name = "heuristic"

    _NEG_RE = re.compile(
        r"\b(no|not|never|don'?t|doesn'?t|didn'?t|can'?t|cannot|couldn'?t|won'?t|"
        r"isn'?t|aren'?t|wasn'?t|weren'?t|stopped|quit|nope|none)\b",
        re.IGNORECASE,
    )
    _SHIFT_RE = re.compile(
        r"\b(switched|switching|moved|changed|changing|anymore|no longer|used to|previously|instead)\b",
        re.IGNORECASE,
    )

    @staticmethod
    def _tokens(text: str) -> set:
        return {t for t in re.findall(r"[A-Za-z]+", text.lower()) if len(t) > 2}

    def contradict_batch(self, pairs: Sequence[Tuple[str, str]]) -> List[float]:
        return [self._score(p, h) for p, h in pairs]

    def _score(self, premise: str, hypothesis: str) -> float:
        pt, ht = self._tokens(premise), self._tokens(hypothesis)
        if not pt or not ht:
            return 0.0
        overlap = len(pt & ht) / len(ht)
        negated = bool(self._NEG_RE.search(hypothesis))
        shifted = bool(self._SHIFT_RE.search(hypothesis))
        score = 0.10
        if overlap > 0.15:
            score += 0.45 * overlap
        if negated:
            score += 0.35
        if shifted:
            score += 0.20
        if negated and overlap > 0.05:
            score += 0.10
        return float(min(max(score, 0.0), 1.0))


class NLIVerifier(ContradictionVerifier):
    """Temperature-scaled cross-encoder NLI verifier (lazy model loading)."""

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-base",
        temperature: float = 1.0,
        device: str = "auto",
        batch_size: int = 16,
        max_length: int = 256,
    ) -> None:
        self.name = model_name
        self.model_name = model_name
        self.temperature = max(float(temperature), 1e-3)
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self.device = device
        self._model = None
        self._tokenizer = None
        self._contr_idx = 2
        self._lock = threading.Lock()

    def load(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = resolve_device(self.device)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self._model.to(self.device)
        self._model.eval()
        self._contr_idx = contradiction_index(getattr(self._model.config, "id2label", None))

    def contradict_batch(self, pairs: Sequence[Tuple[str, str]]) -> List[float]:
        if not pairs:
            return []
        self.load()
        import torch

        out: List[float] = []
        with self._lock, torch.no_grad():
            for start in range(0, len(pairs), self.batch_size):
                chunk = pairs[start : start + self.batch_size]
                enc = self._tokenizer(
                    [p for p, _ in chunk],
                    [h for _, h in chunk],
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)
                logits = self._model(**enc).logits.float() / self.temperature
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                out.extend(float(row[self._contr_idx]) for row in probs)
        return out


def contradiction_index(id2label: Optional[dict]) -> int:
    """Find the ``contradiction`` logit column from a model's ``id2label`` map.

    Checkpoints disagree on label order (DeBERTa cross-encoders use
    [contradiction, entailment, neutral]; many MNLI fine-tunes use
    [entailment, neutral, contradiction]), so the index is always resolved
    from the config rather than assumed.
    """
    for idx, label in (id2label or {}).items():
        if "contr" in str(label).lower():
            return int(idx)
    return 2


def build_verifier(cfg: NLIConfig) -> ContradictionVerifier:
    """Factory: the neural verifier when its dependencies load, else the heuristic."""
    if cfg.model.lower() in {"heuristic", "fallback", "none"}:
        return HeuristicVerifier()
    verifier = NLIVerifier(
        model_name=cfg.model,
        temperature=cfg.temperature,
        device=cfg.device,
        batch_size=cfg.batch_size,
        max_length=cfg.max_length,
    )
    try:
        verifier.load()
    except Exception as exc:  # pragma: no cover - depends on the environment
        if os.environ.get("PRECOMEM_STRICT_NLI") == "1":
            raise
        _LOGGER.warning(
            "Could not load NLI model %s (%s: %s); using the heuristic verifier. "
            "Set PRECOMEM_STRICT_NLI=1 to make this an error.",
            cfg.model,
            type(exc).__name__,
            exc,
        )
        return HeuristicVerifier()
    return verifier
