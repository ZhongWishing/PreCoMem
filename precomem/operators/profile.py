"""PROFILE operator (Eq. 8): write a low-confidence hypothesis.

The Profile-Agent returns a class label ``c ∈ {explicit_event, latent_inference}``,
the hypothesis text, an ``initial_confidence`` scalar ``q_t`` and a
``decay_protection`` flag. The deterministic layer then computes

    b_t^{(c)} = w_min^{(c)} + σ(x_t) (w_max^{(c)} - w_min^{(c)})
    w_0(h)    = ½ b_t^{(c)} + ½ clip(q_t, [w_min^{(c)}, w_max^{(c)}])

so the gate score supplies a reproducible prior inside the class band and
the agent scalar can move ``w_0`` by at most half the band width
(Appendix C.3, "PROFILE initialization": with ``I_c = [0.60, 0.70]``, ``x_t = 0.4``
and ``q_t = 0.65`` one obtains ``b_t = 0.6599`` and ``w_0 = 0.6549``).

Embedding and topic assignment for the new item happen in the agent, which
indexes every item written in a turn (hypotheses here, fresh facts in CORRECT).
"""

from __future__ import annotations

from typing import Tuple

from ..config import ProfileConfig
from ..llm.client import LLMClient
from ..llm.prompts import PROFILE_AGENT_PROMPT
from ..memory.items import HypothesisClass, MemoryItem, MemoryType
from ..memory.store import BeliefStore
from ..retrieval.retriever import RetrievalResult
from ..utils.numeric import as_bool, as_float, as_str_list, clip, sigmoid
from .base import OperatorOutcome, ProfileOutcome, format_evidence


def initial_weight(x_t: float, q_t: float, band: Tuple[float, float]) -> Tuple[float, float, float]:
    """Eq. (8). Returns ``(b_t, clipped q_t, w_0)``."""
    w_min, w_max = band
    prior = w_min + sigmoid(x_t) * (w_max - w_min)
    q_clipped = clip(q_t, w_min, w_max)
    return prior, q_clipped, 0.5 * prior + 0.5 * q_clipped


class ProfileOperator:
    """Writes the hypothesis of Eq. (8).

    With ``direct_commit`` (the "w/o PROFILE" ablation) the same Profile-Agent
    extraction is committed immediately as a ``FACT`` at ``commit_weight``:
    medium-surprisal information bypasses the waiting room.
    """

    def __init__(
        self, llm: LLMClient, cfg: ProfileConfig, direct_commit: bool = False, commit_weight: float = 0.85
    ) -> None:
        self.llm = llm
        self.bands = {
            HypothesisClass.EXPLICIT: (float(cfg.explicit_band[0]), float(cfg.explicit_band[1])),
            HypothesisClass.LATENT: (float(cfg.latent_band[0]), float(cfg.latent_band[1])),
        }
        self.direct_commit = bool(direct_commit)
        self.commit_weight = float(commit_weight)

    def __call__(
        self,
        user_input: str,
        retrieval: RetrievalResult,
        x_t: float,
        store: BeliefStore,
        turn: int,
    ) -> OperatorOutcome:
        prompt = PROFILE_AGENT_PROMPT.format(
            user_input=user_input,
            retrieved_context=format_evidence(retrieval, with_scores=True),
            effective_surprisal=float(x_t),
        )
        payload = self.llm.complete_json(prompt)

        cls = HypothesisClass.parse(payload.get("classification"))
        band = self.bands[cls]
        content = str(payload.get("hypothesis_content") or "").strip() or user_input.strip()
        # Default q_t to the band midpoint when the agent omits or garbles it.
        q_t = as_float(payload.get("initial_confidence"), 0.5 * (band[0] + band[1]))
        prior, q_clipped, w0 = initial_weight(x_t, q_t, band)
        # Explicit events are decay-protected by default (Profile-Agent rubric).
        protection = as_bool(payload.get("decay_protection"), cls is HypothesisClass.EXPLICIT)

        hypothesis = MemoryItem(
            content=content,
            mtype=MemoryType.FACT if self.direct_commit else MemoryType.HYPOTHESIS,
            weight=self.commit_weight if self.direct_commit else w0,
            created_turn=turn,
            last_supported_turn=turn,
            last_updated_turn=turn,
            decay_protection=protection,
            classification=cls,
            keywords=as_str_list(payload.get("evidence_keywords")),
            metadata={"decision_logic": str(payload.get("decision_logic", ""))},
        )
        store.add(hypothesis)
        return ProfileOutcome(
            hypothesis=hypothesis,
            classification=cls,
            prior=prior,
            agent_confidence=q_clipped,
            initial_weight=hypothesis.weight,
            agent_payload=payload,
        )
