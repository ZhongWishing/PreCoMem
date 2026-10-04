"""MAINTAIN operator (Eq. 7): reinforcement of supported beliefs.

    w_{t+1}(m) = w_t(m) + η_t(m) (1 - w_t(m)),    m ∈ C_t ∩ B_t,

where ``η_t(m) ∈ (0, 1]`` is the per-item ``evidence_strength`` returned by
the Maintain-Agent. Reinforced items count as *supported* this turn, which
resets their decay anchor ``l(h)`` (Eq. 9). The agent may also attach one
refinement (a sub-concept, attribute, temporal or additional detail) to a
parent belief; the parent's embedding is invalidated so retrieval sees the
new text.
"""

from __future__ import annotations

from ..config import MaintainConfig
from ..llm.client import LLMClient
from ..llm.prompts import MAINTAIN_AGENT_PROMPT
from ..memory.store import BeliefStore
from ..retrieval.retriever import RetrievalResult
from ..utils.numeric import as_float, clip
from .base import (
    MaintainOutcome,
    NoOpOutcome,
    OperatorOutcome,
    Refinement,
    WeightUpdate,
    format_evidence,
    records_with_ids,
)

_REFINEMENT_KINDS = {"sub_concept", "attribute", "temporal", "additional"}


def reinforce(weight: float, eta: float) -> float:
    """Eq. (7)."""
    return weight + eta * (1.0 - weight)


class MaintainOperator:
    def __init__(self, llm: LLMClient, cfg: MaintainConfig) -> None:
        self.llm = llm
        self.eta_min = float(cfg.eta_min)
        self.eta_max = float(cfg.eta_max)

    def __call__(
        self,
        user_input: str,
        retrieval: RetrievalResult,
        x_t: float,
        store: BeliefStore,
        turn: int,
    ) -> OperatorOutcome:
        if retrieval.is_empty:
            return NoOpOutcome(reason="no retrieved evidence to reinforce")

        prompt = MAINTAIN_AGENT_PROMPT.format(
            existing_memories=format_evidence(retrieval),
            user_input=user_input,
            effective_surprisal=float(x_t),
        )
        payload = self.llm.complete_json(prompt)
        retrieved_ids = {m.item_id for m in retrieval.items}
        outcome = MaintainOutcome(agent_payload=payload)

        for rec in records_with_ids(payload, "reinforced_memories"):
            item = store.get(rec["id"])
            if item is None or item.item_id not in retrieved_ids:
                continue  # the agent may only touch items in C_t
            eta = clip(as_float(rec.get("evidence_strength"), self.eta_min), self.eta_min, self.eta_max)
            old = item.weight
            item.set_weight(reinforce(old, eta), turn)
            item.mark_supported(turn)
            outcome.updates.append(
                WeightUpdate(
                    item_id=item.item_id,
                    old_weight=old,
                    new_weight=item.weight,
                    scalar=eta,
                    reason=str(rec.get("reason", "")),
                )
            )

        refinement = payload.get("refinement_action") if isinstance(payload, dict) else None
        if isinstance(refinement, dict):
            kind = str(refinement.get("type") or "none").strip().lower()
            parent_id = refinement.get("target_parent_id")
            detail = refinement.get("new_content")
            if kind in _REFINEMENT_KINDS and isinstance(parent_id, str) and isinstance(detail, str) and detail.strip():
                parent = store.get(parent_id)
                if parent is not None and parent.item_id in retrieved_ids:
                    store.update_content(parent.item_id, f"{parent.content} | {kind}: {detail.strip()}", turn)
                    outcome.refinements.append(Refinement(parent_id=parent.item_id, kind=kind, detail=detail.strip()))
        return outcome
