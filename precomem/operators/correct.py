"""CORRECT operator (Eq. 10): soft correction of contradicted beliefs.

For every retrieved belief the Correct-Agent names, with severity
``ζ_t(m) ∈ [0, 1]``,

    w_{t+1}(m) = w_t(m) · exp(-β · x_t · ζ_t(m)),    m ∈ C_t ∩ B_t.

Items the agent marks ``archive_and_replace`` are additionally flagged as
deprecated. They are kept at low confidence and never hard-deleted (see
Appendix C.1, "Why low-confidence facts are retained rather than pruned").
The new observation is recorded as a fresh belief node with weight
``new_fact_weight`` (0.85 by default, in the ``w >= 0.80`` range of stable
facts). A contradiction is not supporting evidence, so ``l(h)`` of a decayed
hypothesis is left untouched.
"""

from __future__ import annotations

import math

from ..config import CorrectConfig
from ..llm.client import LLMClient
from ..llm.prompts import CORRECT_AGENT_PROMPT
from ..memory.items import MemoryItem, MemoryType
from ..memory.store import BeliefStore
from ..retrieval.retriever import RetrievalResult
from ..utils.numeric import as_float, clip01
from .base import CorrectOutcome, NoOpOutcome, OperatorOutcome, WeightUpdate, format_evidence, records_with_ids

_ARCHIVE = "archive_and_replace"
_DECREASE = "decrease_confidence"


def soft_decay(weight: float, beta: float, x_t: float, severity: float) -> float:
    """Eq. (10)."""
    return weight * math.exp(-beta * x_t * severity)


class CorrectOperator:
    def __init__(self, llm: LLMClient, cfg: CorrectConfig) -> None:
        self.llm = llm
        self.beta = float(cfg.beta)
        self.new_fact_weight = float(cfg.new_fact_weight)

    def __call__(
        self,
        user_input: str,
        retrieval: RetrievalResult,
        x_t: float,
        store: BeliefStore,
        turn: int,
    ) -> OperatorOutcome:
        if retrieval.is_empty:
            return NoOpOutcome(reason="no retrieved evidence to correct")

        prompt = CORRECT_AGENT_PROMPT.format(
            user_input=user_input,
            retrieved_context=format_evidence(retrieval),
            effective_surprisal=float(x_t),
        )
        payload = self.llm.complete_json(prompt)
        retrieved_ids = {m.item_id for m in retrieval.items}
        outcome = CorrectOutcome(
            diagnosis=str(payload.get("diagnosis", "")),
            conflict_type=str(payload.get("conflict_type", "")),
            agent_payload=payload,
        )

        for rec in records_with_ids(payload, "contradicted_memories"):
            item = store.get(rec["id"])
            if item is None or item.item_id not in retrieved_ids:
                continue
            severity = clip01(as_float(rec.get("severity"), 0.0))
            operation = str(rec.get("operation") or _DECREASE).strip().lower()
            old = item.weight
            item.set_weight(soft_decay(old, self.beta, float(x_t), severity), turn)
            if operation == _ARCHIVE and not item.deprecated:
                item.deprecated = True
                outcome.archived.append(item)
            outcome.updates.append(
                WeightUpdate(
                    item_id=item.item_id,
                    old_weight=old,
                    new_weight=item.weight,
                    scalar=severity,
                    reason=str(rec.get("reason", "")),
                    operation=operation if operation in {_ARCHIVE, _DECREASE} else _DECREASE,
                )
            )

        new_content = payload.get("new_fact_content")
        if isinstance(new_content, str) and new_content.strip():
            outcome.new_fact = store.add(
                MemoryItem(
                    content=new_content.strip(),
                    mtype=MemoryType.FACT,
                    weight=self.new_fact_weight,
                    created_turn=turn,
                    last_supported_turn=turn,
                    last_updated_turn=turn,
                    metadata={"origin": "correct_agent", "conflict_type": outcome.conflict_type},
                )
            )
        return outcome
