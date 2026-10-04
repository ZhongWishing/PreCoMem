"""Hypothesis lifecycle (Eq. 9, promotion and pruning): Phase D of Algorithm 1.

For every unprotected ``HYPOTHESIS`` that received no supporting evidence in
the current turn,

    w_t(h) <- w_t(h) · exp(-γ · a_t(h)),    a_t(h) = t - l(h),

where ``l(h)`` is the turn of creation or of the last supporting update.
Because ``a_t`` grows with every unsupported turn the factors compound, so
after ``n`` consecutive unsupported turns the cumulative factor is
``exp(-γ n(n+1)/2)`` (Appendix C.3, "Hypothesis decay horizon"). A hypothesis
supported this turn has ``a_t(h) = 0`` and is left unchanged, which is
exactly the "reset anchor, skip decay" branch of Algorithm 1.

Afterwards hypotheses at ``w >= τ_promote`` are promoted (one-way) to
``FACT`` and hypotheses below the prune threshold are removed. Facts are
never deleted by default; the optional ``fact_optional_prune`` threshold
exists only for the fact-pruning ablation (Appendix D.7).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List

from ..config import LifecycleConfig
from ..memory.items import MemoryItem, MemoryType
from ..memory.store import BeliefStore


@dataclass
class LifecycleStats:
    decayed: int = 0
    promoted: int = 0
    pruned: int = 0
    fact_pruned: int = 0
    removed: List[MemoryItem] = field(default_factory=list, repr=False)

    def to_dict(self) -> Dict[str, int]:
        return {
            "decayed": self.decayed,
            "promoted": self.promoted,
            "pruned": self.pruned,
            "fact_pruned": self.fact_pruned,
        }


def cumulative_decay(gamma: float, n_unsupported_turns: int) -> float:
    """``exp(-γ n(n+1)/2)``: total factor after ``n`` consecutive unsupported turns."""
    n = max(0, int(n_unsupported_turns))
    return math.exp(-gamma * n * (n + 1) / 2.0)


class HypothesisLifecycle:
    def __init__(self, cfg: LifecycleConfig) -> None:
        self.gamma = float(cfg.decay_gamma)
        self.promote_threshold = float(cfg.promote_threshold)
        self.prune_threshold = float(cfg.hypothesis_prune)
        self.fact_optional_prune = None if cfg.fact_optional_prune is None else float(cfg.fact_optional_prune)

    def step(self, store: BeliefStore, turn: int) -> LifecycleStats:
        stats = LifecycleStats()
        to_remove: List[MemoryItem] = []

        for item in store.items():
            if item.mtype is MemoryType.HYPOTHESIS:
                age = item.age_since_support(turn)
                if age > 0 and not item.decay_protection:
                    item.set_weight(item.weight * math.exp(-self.gamma * age), turn)
                    stats.decayed += 1
                if item.weight >= self.promote_threshold:
                    item.mtype = MemoryType.FACT
                    item.last_updated_turn = turn
                    item.metadata["promoted_at_turn"] = turn
                    stats.promoted += 1
                elif item.weight < self.prune_threshold:
                    to_remove.append(item)
            elif self.fact_optional_prune is not None and item.weight < self.fact_optional_prune:
                to_remove.append(item)

        for item in to_remove:
            if store.remove(item.item_id) is None:
                continue
            stats.removed.append(item)
            if item.mtype is MemoryType.HYPOTHESIS:
                stats.pruned += 1
            else:
                stats.fact_pruned += 1
        return stats
