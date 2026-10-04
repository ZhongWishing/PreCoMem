"""Per-turn trace emitted by :class:`precomem.agent.PreCoMem`.

A trace records everything needed to audit one pass through Algorithm 1
without re-running the model: the retrieved evidence, the surprisal
components, the gate decision, the operator's closed-form updates, the
lifecycle counters, memory-size summaries, latency and token usage.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Union

from .gate import GateDecision
from .surprisal.signals import SurprisalSignal


@dataclass
class TurnTrace:
    turn: int
    user_input: str
    response: str
    gate: GateDecision
    surprisal: SurprisalSignal
    retrieved: List[Dict[str, Any]] = field(default_factory=list)
    operator: Dict[str, Any] = field(default_factory=dict)
    lifecycle: Dict[str, int] = field(default_factory=dict)
    memory: Dict[str, Any] = field(default_factory=dict)
    latency_ms: Dict[str, float] = field(default_factory=dict)
    tokens: Dict[str, int] = field(default_factory=dict)
    trace_id: str = ""

    @property
    def regime(self) -> str:
        return self.gate.regime.value

    @property
    def s_eff(self) -> float:
        return self.surprisal.s_eff

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn": self.turn,
            "user": self.user_input,
            "response": self.response,
            "regime": self.regime,
            "gate": self.gate.to_dict(),
            "surprisal": self.surprisal.to_dict(),
            "retrieved": self.retrieved,
            "operator": self.operator,
            "lifecycle": self.lifecycle,
            "memory": self.memory,
            "latency_ms": {k: round(v, 2) for k, v in self.latency_ms.items()},
            "tokens": self.tokens,
            "trace_id": self.trace_id,
        }

    def one_line(self) -> str:
        s = self.surprisal
        return (
            f"[turn {self.turn:04d}] {self.regime:<8} rho={s.rho:.3f} d_sem={s.d_sem:.3f} "
            f"d_conf={s.d_conf:.3f} S_eff={s.s_eff:.3f}"
        )


def write_traces(traces: Iterable[TurnTrace], path: Union[str, Path]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for tr in traces:
            fh.write(json.dumps(tr.to_dict(), ensure_ascii=False) + "\n")
    return path
