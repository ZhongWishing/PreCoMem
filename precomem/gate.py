"""The two-threshold regime gate (Eq. 6) and its threshold policies.

``gate(x_t)`` maps the effective surprisal to one of three regimes::

    MAINTAIN   x_t <= theta_low
    PROFILE    theta_low < x_t <= theta_high
    CORRECT    x_t > theta_high

The thresholds come from a :class:`ThresholdPolicy`: :class:`FixedThresholds`
is the main-text configuration ``(0.2, 0.6)``; :class:`AdaptiveThresholds`
is the PreCoMem(DT) variant of Appendix D.5, which
personalizes the thresholds through a user-stability index.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Deque, Optional, Tuple

import numpy as np

from .config import GateConfig


class Regime(str, Enum):
    MAINTAIN = "MAINTAIN"
    PROFILE = "PROFILE"
    CORRECT = "CORRECT"
    NONE = "NONE"  # read-only turn, no memory update


@dataclass(frozen=True)
class GateDecision:
    regime: Regime
    x_t: float
    theta_low: float
    theta_high: float
    cold_start: bool = False

    def to_dict(self) -> dict:
        return {
            "regime": self.regime.value,
            "x_t": round(self.x_t, 4),
            "theta_low": round(self.theta_low, 4),
            "theta_high": round(self.theta_high, 4),
            "cold_start": self.cold_start,
        }


def gate(x_t: float, theta_low: float, theta_high: float, single_threshold: bool = False) -> Regime:
    """Eq. (6). With ``single_threshold`` the PROFILE band is removed and the medium band goes to CORRECT."""
    if x_t <= theta_low:
        return Regime.MAINTAIN
    if x_t <= theta_high and not single_threshold:
        return Regime.PROFILE
    return Regime.CORRECT


class ThresholdPolicy:
    """Source of the current ``(theta_low, theta_high)``."""

    def current(self) -> Tuple[float, float]:
        raise NotImplementedError

    def observe_turn(self, regime: Regime, s_eff: float) -> None:
        """Called once per memory-updating turn."""

    def observe_lifespan(self, turns_alive: int) -> None:
        """Called when a belief leaves the active set (pruned or archived)."""

    def reset(self) -> None:
        """Forget per-user statistics (new dialogue)."""


class FixedThresholds(ThresholdPolicy):
    def __init__(self, theta_low: float, theta_high: float) -> None:
        if not theta_low < theta_high:
            raise ValueError("theta_low must be smaller than theta_high")
        self.theta_low = float(theta_low)
        self.theta_high = float(theta_high)

    def current(self) -> Tuple[float, float]:
        return self.theta_low, self.theta_high


class AdaptiveThresholds(ThresholdPolicy):
    """PreCoMem(DT): thresholds driven by the user-stability index ``psi_t``.

    psi_t = ω₁ psi_corr + ω₂ psi_life + ω₃ psi_var, with
      psi_corr = 1 - N_correct / N_total       (how rarely CORRECT fires),
      psi_life = min(1, L_belief / L_ref)      (mean lifespan of retired beliefs),
      psi_var  = exp(-κ · Var[S_eff])          (how steady the surprisal stream is),
    all over a sliding window of ``W`` turns. Thresholds are interpolated
    linearly inside their ranges and smoothed with an EMA of coefficient ``μ``.
    """

    def __init__(self, cfg: GateConfig) -> None:
        if not cfg.adaptive:
            raise ValueError("AdaptiveThresholds requires gate.adaptive = true")
        weights = np.asarray(cfg.adaptive_weights, dtype=np.float64)
        self.weights = weights / max(weights.sum(), 1e-9)
        self.low_range = tuple(float(v) for v in cfg.adaptive_low_range)
        self.high_range = tuple(float(v) for v in cfg.adaptive_high_range)
        self.ema = float(cfg.adaptive_ema)
        self.kappa = float(cfg.adaptive_kappa)
        self.ref_lifespan = max(1, int(cfg.adaptive_ref_lifespan))
        self.window = max(1, int(cfg.adaptive_window))
        self.default_low = float(cfg.theta_low)
        self.default_high = float(cfg.theta_high)
        self.reset()

    def reset(self) -> None:
        self.theta_low = self.default_low
        self.theta_high = self.default_high
        self._regimes: Deque[Regime] = deque(maxlen=self.window)
        self._s_eff: Deque[float] = deque(maxlen=self.window)
        self._lifespans: Deque[int] = deque(maxlen=self.window)

    def current(self) -> Tuple[float, float]:
        return self.theta_low, self.theta_high

    def observe_turn(self, regime: Regime, s_eff: float) -> None:
        if regime is Regime.NONE:
            return
        self._regimes.append(regime)
        self._s_eff.append(float(s_eff))
        self._update()

    def observe_lifespan(self, turns_alive: int) -> None:
        self._lifespans.append(max(0, int(turns_alive)))

    def stability_index(self) -> Optional[float]:
        if not self._regimes:
            return None
        n_correct = sum(1 for r in self._regimes if r is Regime.CORRECT)
        psi_corr = 1.0 - n_correct / len(self._regimes)
        mean_life = float(np.mean(self._lifespans)) if self._lifespans else float(self.ref_lifespan)
        psi_life = min(1.0, mean_life / self.ref_lifespan)
        psi_var = float(np.exp(-self.kappa * float(np.var(self._s_eff))))
        psi = float(np.dot(self.weights, [psi_corr, psi_life, psi_var]))
        return min(max(psi, 0.0), 1.0)

    def _update(self) -> None:
        psi = self.stability_index()
        if psi is None:
            return
        target_low = self.low_range[0] + psi * (self.low_range[1] - self.low_range[0])
        target_high = self.high_range[0] + psi * (self.high_range[1] - self.high_range[0])
        self.theta_low = (1.0 - self.ema) * self.theta_low + self.ema * target_low
        self.theta_high = (1.0 - self.ema) * self.theta_high + self.ema * target_high
        if self.theta_high <= self.theta_low + 1e-3:
            self.theta_high = self.theta_low + 0.05


def build_threshold_policy(cfg: GateConfig) -> ThresholdPolicy:
    return AdaptiveThresholds(cfg) if cfg.adaptive else FixedThresholds(cfg.theta_low, cfg.theta_high)
