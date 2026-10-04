import pytest

from precomem.config import load_config
from precomem.gate import AdaptiveThresholds, FixedThresholds, Regime, build_threshold_policy, gate


def test_gate_boundaries_follow_eq_6():
    assert gate(0.2, 0.2, 0.6) is Regime.MAINTAIN  # x <= theta_low
    assert gate(0.2000001, 0.2, 0.6) is Regime.PROFILE
    assert gate(0.6, 0.2, 0.6) is Regime.PROFILE  # x <= theta_high
    assert gate(0.61, 0.2, 0.6) is Regime.CORRECT


def test_fixed_policy_from_config():
    cfg = load_config().gate
    policy = build_threshold_policy(cfg)
    assert isinstance(policy, FixedThresholds) and policy.current() == (0.2, 0.6)
    with pytest.raises(ValueError):
        FixedThresholds(0.5, 0.5)


def test_adaptive_policy_moves_towards_conservative_thresholds_for_stable_users():
    cfg = load_config(overrides={"gate": {"adaptive": True}}).gate
    policy = build_threshold_policy(cfg)
    assert isinstance(policy, AdaptiveThresholds)
    low0, high0 = policy.current()
    for _ in range(60):  # a user who never triggers CORRECT and has steady surprisal
        policy.observe_turn(Regime.MAINTAIN, 0.1)
        policy.observe_lifespan(400)
    low, high = policy.current()
    assert policy.stability_index() == pytest.approx(1.0, abs=1e-6)
    assert low > low0 and high > high0
    assert cfg.adaptive_low_range[0] <= low <= cfg.adaptive_low_range[1]
    assert cfg.adaptive_high_range[0] <= high <= cfg.adaptive_high_range[1]

    policy.reset()
    assert policy.current() == (low0, high0)
    for _ in range(60):  # a volatile user: CORRECT fires constantly, beliefs die young
        policy.observe_turn(Regime.CORRECT, 0.9)
        policy.observe_lifespan(2)
    low2, high2 = policy.current()
    assert low2 < low0 and high2 < high0 and high2 > low2


def test_adaptive_policy_ignores_read_only_turns():
    cfg = load_config(overrides={"gate": {"adaptive": True}}).gate
    policy = AdaptiveThresholds(cfg)
    policy.observe_turn(Regime.NONE, 0.5)
    assert policy.stability_index() is None and policy.current() == (0.2, 0.6)
