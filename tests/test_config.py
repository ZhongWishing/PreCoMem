import pytest

from precomem.config import ConfigError, load_config, merge_overrides, parse_override


def test_defaults_match_paper_hyperparameters():
    cfg = load_config()
    assert (cfg.gate.theta_low, cfg.gate.theta_high) == (0.2, 0.6)
    assert cfg.retrieval.top_k == 10
    assert cfg.retrieval.alpha.as_tuple() == (0.4, 0.4, 0.2)
    assert cfg.surprisal.lambda_ == 0.5
    assert cfg.operators.correct.beta == 1.0
    assert cfg.lifecycle.decay_gamma == 0.05
    assert cfg.lifecycle.promote_threshold == 0.80
    assert cfg.lifecycle.hypothesis_prune == 0.10
    assert cfg.operators.profile.explicit_band == (0.60, 0.70)
    assert cfg.operators.profile.latent_band == (0.50, 0.59)
    assert cfg.evaluation.seeds == (13, 42, 1234)


def test_overrides_and_roundtrip():
    cfg = load_config(overrides={"backbone": {"model": "gpt-4.1"}, "retrieval": {"top_k": 5}})
    assert cfg.backbone.model == "gpt-4.1"
    assert cfg.retrieval.top_k == 5
    again = cfg.with_overrides({"gate": {"adaptive": True}})
    assert again.gate.adaptive is True and again.retrieval.top_k == 5
    assert again.to_dict()["surprisal"]["lambda"] == 0.5  # YAML key preserved


@pytest.mark.parametrize(
    "override",
    [
        {"gate": {"theta_low": 0.7}},  # low >= high
        {"operators": {"profile": {"explicit_band": [0.6, 0.9]}}},  # band crosses tau_promote
        {"gate": {"theta_lo": 0.1}},  # unknown key
        {"retrieval": {"top_k": 0}},
        {"surprisal": {"lambda": 1.5}},
        {"backbone": {"temperature": "hot"}},  # wrong type
    ],
)
def test_invalid_configs_are_rejected(override):
    with pytest.raises(ConfigError):
        load_config(overrides=override)


def test_cli_override_parsing():
    assert parse_override("gate.theta_high=0.7") == {"gate": {"theta_high": 0.7}}
    assert parse_override("gate.adaptive=true") == {"gate": {"adaptive": True}}
    merged = merge_overrides(["retrieval.alpha.topic=0", "retrieval.top_k=5"])
    assert merged == {"retrieval": {"alpha": {"topic": 0}, "top_k": 5}}
    cfg = load_config(overrides=merged)
    assert cfg.retrieval.alpha.topic == 0.0
