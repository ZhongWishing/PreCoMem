import numpy as np
import pytest

from precomem.cli.build_dimobench import DEFAULT_SEEDS, load_seeds
from precomem.cli.calibrate_nli import model_label_map, nll_at_temperature
from precomem.cli.run_benchmark import build_parser, resolve_backbone
from precomem.dimobench import spec_for_seed


def test_backbone_precedence(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("backbone:\n  model: gpt-4.1\n")
    assert resolve_backbone("flag", {"backbone": {"model": "set"}}, str(cfg), "default") == "flag"
    assert resolve_backbone(None, {"backbone": {"model": "set"}}, str(cfg), "default") == "set"
    assert resolve_backbone(None, {}, str(cfg), "default") == "gpt-4.1"
    assert resolve_backbone(None, {}, None, "default") == "default"


def test_run_parser_accepts_repeated_overrides():
    args = build_parser().parse_args(
        ["--benchmark", "locomo", "--data", "x.json", "--set", "ablation.es=true", "--set", "retrieval.top_k=5"]
    )
    assert args.overrides == ["ablation.es=true", "retrieval.top_k=5"]


def test_calibration_maps_mnli_labels_by_name():
    # cross-encoder/nli-deberta-v3-base orders its logits [contradiction, entailment, neutral]
    assert model_label_map({0: "contradiction", 1: "entailment", 2: "neutral"}) == {0: 1, 1: 2, 2: 0}
    logits = np.array([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
    assert nll_at_temperature(logits, [0, 1], 1.0) < nll_at_temperature(logits, [1, 0], 1.0)
    with pytest.raises(ValueError):
        model_label_map({0: "yes", 1: "no"})


@pytest.mark.parametrize("variant, n_turns", [("base", 350), ("xl", 1050)])
def test_shipped_persona_seeds_are_valid(variant, n_turns):
    seeds = load_seeds(DEFAULT_SEEDS, variant)
    assert len(seeds) == 5 and len({s["persona_id"] for s in seeds}) == 5
    for seed in seeds:
        spec = spec_for_seed(seed, variant)
        assert spec.n_turns == n_turns
        assert len(spec.qa_cells()) == (200 if variant == "base" else 420)
