"""Run PreCoMem on a benchmark and score it with the judge ensemble.

    python -m precomem.cli.run_benchmark --benchmark locomo --data data/locomo10.json
    python -m precomem.cli.run_benchmark --benchmark longmemeval --data data/longmemeval_s.json --backbone gpt-4.1
    python -m precomem.cli.run_benchmark --benchmark dimobench --data data/dimobench/suite.json \\
        --set gate.adaptive=true --seeds 13 --workers 2

Any config field can be overridden with ``--set section.key=value``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from ..benchmarks import BENCHMARKS, get_benchmark
from ..config import load_config, merge_overrides
from ..evaluation.runner import BenchmarkRunner, RunSettings
from ..utils.logging import get_logger

_LOGGER = get_logger("cli.run_benchmark")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run PreCoMem on a long-term memory benchmark")
    parser.add_argument("--benchmark", required=True, choices=sorted(BENCHMARKS) + ["dimobench_xl"])
    parser.add_argument("--data", required=True, help="Path to the benchmark file (or DiMoBench suite.json)")
    parser.add_argument("--config", default=None, help="YAML overriding configs/default.yaml")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Config override, e.g. --set retrieval.top_k=5 (repeatable)",
    )
    parser.add_argument(
        "--backbone", default=None, help="Backbone model (default: the paper's choice for the benchmark)"
    )
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible endpoint for the backbone")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--seeds", default=None, help="Comma-separated seeds (default from config: 13,42,1234)")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N samples")
    parser.add_argument("--workers", type=int, default=None, help="Concurrent samples (default from config)")
    parser.add_argument("--skip-judge", action="store_true", help="Inference only; write predictions without scores")
    parser.add_argument("--judge-primary", default=None)
    parser.add_argument("--judge-aux", default=None, help="Comma-separated auxiliary judges ('' for none)")
    parser.add_argument("--judge-base-url", default=None, help="OpenAI-compatible endpoint for OpenAI-style judges")
    parser.add_argument("--no-save-memory", action="store_true", help="Do not dump the final belief store per sample")
    return parser


def resolve_backbone(
    flag: Optional[str], overrides: Dict[str, Any], config_path: Optional[str], benchmark_default: str
) -> str:
    """Precedence: ``--backbone`` > ``--set backbone.model`` > ``--config`` YAML > the benchmark's paper default."""
    if flag:
        return flag
    from_set = overrides.get("backbone", {}).get("model")
    if from_set:
        return str(from_set)
    if config_path:
        user_cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
        from_yaml = (user_cfg.get("backbone") or {}).get("model")
        if from_yaml:
            return str(from_yaml)
    return benchmark_default


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    info = get_benchmark(args.benchmark)

    overrides = merge_overrides(args.overrides)
    backbone = resolve_backbone(args.backbone, overrides, args.config, info.default_backbone)
    overrides.setdefault("backbone", {})["model"] = backbone
    if args.base_url:
        overrides["backbone"]["base_url"] = args.base_url
    cfg = load_config(args.config, overrides=overrides)

    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else list(cfg.evaluation.seeds)
    judge_aux = (
        [m for m in args.judge_aux.split(",") if m.strip()]
        if args.judge_aux is not None
        else list(cfg.evaluation.aux_judges)
    )
    settings = RunSettings(
        benchmark=info.name,
        data_path=args.data,
        out_dir=args.out_dir,
        config=cfg,
        seeds=seeds,
        limit=args.limit,
        skip_judge=args.skip_judge,
        judge_primary=args.judge_primary or cfg.evaluation.primary_judge,
        judge_aux=judge_aux,
        judge_base_url=args.judge_base_url,
        workers=args.workers or cfg.evaluation.num_workers,
        save_memory=not args.no_save_memory,
        run_name=args.run_name,
    )
    samples = info.load(args.data)
    _LOGGER.info("%s: %d samples, backbone=%s, seeds=%s", info.name, len(samples), backbone, seeds)
    summary = BenchmarkRunner(settings).run(samples)
    metrics = summary.get("metrics")
    if metrics:
        _LOGGER.info("overall: %.2f ± %.2f", metrics["overall_mean"], metrics["overall_std"])
        for cat, m in metrics["per_category"].items():
            _LOGGER.info("  %-28s %.2f ± %.2f", cat, m["mean"], m["std"])


if __name__ == "__main__":
    main()
