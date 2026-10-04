"""Build DiMoBench (base) or DiMoBench-XL from persona seeds.

    python -m precomem.cli.build_dimobench --variant base --out data/dimobench
    python -m precomem.cli.build_dimobench --variant xl   --out data/dimobench_xl

Generator defaults to GPT-4o and verifier to Gemini-2.5-Pro (the paper's
generator–verifier separation). Use ``--max-sessions`` / ``--max-qa`` for a
cheap smoke run, and ``--resume`` (default) to continue an interrupted build
from the stage checkpoints in ``<out>/_work``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from ..dimobench.audit import audit_summary, write_audit_sheets
from ..dimobench.pipeline import BuildOptions, DiMoBenchBuilder, spec_for_seed
from ..dimobench.schema import QARecord, Turn
from ..llm.client import LLMClient, build_llm_client
from ..utils.logging import get_logger, set_seed

DEFAULT_SEEDS = Path(__file__).resolve().parents[1] / "configs" / "dimobench_personas.yaml"
_LOGGER = get_logger("cli.build_dimobench")


def load_seeds(path: Path, variant: str) -> List[Dict[str, Any]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    seeds = data.get(variant) if isinstance(data, dict) else data
    if not seeds:
        raise SystemExit(f"no '{variant}' seeds found in {path}")
    return list(seeds)


def build_clients(args: argparse.Namespace) -> Tuple[LLMClient, LLMClient]:
    generator = build_llm_client(args.generator_model, temperature=0.7, max_tokens=4096)
    try:
        verifier = build_llm_client(args.verifier_model, temperature=0.0, max_tokens=1024)
    except Exception as exc:
        fallback = args.verifier_fallback
        _LOGGER.warning("verifier %s unavailable (%s); using %s instead", args.verifier_model, exc, fallback)
        verifier = build_llm_client(fallback, temperature=0.0, max_tokens=1024)
    return generator, verifier


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Build the DiMoBench suite")
    parser.add_argument("--variant", choices=["base", "xl"], default="base")
    parser.add_argument("--seeds", type=Path, default=DEFAULT_SEEDS, help="YAML with persona seeds")
    parser.add_argument("--personas", default=None, help="Comma-separated persona ids to build (default: all)")
    parser.add_argument("--out", type=Path, default=None, help="Output directory (default data/dimobench[_xl])")
    parser.add_argument("--generator-model", default="gpt-4o")
    parser.add_argument("--verifier-model", default="gemini-2.5-pro")
    parser.add_argument("--verifier-fallback", default="gpt-4.1", help="Used when the verifier provider is unavailable")
    parser.add_argument("--rounds", type=int, default=3, help="Max verifier rounds per question")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-sessions", type=int, default=None, help="Smoke runs: truncate the dialogue")
    parser.add_argument("--max-qa", type=int, default=None, help="Smoke runs: cap the number of questions")
    parser.add_argument(
        "--keep-unresolved", action="store_true", help="Keep questions still marked REVISE after the last round"
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--no-resume", action="store_true", help="Ignore stage checkpoints")
    args = parser.parse_args(argv)

    set_seed(args.seed)
    out_dir = args.out or Path("data") / ("dimobench" if args.variant == "base" else "dimobench_xl")
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = out_dir / "_work"

    seeds = load_seeds(args.seeds, args.variant)
    if args.personas:
        wanted = {p.strip() for p in args.personas.split(",") if p.strip()}
        seeds = [s for s in seeds if s["persona_id"] in wanted]
        if not seeds:
            raise SystemExit(f"none of {sorted(wanted)} found in {args.seeds}")

    generator, verifier = build_clients(args)
    options = BuildOptions(
        rounds=args.rounds,
        workers=args.workers,
        max_sessions=args.max_sessions,
        max_qa=args.max_qa,
        keep_unresolved=args.keep_unresolved,
        seed=args.seed,
    )

    written: List[str] = []
    report: Dict[str, Any] = {"variant": args.variant, "personas": {}}
    for seed in seeds:
        spec = spec_for_seed(seed, args.variant, rng_seed=args.seed)
        builder = DiMoBenchBuilder(spec, generator, verifier, options, work_dir=work_dir, resume=not args.no_resume)
        try:
            dataset = builder.build(seed)
        except Exception as exc:
            _LOGGER.exception("persona %s failed: %s", seed["persona_id"], exc)
            report["personas"][seed["persona_id"]] = {"error": str(exc)}
            continue
        path = out_dir / f"{dataset['persona_id']}.json"
        path.write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")
        written.append(path.name)

        qas = [QARecord.from_dict(q) for q in dataset["qa_pairs"]]
        turns = [Turn.from_dict(t) for t in dataset["dialogue"]]
        sheets = write_audit_sheets(dataset["persona_id"], qas, turns, out_dir / "audit")
        report["personas"][dataset["persona_id"]] = {
            "file": path.name,
            "stats": dataset["stats"],
            "audit": audit_summary(qas),
            "audit_files": {k: str(v) for k, v in sheets.items()},
        }
        _LOGGER.info("wrote %s (%d turns, %d questions)", path, len(turns), len(qas))

    suite = {
        "benchmark": "dimobench" if args.variant == "base" else "dimobench_xl",
        "personas": written,
        "n_personas": len(written),
        "generator_model": generator.model,
        "verifier_model": verifier.model,
    }
    (out_dir / "suite.json").write_text(json.dumps(suite, indent=2), encoding="utf-8")
    (out_dir / "build_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _LOGGER.info("built %d / %d personas -> %s", len(written), len(seeds), out_dir)
    if len(written) < len(seeds):
        sys.exit(1)


if __name__ == "__main__":
    main()
