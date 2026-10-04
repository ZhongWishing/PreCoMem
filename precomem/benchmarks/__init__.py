"""Benchmark loaders and the registry used by the CLI.

Every entry maps a benchmark name to its loader, the default backbone used
for that benchmark in the paper, and the QA-prompt key in
``precomem.evaluation.prompts``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Union

from .base import BenchmarkSample, DialogueTurn, Exchange, QAPair, infer_role, load_json, load_jsonl, pair_exchanges
from .dimobench import DiMoBenchLoader, parse_persona
from .locomo import LoCoMoLoader
from .longmemeval import LongMemEvalLoader
from .personamem import PersonaMemLoader

Loader = Callable[[Union[str, Path]], List[BenchmarkSample]]


@dataclass(frozen=True)
class BenchmarkInfo:
    name: str
    load: Loader
    default_backbone: str
    description: str


BENCHMARKS: Dict[str, BenchmarkInfo] = {
    "locomo": BenchmarkInfo("locomo", LoCoMoLoader.load, "gpt-4o-mini", "LoCoMo (Table 1)"),
    "longmemeval": BenchmarkInfo("longmemeval", LongMemEvalLoader.load, "gpt-4.1", "LongMemEval-S (Tables 1 and 7)"),
    "personamem": BenchmarkInfo("personamem", PersonaMemLoader.load, "gpt-4.1", "PersonaMem-v2 (Figure 3)"),
    "dimobench": BenchmarkInfo(
        "dimobench", DiMoBenchLoader.load, "gpt-4o-mini", "DiMoBench / DiMoBench-XL (Section 3.3)"
    ),
}


def get_benchmark(name: str) -> BenchmarkInfo:
    key = name.lower().replace("-", "_")
    if key == "dimobench_xl":
        key = "dimobench"
    if key not in BENCHMARKS:
        raise KeyError(f"Unknown benchmark {name!r}; choose from {sorted(BENCHMARKS)}")
    return BENCHMARKS[key]


__all__ = [
    "BenchmarkInfo",
    "BENCHMARKS",
    "get_benchmark",
    "BenchmarkSample",
    "DialogueTurn",
    "Exchange",
    "QAPair",
    "infer_role",
    "load_json",
    "load_jsonl",
    "pair_exchanges",
    "DiMoBenchLoader",
    "parse_persona",
    "LoCoMoLoader",
    "LongMemEvalLoader",
    "PersonaMemLoader",
]
