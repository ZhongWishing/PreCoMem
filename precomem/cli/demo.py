"""End-to-end demo on the canonical coffee-to-tea example (Figure 1).

    export OPENAI_API_KEY=...
    python -m precomem.cli.demo [--offline]

``--offline`` swaps in the hashing embedder and the heuristic verifier (no
model downloads) but still needs an LLM backbone for the sub-agents.
"""

from __future__ import annotations

import argparse
from typing import List, Optional

from ..agent import PreCoMem
from ..config import load_config
from ..utils.logging import set_seed

DIALOGUE = [
    "I usually start my mornings with coffee.",
    "I tried matcha at a new cafe today, pretty refreshing.",
    "I've gone back for matcha several times this week.",
    "Coffee has been making my stomach uneasy lately.",
    "Switching to tea is becoming part of my routine.",
    "I really don't drink coffee anymore at all.",
    "Matcha lattes are now my non-negotiable morning ritual.",
]
QUESTION = "What drink should I prepare for my morning routine now?"


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="PreCoMem demo")
    parser.add_argument("--backbone", default=None)
    parser.add_argument("--offline", action="store_true", help="hashing embedder + heuristic NLI (no model downloads)")
    args = parser.parse_args(argv)

    overrides = {}
    if args.backbone:
        overrides["backbone"] = {"model": args.backbone}
    if args.offline:
        overrides["embedding"] = {"provider": "hashing"}
        overrides["nli"] = {"model": "heuristic"}
    set_seed(42)
    agent = PreCoMem(load_config(overrides=overrides))

    for utterance in DIALOGUE:
        trace = agent.step(utterance)
        print(trace.one_line(), f"| {utterance}")
    print("\n=== Belief store ===")
    print(agent.render_memory())
    print("\nQ:", QUESTION)
    print("A:", agent.query(QUESTION))


if __name__ == "__main__":
    main()
