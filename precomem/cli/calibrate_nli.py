"""Fit the scalar temperature ``T`` of the NLI verifier on MNLI matched-dev.

Appendix C.2 ("Surprisal components"): ``π_t(m) = softmax(z / T)_contr`` with ``T``
fitted on the MultiNLI matched development split. We minimize the negative
log-likelihood over a grid of temperatures and report the best value; with
``--out`` it is written to an override YAML under ``nli.temperature``.

    precomem-calibrate-nli --model cross-encoder/nli-deberta-v3-base --limit 2000 --out configs/nli.yaml
    precomem-run ... --config configs/nli.yaml

The MNLI dataset labels (0 entailment, 1 neutral, 2 contradiction) are mapped
onto the checkpoint's own label order through its ``id2label`` map, so the
calibration is correct for any label ordering.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

from ..retrieval.embedder import resolve_device
from ..utils.logging import get_logger

_LOGGER = get_logger("cli.calibrate_nli")
MNLI_LABELS = {0: "entailment", 1: "neutral", 2: "contradiction"}


def load_mnli_dev(limit: int) -> Tuple[List[Tuple[str, str]], List[int]]:
    from datasets import load_dataset

    ds = load_dataset("nyu-mll/multi_nli", split="validation_matched")
    pairs: List[Tuple[str, str]] = []
    labels: List[int] = []
    for ex in ds:
        if len(pairs) >= limit:
            break
        if ex["label"] in MNLI_LABELS:
            pairs.append((ex["premise"], ex["hypothesis"]))
            labels.append(int(ex["label"]))
    return pairs, labels


def model_label_map(id2label: Dict[int, str]) -> Dict[int, int]:
    """MNLI label id -> column index of the same label in the model's logits."""
    by_name = {str(v).lower(): int(k) for k, v in id2label.items()}
    out: Dict[int, int] = {}
    for mnli_id, name in MNLI_LABELS.items():
        matches = [idx for lname, idx in by_name.items() if name[:5] in lname]
        if not matches:
            raise ValueError(f"model labels {id2label} do not contain {name!r}")
        out[mnli_id] = matches[0]
    return out


def nll_at_temperature(logits: np.ndarray, target_cols: Sequence[int], temperature: float) -> float:
    z = logits / max(temperature, 1e-3)
    z = z - z.max(axis=1, keepdims=True)
    log_probs = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    rows = np.arange(len(target_cols))
    return float(-log_probs[rows, np.asarray(target_cols)].mean())


def collect_logits(
    model_name: str, pairs: Sequence[Tuple[str, str]], device: str, batch_size: int = 16, max_length: int = 256
):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = resolve_device(device)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device).eval()
    logits: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(pairs), batch_size):
            chunk = pairs[i : i + batch_size]
            enc = tok(
                [a for a, _ in chunk],
                [b for _, b in chunk],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            logits.append(model(**enc).logits.float().cpu().numpy())
    return np.concatenate(logits, axis=0), dict(model.config.id2label)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Calibrate the NLI temperature on MNLI matched-dev")
    parser.add_argument("--model", default="cross-encoder/nli-deberta-v3-base")
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--grid", default="0.5,0.6,0.75,0.85,1.0,1.15,1.25,1.5,1.75,2.0,2.5,3.0")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Override YAML to create or update with nli.temperature (pass it later with --config)",
    )
    args = parser.parse_args(argv)

    pairs, labels = load_mnli_dev(args.limit)
    _LOGGER.info("calibrating %s on %d MNLI matched-dev pairs", args.model, len(pairs))
    logits, id2label = collect_logits(args.model, pairs, args.device)
    col_of = model_label_map({int(k): v for k, v in id2label.items()})
    targets = [col_of[lab] for lab in labels]

    best_t, best_nll = 1.0, float("inf")
    for t in (float(g) for g in args.grid.split(",")):
        nll = nll_at_temperature(logits, targets, t)
        _LOGGER.info("T=%.3f  NLL=%.4f", t, nll)
        if nll < best_nll:
            best_t, best_nll = t, nll
    _LOGGER.info("best T=%.3f (NLL=%.4f)", best_t, best_nll)

    if args.out is None:
        _LOGGER.info("use it with: precomem-run ... --set nli.temperature=%s", best_t)
        return
    cfg = yaml.safe_load(args.out.read_text(encoding="utf-8")) if args.out.exists() else None
    cfg = cfg or {}
    cfg.setdefault("nli", {})["temperature"] = float(best_t)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    _LOGGER.info("wrote nli.temperature=%s to %s (use it with --config %s)", best_t, args.out, args.out)


if __name__ == "__main__":
    main()
