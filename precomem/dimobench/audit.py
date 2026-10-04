"""Stage 6: human–machine review support.

The paper reserves a 20% stratified sample of every persona's questions
(stratified by temporal phase × interference tag × reasoning category) for
full human review, after a calibration round in which three annotators label
the same 20 questions until Cohen's κ ≥ 0.75. This module

* marks the stratified audit shard on the QA records,
* exports annotation sheets (JSONL + CSV) with the evidence text inlined, and
* provides :func:`cohen_kappa` / :func:`pairwise_kappa` to check calibration.
"""

from __future__ import annotations

import csv
import json
import random
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from .schema import QARecord, Turn

SHEET_COLUMNS = (
    "persona_id",
    "qid",
    "phase",
    "category",
    "interference",
    "anchor_turn",
    "question",
    "gold_answer",
    "evidence_ids",
    "evidence_text",
    "annotator_correct",
    "answer_fix",
    "naturalness_1_10",
    "drift_subtlety_1_10",
    "notes",
)


def mark_audit_shard(qas: List[QARecord], fraction: float = 0.20, seed: int = 1234) -> List[QARecord]:
    """Flag a stratified ``fraction`` of the questions for human review.

    Stratification is by (phase, interference, category). The total is exactly
    ``round(fraction * N)``; strata receive quotas proportional to their size
    (largest-remainder allocation) and members are sampled uniformly within
    each stratum.
    """
    rng = random.Random(seed)
    strata: Dict[Tuple[str, str, str], List[QARecord]] = {}
    for q in qas:
        q.audit = False
        strata.setdefault((q.phase, q.interference.value, q.category.value), []).append(q)
    total = int(round(fraction * len(qas)))
    if total <= 0 or not strata:
        return qas
    keys = sorted(strata)
    exact = {k: fraction * len(strata[k]) for k in keys}
    quotas = {k: int(exact[k]) for k in keys}
    remainder = total - sum(quotas.values())
    for k in sorted(keys, key=lambda k: (-(exact[k] - quotas[k]), k))[: max(remainder, 0)]:
        quotas[k] += 1
    for k in keys:
        members = strata[k]
        rng.shuffle(members)
        for q in members[: quotas[k]]:
            q.audit = True
    return qas


def write_audit_sheets(
    persona_id: str,
    qas: Sequence[QARecord],
    turns: Sequence[Turn],
    out_dir: Path,
    calibration_size: int = 20,
) -> Dict[str, Path]:
    """Write the audit shard (JSONL + CSV) and the shared calibration set."""
    out_dir.mkdir(parents=True, exist_ok=True)
    by_id = {t.dialogue_id: t for t in turns}
    audit = [q for q in qas if q.audit]
    rows = [_row(persona_id, q, by_id) for q in audit]

    jsonl_path = out_dir / f"{persona_id}_audit.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    csv_path = out_dir / f"{persona_id}_audit_sheet.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(SHEET_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)

    calibration = sorted(qas, key=lambda q: q.qid)[:calibration_size]
    cal_path = out_dir / f"{persona_id}_calibration.jsonl"
    with cal_path.open("w", encoding="utf-8") as fh:
        for q in calibration:
            fh.write(json.dumps(_row(persona_id, q, by_id), ensure_ascii=False) + "\n")
    return {"audit_jsonl": jsonl_path, "audit_csv": csv_path, "calibration_jsonl": cal_path}


def _row(persona_id: str, q: QARecord, by_id: Mapping[str, Turn]) -> Dict[str, object]:
    evidence_text = " || ".join(f"{e}: {by_id[e].text}" for e in q.evidence if e in by_id)
    return {
        "persona_id": persona_id,
        "qid": q.qid,
        "phase": q.phase,
        "category": q.category.value,
        "interference": q.interference.value,
        "anchor_turn": q.anchor_turn,
        "question": q.question,
        "gold_answer": q.answer,
        "evidence_ids": " ".join(q.evidence),
        "evidence_text": evidence_text,
        "annotator_correct": "",
        "answer_fix": "",
        "naturalness_1_10": "",
        "drift_subtlety_1_10": "",
        "notes": "",
    }


def audit_summary(qas: Iterable[QARecord]) -> Dict[str, object]:
    qas = list(qas)
    audit = [q for q in qas if q.audit]
    return {
        "n_total": len(qas),
        "n_audit": len(audit),
        "fraction": round(len(audit) / len(qas), 4) if qas else 0.0,
        "by_phase": dict(Counter(q.phase for q in audit)),
        "by_interference": dict(Counter(q.interference.value for q in audit)),
        "by_category": dict(Counter(q.category.value for q in audit)),
    }


def cohen_kappa(labels_a: Sequence[str], labels_b: Sequence[str]) -> float:
    """Cohen's κ between two annotators over the same items."""
    if len(labels_a) != len(labels_b) or not labels_a:
        raise ValueError("label sequences must be non-empty and of equal length")
    n = len(labels_a)
    agree = sum(1 for a, b in zip(labels_a, labels_b) if a == b) / n
    ca, cb = Counter(labels_a), Counter(labels_b)
    expected = sum(ca[k] * cb.get(k, 0) for k in ca) / (n * n)
    if expected >= 1.0:
        return 1.0
    return (agree - expected) / (1.0 - expected)


def pairwise_kappa(labels_by_annotator: Mapping[str, Sequence[str]]) -> Dict[str, float]:
    """κ for every annotator pair plus their mean (calibration threshold in the paper: 0.75)."""
    out: Dict[str, float] = {}
    for (name_a, a), (name_b, b) in combinations(labels_by_annotator.items(), 2):
        out[f"{name_a}~{name_b}"] = round(cohen_kappa(a, b), 4)
    if out:
        out["mean"] = round(sum(out.values()) / len(out), 4)
    return out
