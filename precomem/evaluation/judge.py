"""LLM-as-judge with the three-judge ensemble used in the paper.

Section 3.1 (protocol adopted from MemOS): every prediction is labeled
CORRECT / WRONG by a primary judge (gpt-4.1-mini) and two auxiliary
gemini-2.5-flash judges that run with *distinct system prompts* to reduce
judge correlation, in a blinded setup (the judge never sees which system
produced the answer). Accuracy is the mean of the three binary judgements;
:class:`EnsembleJudge` also exposes the majority verdict.

A judge reply that names no label (or both labels) is re-asked once. If it is
still unusable the vote is marked invalid and left out of the mean instead of
being silently counted as WRONG; ``n_valid`` records how many votes were used.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from ..llm.client import LLMClient, build_llm_client, parse_json_object
from ..utils.logging import get_logger
from .prompts import JUDGE_PROMPTS, build_judge_prompt

_LOGGER = get_logger("evaluation.judge")

CORRECT, WRONG, INVALID = "CORRECT", "WRONG", "INVALID"

PRIMARY_SYSTEM_PROMPT = "Return strict JSON."

# The two auxiliary judges deliberately receive different framings.
AUX_SYSTEM_PROMPTS = (
    "You are a meticulous grader for question answering over conversation memories. "
    "Decide only whether the generated answer conveys the same fact as the gold answer. Return strict JSON.",
    "You are an impartial reviewer comparing a candidate answer with a reference answer. "
    "Accept paraphrases and equivalent formats; reject contradictions, unsupported additions, "
    "and guesses where the reference abstains. Return strict JSON.",
)

_LABEL_RE = re.compile(r"\b(INCORRECT|CORRECT|WRONG)\b")
_RETRY_SUFFIX = '\n\nReply with exactly {"label": "CORRECT"} or {"label": "WRONG"} and nothing else.'


def parse_label(text: str) -> Optional[str]:
    """Map a judge reply to CORRECT / WRONG; ``None`` if it names no label or both."""
    parsed = parse_json_object(text) or {}
    label = str(parsed.get("label", "")).strip().upper()
    if label in {CORRECT, WRONG}:
        return label
    if label == "INCORRECT":
        return WRONG
    found = {WRONG if hit == "INCORRECT" else hit for hit in _LABEL_RE.findall((text or "").upper())}
    return found.pop() if len(found) == 1 else None


@dataclass
class JudgeResult:
    judge: str
    label: str
    raw: str = ""

    @property
    def valid(self) -> bool:
        return self.label in {CORRECT, WRONG}

    @property
    def correct(self) -> bool:
        return self.label == CORRECT

    def to_dict(self) -> Dict[str, Any]:
        return {"judge": self.judge, "label": self.label, "correct": self.correct, "valid": self.valid}


@dataclass
class EnsembleVerdict:
    per_judge: List[JudgeResult] = field(default_factory=list)

    @property
    def n_valid(self) -> int:
        return sum(r.valid for r in self.per_judge)

    @property
    def score(self) -> float:
        """Mean of the valid binary judgements (the paper's accuracy definition)."""
        valid = [r for r in self.per_judge if r.valid]
        return sum(r.correct for r in valid) / len(valid) if valid else 0.0

    @property
    def majority_correct(self) -> bool:
        valid = [r for r in self.per_judge if r.valid]
        return sum(r.correct for r in valid) * 2 > len(valid)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "per_judge": [r.to_dict() for r in self.per_judge],
            "score": round(self.score, 4),
            "n_valid": self.n_valid,
            "majority_correct": self.majority_correct,
        }


class LLMJudge:
    def __init__(self, llm: LLMClient, name: Optional[str] = None, system_prompt: str = PRIMARY_SYSTEM_PROMPT) -> None:
        self.llm = llm
        self.name = name or llm.model
        self.system_prompt = system_prompt

    def judge(
        self,
        benchmark: str,
        question: str,
        gold: str,
        prediction: str,
        question_type: Optional[str] = None,
        options: Optional[Sequence[str]] = None,
    ) -> JudgeResult:
        if benchmark.lower() not in JUDGE_PROMPTS:
            raise ValueError(f"Unknown benchmark: {benchmark}")
        prompt = build_judge_prompt(benchmark, question, gold, prediction, question_type, list(options or []))
        text = self.llm.complete(prompt, system=self.system_prompt, json_mode=True).text
        label = parse_label(text)
        if label is None:
            text = self.llm.complete(prompt + _RETRY_SUFFIX, system=self.system_prompt, json_mode=True).text
            label = parse_label(text)
        if label is None:
            _LOGGER.warning("Judge %s returned no usable label; the vote is excluded.", self.name)
            label = INVALID
        return JudgeResult(judge=self.name, label=label, raw=text)


class EnsembleJudge:
    def __init__(self, judges: Sequence[LLMJudge]) -> None:
        if not judges:
            raise ValueError("EnsembleJudge requires at least one judge")
        self.judges = list(judges)

    def judge(
        self,
        benchmark: str,
        question: str,
        gold: str,
        prediction: str,
        question_type: Optional[str] = None,
        options: Optional[Sequence[str]] = None,
    ) -> EnsembleVerdict:
        return EnsembleVerdict(
            per_judge=[j.judge(benchmark, question, gold, prediction, question_type, options) for j in self.judges]
        )


def build_ensemble_judge(
    primary_model: str, aux_models: Sequence[str] = (), base_url: Optional[str] = None
) -> EnsembleJudge:
    """Primary judge plus auxiliary judges; auxiliaries that cannot be built are skipped with a warning.

    ``base_url`` applies to OpenAI-compatible judges only (Gemini judges ignore it).
    """
    judges = [
        LLMJudge(build_llm_client(primary_model, base_url=base_url, temperature=0.0, max_tokens=64), name=primary_model)
    ]
    for i, model in enumerate(aux_models):
        system_prompt = AUX_SYSTEM_PROMPTS[i % len(AUX_SYSTEM_PROMPTS)]
        try:
            client = build_llm_client(model, base_url=base_url, temperature=0.0, max_tokens=64)
            judges.append(LLMJudge(client, name=f"{model}#{i + 1}", system_prompt=system_prompt))
        except Exception as exc:
            _LOGGER.warning("Skipping auxiliary judge %s: %s", model, exc)
    return EnsembleJudge(judges)
