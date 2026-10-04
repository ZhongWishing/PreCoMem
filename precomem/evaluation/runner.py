"""Benchmark runner: stream a dialogue through PreCoMem, answer its questions, judge.

Per benchmark sample the runner

1. builds a fresh :class:`~precomem.agent.PreCoMem` (heavy backends are shared),
2. replays the dialogue exchange by exchange through :meth:`PreCoMem.observe`,
   so MAINTAIN / PROFILE / CORRECT fire exactly as they would online,
3. answers each question with question-conditioned Top-K recall
   (:meth:`PreCoMem.recall`) and the benchmark's QA prompt. Questions that
   carry an ``anchor_turn`` (DiMoBench) are asked as soon as the dialogue
   passes that turn, so the answer reflects the belief state at that moment;
   all other questions are asked after the full dialogue,
4. scores predictions with the judge ensemble and aggregates across seeds.

Samples are processed concurrently by a thread pool; the work is LLM-bound,
and the shared embedder / NLI verifier serialize their own forward passes.
"""

from __future__ import annotations

import json
import platform
import re
import subprocess
import time
import zlib
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

from .. import __version__
from ..agent import Backends, PreCoMem, TurnMeta, build_backends
from ..benchmarks.base import BenchmarkSample, QAPair, pair_exchanges
from ..config import PreCoMemConfig
from ..llm.client import LLMClient, TokenUsage
from ..utils.logging import get_logger, set_seed
from .judge import EnsembleJudge, build_ensemble_judge
from .metrics import aggregate_seeds, summarize_records
from .prompts import build_qa_prompt

_LOGGER = get_logger("evaluation.runner")
_ANSWER_RE = re.compile(r"^\s*\**answer\**\s*[:：]\s*(.*)$", re.IGNORECASE)


@dataclass
class RunSettings:
    benchmark: str
    data_path: str
    out_dir: str
    config: PreCoMemConfig
    seeds: Sequence[int] = (13, 42, 1234)
    limit: Optional[int] = None
    skip_judge: bool = False
    judge_primary: str = "gpt-4.1-mini"
    judge_aux: Sequence[str] = ("gemini-2.5-flash", "gemini-2.5-flash")
    judge_base_url: Optional[str] = None
    workers: int = 4
    save_memory: bool = True
    run_name: Optional[str] = None


@dataclass
class SampleResult:
    sample_id: str
    records: List[Dict[str, Any]]
    agent: PreCoMem
    elapsed_s: float
    tokens: TokenUsage = field(default_factory=TokenUsage)


def extract_answer(text: str) -> str:
    """Return the content of the last ``Answer:`` line, or the whole reply."""
    for line in reversed((text or "").splitlines()):
        m = _ANSWER_RE.match(line)
        if m:
            return m.group(1).strip()
    return (text or "").strip()


class BenchmarkRunner:
    def __init__(
        self,
        settings: RunSettings,
        backends: Optional[Backends] = None,
        judge: Optional[EnsembleJudge] = None,
        qa_llm: Optional[LLMClient] = None,
    ) -> None:
        self.settings = settings
        self.backends = backends or build_backends(settings.config)
        # Questions are answered by the backbone unless a dedicated client is injected.
        self.qa_llm = qa_llm
        self.judge = judge
        if self.judge is None and not settings.skip_judge:
            self.judge = build_ensemble_judge(
                settings.judge_primary, [m for m in settings.judge_aux if m], base_url=settings.judge_base_url
            )

    # ------------------------------------------------------------------
    def run(self, samples: Sequence[BenchmarkSample]) -> Dict[str, Any]:
        s = self.settings
        if s.limit is not None:
            samples = list(samples)[: s.limit]
        run_dir = Path(s.out_dir) / (s.run_name or f"{s.benchmark}_{time.strftime('%Y%m%d_%H%M%S')}")
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "run_meta.json").write_text(json.dumps(self._run_meta(len(samples)), indent=2), encoding="utf-8")

        summary: Dict[str, Any] = {
            "benchmark": s.benchmark,
            "backbone": s.config.backbone.model,
            "n_samples": len(samples),
            "seeds": list(s.seeds),
            "metrics_by_seed": {},
            "tokens_by_seed": {},
        }
        for seed in s.seeds:
            set_seed(seed)
            seed_dir = run_dir / f"seed_{seed}"
            records = self._run_seed(samples, seed, seed_dir)
            if self.judge is not None:
                records = self._judge_records(records)
            with (seed_dir / "predictions.jsonl").open("w", encoding="utf-8") as fh:
                for rec in records:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            summary["metrics_by_seed"][str(seed)] = (
                summarize_records(records) if self.judge is not None else {"n": len(records)}
            )
            summary["tokens_by_seed"][str(seed)] = self._token_summary(records)
            _LOGGER.info("seed %d done: %s", seed, summary["metrics_by_seed"][str(seed)].get("overall", "n/a"))

        if self.judge is not None:
            summary["metrics"] = aggregate_seeds(summary["metrics_by_seed"])
        (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        _LOGGER.info("Run complete: %s", run_dir)
        summary["run_dir"] = str(run_dir)
        return summary

    # ------------------------------------------------------------------
    def _run_seed(self, samples: Sequence[BenchmarkSample], seed: int, seed_dir: Path) -> List[Dict[str, Any]]:
        (seed_dir / "traces").mkdir(parents=True, exist_ok=True)
        if self.settings.save_memory:
            (seed_dir / "memory").mkdir(parents=True, exist_ok=True)

        def work(sample: BenchmarkSample) -> SampleResult:
            return self.evaluate_sample(sample, seed)

        results: List[SampleResult] = []
        workers = max(1, int(self.settings.workers))
        if workers == 1:
            results = [work(sm) for sm in samples]
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(work, samples))

        records: List[Dict[str, Any]] = []
        for res in results:
            res.agent.dump_history(seed_dir / "traces" / f"{_safe(res.sample_id)}.jsonl")
            if self.settings.save_memory:
                res.agent.save_memory(seed_dir / "memory" / f"{_safe(res.sample_id)}.json")
            records.extend(res.records)
            _LOGGER.info(
                "[seed %d] %s: %d turns, %d questions, %.1fs, %d tokens",
                seed,
                res.sample_id,
                res.agent.state.turn,
                len(res.records),
                res.elapsed_s,
                res.tokens.total_tokens,
            )
        return records

    def evaluate_sample(self, sample: BenchmarkSample, seed: int) -> SampleResult:
        """Replay one dialogue and answer its questions (anchored ones mid-stream)."""
        t0 = time.perf_counter()
        agent = PreCoMem(self.settings.config, self.backends, seed=zlib.crc32(f"{seed}:{sample.sample_id}".encode()))

        anchored, unanchored = _split_by_anchor(sample.qa_pairs)
        pending: Deque[QAPair] = deque(anchored)
        records: List[Dict[str, Any]] = []

        reached = 0  # last dialogue turn index replayed so far
        for exchange in pair_exchanges(sample.dialogue):
            agent.observe(
                exchange.memory_input(),
                exchange.assistant,
                meta=TurnMeta(
                    speaker=exchange.speaker,
                    session=exchange.session,
                    dialogue_id=exchange.dialogue_id,
                    timestamp=exchange.timestamp,
                ),
            )
            reached = max([reached, *exchange.turn_indices])
            while pending and int(pending[0].meta["anchor_turn"]) <= reached:
                records.append(self._answer(agent, sample, pending.popleft(), seed, reached))
        for qa in list(pending) + unanchored:
            records.append(self._answer(agent, sample, qa, seed, reached))

        tokens = TokenUsage(agent.llm.usage.prompt_tokens, agent.llm.usage.completion_tokens)
        return SampleResult(sample.sample_id, records, agent, time.perf_counter() - t0, tokens)

    def _answer(
        self, agent: PreCoMem, sample: BenchmarkSample, qa: QAPair, seed: int, dialogue_turn: int
    ) -> Dict[str, Any]:
        evidence = agent.recall(qa.question)
        prompt = build_qa_prompt(
            self.settings.benchmark,
            qa.question,
            evidence.render(),
            qa.options,
            question_date=qa.meta.get("question_date"),
        )
        response = (self.qa_llm or agent.llm).complete(prompt)
        return {
            "qid": qa.qid,
            "sample_id": sample.sample_id,
            "seed": seed,
            "question": qa.question,
            "category": qa.category,
            "options": qa.options,
            "gold": qa.answer,
            "prediction": response.text.strip(),
            "answer": extract_answer(response.text),
            "evidence": qa.evidence,
            "meta": qa.meta,
            "asked_after_dialogue_turn": dialogue_turn,  # index of the last replayed dialogue turn
            "memory_turn": agent.state.turn,  # number of exchanges written to memory so far
            "retrieved_ids": [m.item_id for m in evidence.beliefs],
            "retrieved_weights": [round(m.weight, 4) for m in evidence.beliefs],
            # Top-k contamination (Appendix D.7): share of slots held by beliefs with w < 0.1,
            # their share of the score mass, and the mean retrieved confidence.
            "retrieved_low_share": _share(evidence.beliefs, evidence.retrieval.scores),
            "retrieved_mean_weight": round(sum(m.weight for m in evidence.beliefs) / len(evidence.beliefs), 4)
            if evidence.beliefs
            else None,
            "n_excerpts": len(evidence.excerpts),
            "qa_tokens": response.usage.to_dict(),
        }

    def _judge_records(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        assert self.judge is not None

        def judge_one(rec: Dict[str, Any]) -> Dict[str, Any]:
            verdict = self.judge.judge(
                benchmark=self.settings.benchmark,
                question=rec["question"],
                gold=rec["gold"],
                prediction=rec["answer"] or rec["prediction"],
                question_type=rec.get("category"),
                options=rec.get("options"),
            )
            out = dict(rec)
            out["judge"] = verdict.to_dict()
            out["score"] = verdict.score
            return out

        workers = max(1, int(self.settings.workers))
        if workers == 1:
            return [judge_one(r) for r in records]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(judge_one, records))

    # ------------------------------------------------------------------
    @staticmethod
    def _token_summary(records: Sequence[Dict[str, Any]]) -> Dict[str, float]:
        if not records:
            return {}
        prompt = [r["qa_tokens"]["prompt_tokens"] for r in records if "qa_tokens" in r]
        return {
            "qa_prompt_tokens_mean": round(sum(prompt) / max(len(prompt), 1), 1),
            "qa_prompt_tokens_max": max(prompt) if prompt else 0,
        }

    def _run_meta(self, n_samples: int) -> Dict[str, Any]:
        s = self.settings
        return {
            "benchmark": s.benchmark,
            "data_path": s.data_path,
            "n_samples": n_samples,
            "seeds": list(s.seeds),
            "config": s.config.to_dict(),
            "embedder": getattr(self.backends.embedder, "name", "?"),
            "verifier": getattr(self.backends.verifier, "name", "?"),
            "judges": [] if s.skip_judge else [s.judge_primary, *[m for m in s.judge_aux if m]],
            "precomem_version": __version__,
            "python": platform.python_version(),
            "git_commit": _git_commit(),
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }


def _share(beliefs: Sequence[Any], scores: Sequence[float], threshold: float = 0.1) -> Dict[str, Optional[float]]:
    if not beliefs:
        return {"slots": None, "score_mass": None}
    low = [i for i, m in enumerate(beliefs) if m.weight < threshold]
    total = float(sum(scores)) or 1.0
    return {
        "slots": round(len(low) / len(beliefs), 4),
        "score_mass": round(sum(scores[i] for i in low) / total, 4),
    }


def _split_by_anchor(qa_pairs: Sequence[QAPair]) -> Tuple[List[QAPair], List[QAPair]]:
    anchored = [q for q in qa_pairs if q.meta.get("anchor_turn") is not None]
    anchored.sort(key=lambda q: int(q.meta["anchor_turn"]))
    unanchored = [q for q in qa_pairs if q.meta.get("anchor_turn") is None]
    return anchored, unanchored


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)[:120]


def _git_commit() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=Path(__file__).resolve().parent,
        )
        return out.stdout.strip() or None
    except Exception:
        return None
