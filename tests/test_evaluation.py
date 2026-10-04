import json

import pytest

from precomem.agent import Backends
from precomem.benchmarks import BenchmarkSample, DialogueTurn, QAPair
from precomem.evaluation import (
    AUX_SYSTEM_PROMPTS,
    PRIMARY_SYSTEM_PROMPT,
    BenchmarkRunner,
    EnsembleJudge,
    LLMJudge,
    RunSettings,
    aggregate_seeds,
    build_judge_prompt,
    build_qa_prompt,
    extract_answer,
    mean_std,
    parse_label,
    summarize_records,
)
from precomem.retrieval import HashingEmbedder
from precomem.surprisal import HeuristicVerifier
from tests.conftest import ScriptedLLM


def test_extract_answer_takes_last_answer_line():
    assert extract_answer("Thought: x\nAnswer: 5\nAnswer: 7") == "7"
    assert extract_answer("**Answer**: Unknown") == "Unknown"
    assert extract_answer("no marker here") == "no marker here"
    assert extract_answer("") == ""


def test_qa_and_judge_prompt_builders():
    mcq = build_qa_prompt("personamem", "music?", "ctx", options=["A. jazz", "B. rock"])
    assert "A. jazz\nB. rock" in mcq and "Output ONLY the option label" in mcq
    open_ended = build_qa_prompt("personamem", "music?", "ctx", options=None)
    assert "Option" not in open_ended and "Answer: [Your response]" in open_ended
    assert "Options:" in build_judge_prompt("personamem", "q", "(a)", "a", options=["(a) x", "(b) y"])
    assert "Options:" not in build_judge_prompt("personamem", "q", "likes jazz", "jazz night")
    lme = build_qa_prompt(
        "longmemeval", "How many days ago did I buy it?", "ctx", question_date="2023/05/30 (Tue) 23:40"
    )
    assert "Question: How many days ago did I buy it?\nCurrent date: 2023/05/30 (Tue) 23:40" in lme
    j = build_judge_prompt("longmemeval", "q", "gold", "pred", question_type="KU")
    assert "Question type: KU" in j
    with pytest.raises(ValueError):
        build_judge_prompt("unknown", "q", "g", "p")


@pytest.mark.parametrize(
    "reply, label",
    [
        ('{"label": "CORRECT"}', "CORRECT"),
        ('{"label": "wrong"}', "WRONG"),
        ('{"label": "INCORRECT"}', "WRONG"),
        ("The answer is INCORRECT.", "WRONG"),  # must not be read as CORRECT
        ("Label: CORRECT", "CORRECT"),
        ('{"label": "CORRECT" | "WRONG"}', None),  # template echoed back: ambiguous
        ("", None),
    ],
)
def test_parse_label(reply, label):
    assert parse_label(reply) == label


def test_unusable_judge_votes_are_retried_then_excluded():
    flaky = ScriptedLLM()
    replies = iter(["I am not sure.", '{"label": "WRONG"}'])
    flaky.custom = lambda prompt: next(replies) if "label an answer" in prompt else None
    recovered = LLMJudge(flaky, "flaky").judge("locomo", "q", "gold", "pred")
    assert recovered.label == "WRONG" and recovered.valid and len(flaky.prompts) == 2

    silent = ScriptedLLM()
    silent.custom = lambda prompt: "" if "label an answer" in prompt else None
    yes = ScriptedLLM()
    verdict = EnsembleJudge([LLMJudge(yes, "j1"), LLMJudge(silent, "j2")]).judge("locomo", "q", "gold", "pred")
    assert verdict.n_valid == 1 and verdict.score == 1.0  # the invalid vote is not counted as WRONG
    assert verdict.to_dict()["per_judge"][1]["label"] == "INVALID"


def test_metrics_aggregation():
    assert mean_std([1.0, 3.0]) == (2.0, pytest.approx(1.4142, abs=1e-3))
    assert mean_std([5.0]) == (5.0, 0.0)
    recs = [{"category": "a", "score": 1.0}, {"category": "a", "score": 0.0}, {"category": "b", "score": 1.0}]
    summary = summarize_records(recs)
    assert summary["overall"] == pytest.approx(66.6667, abs=1e-3)
    assert summary["per_category"]["a"] == {"n": 2, "accuracy": 50.0}
    agg = aggregate_seeds({"13": summary, "42": summarize_records(recs[:2])})
    assert agg["overall_mean"] == pytest.approx((66.6667 + 50.0) / 2, abs=1e-3)
    assert agg["per_category"]["b"]["n_seeds"] == 1


def test_ensemble_judge_scores_mean_of_binary_votes_with_distinct_system_prompts():
    yes, no, aux = ScriptedLLM(), ScriptedLLM(), ScriptedLLM()
    no.judge_label = "WRONG"
    ens = EnsembleJudge(
        [
            LLMJudge(yes, "j1"),
            LLMJudge(no, "j2", system_prompt=AUX_SYSTEM_PROMPTS[0]),
            LLMJudge(aux, "j3", system_prompt=AUX_SYSTEM_PROMPTS[1]),
        ]
    )
    verdict = ens.judge("locomo", "q", "gold", "pred")
    assert verdict.score == pytest.approx(2 / 3) and verdict.majority_correct is True
    assert [r.judge for r in verdict.per_judge] == ["j1", "j2", "j3"]
    assert (
        yes.systems == [PRIMARY_SYSTEM_PROMPT]
        and no.systems == [AUX_SYSTEM_PROMPTS[0]]
        and aux.systems == [AUX_SYSTEM_PROMPTS[1]]
    )
    assert AUX_SYSTEM_PROMPTS[0] != AUX_SYSTEM_PROMPTS[1]


def _dimobench_like_sample():
    turns = []
    for i in range(1, 7):
        speaker = "user" if i % 2 else "assistant"
        turns.append(
            DialogueTurn(
                index=i,
                speaker=speaker,
                text=f"turn {i} about weekend plans",
                role=speaker,
                session=str((i - 1) // 2 + 1),
                dialogue_id=f"D{(i - 1) // 2 + 1}:{(i - 1) % 2 + 1}",
            )
        )
    qas = [
        QAPair(qid="late", question="late?", answer="x", category="current-state inference", meta={"anchor_turn": 6}),
        QAPair(qid="early", question="early?", answer="y", category="history recall", meta={"anchor_turn": 2}),
        QAPair(qid="end", question="end?", answer="z", category="history recall"),
    ]
    return BenchmarkSample(sample_id="p", dialogue=turns, qa_pairs=qas, meta={"source": "dimobench"})


def test_runner_asks_anchored_questions_mid_dialogue(offline_config, tmp_path):
    llm = ScriptedLLM()
    backends = Backends(llm=llm, embedder=HashingEmbedder(dim=64), verifier=HeuristicVerifier())
    settings = RunSettings(
        benchmark="dimobench",
        data_path="x",
        out_dir=str(tmp_path),
        config=offline_config,
        seeds=[13],
        workers=1,
        skip_judge=False,
        run_name="run",
    )
    judge = EnsembleJudge([LLMJudge(llm, "j")])
    runner = BenchmarkRunner(settings, backends=backends, judge=judge)
    summary = runner.run([_dimobench_like_sample()])

    preds = [json.loads(line) for line in (tmp_path / "run" / "seed_13" / "predictions.jsonl").read_text().splitlines()]
    by_qid = {p["qid"]: p for p in preds}
    # anchored questions are asked right after their dialogue turn; turns 1-2 form one exchange
    assert (by_qid["early"]["asked_after_dialogue_turn"], by_qid["early"]["memory_turn"]) == (2, 1)
    assert (by_qid["late"]["asked_after_dialogue_turn"], by_qid["late"]["memory_turn"]) == (6, 3)
    assert (by_qid["end"]["asked_after_dialogue_turn"], by_qid["end"]["memory_turn"]) == (6, 3)
    assert all(p["answer"] == "tea" and p["score"] == 1.0 for p in preds)
    assert summary["metrics"]["overall_mean"] == 100.0
    assert summary["metrics_by_seed"]["13"]["per_category"]["history recall"]["n"] == 2
    assert (tmp_path / "run" / "run_meta.json").exists()
    assert (tmp_path / "run" / "seed_13" / "traces" / "p.jsonl").exists()
    assert (tmp_path / "run" / "seed_13" / "memory" / "p.json").exists()
    assert preds[0]["qa_tokens"]["prompt_tokens"] > 0
    # QA prompts carry the question-conditioned context, never confidence weights.
    qa_prompts = [p for p in llm.prompts if "# OUTPUT FORMAT" in p]
    assert len(qa_prompts) == 3 and all("weight=" not in p for p in qa_prompts)
    assert all("Most recent turns:" in p for p in qa_prompts)  # the six-turn window is part of the response input


def test_runner_skip_judge_and_parallel_workers(offline_config, tmp_path):
    llm = ScriptedLLM()
    backends = Backends(llm=llm, embedder=HashingEmbedder(dim=64), verifier=HeuristicVerifier())
    settings = RunSettings(
        benchmark="dimobench",
        data_path="x",
        out_dir=str(tmp_path),
        config=offline_config,
        seeds=[1, 2],
        workers=3,
        skip_judge=True,
        run_name="r2",
        save_memory=False,
    )
    runner = BenchmarkRunner(settings, backends=backends)
    samples = [_dimobench_like_sample() for _ in range(4)]
    for i, s in enumerate(samples):
        s.sample_id = f"p{i}"
    summary = runner.run(samples)
    assert "metrics" not in summary and summary["metrics_by_seed"]["1"] == {"n": 12}
    assert not (tmp_path / "r2" / "seed_1" / "memory").exists()
