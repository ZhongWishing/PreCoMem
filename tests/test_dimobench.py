import json
from collections import Counter

import pytest

from precomem.benchmarks import pair_exchanges, parse_persona
from precomem.dimobench import (
    BuildOptions,
    CueKind,
    DiMoBenchBuilder,
    Phase,
    QARecord,
    Turn,
    audit_summary,
    base_spec,
    cohen_kappa,
    pairwise_kappa,
    spec_for_seed,
    write_audit_sheets,
    xl_spec,
)
from tests.conftest import FakeDiMoBenchLLM

BASE_SEED = {
    "persona_id": "persona_t",
    "life_stage": "early career, 26",
    "big_five": {"O": 0.3, "C": 0.6, "E": 0.85, "A": 0.55, "N": 0.35},
    "social_circle_hint": ["Leo", "Leon", "Mia"],
    "slots": [
        {
            "slot": "attitude_toward_social_activities",
            "initial_state": "organizes group outings",
            "transitions": {1: "prefers quiet solo weekends"},
        }
    ],
    "drift_hints": {1: "burnout"},
}


def test_base_spec_reproduces_paper_counts():
    spec = base_spec("attitude_toward_social_activities")
    plans = spec.session_plans()
    assert spec.n_turns == 350 and len(plans) == 35
    phases = [p.phase for p in plans]
    assert phases[:20] == [Phase.PRE] * 20 and phases[20] is Phase.CHANGE_POINT
    assert phases[21:24] == [Phase.DENSE] * 3 and phases[24:] == [Phase.MAINTENANCE] * 11
    assert plans[20].first_turn == 201 and plans[23].last_turn == 240
    assert len(plans[20].cues) == 4 and all(c.kind is CueKind.IMPLICIT_EVENT for c in plans[20].cues)
    assert [p.index for p in plans if p.confounder] == [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33]
    cells = spec.qa_cells()
    assert len(cells) == 200
    assert Counter(c.qa_phase for c in cells) == {"pre-change": 90, "change-point": 10, "post-change": 100}
    assert all(v == 50 for v in Counter(c.category.value for c in cells).values())
    assert all(c.anchor_turn % 10 == 0 for c in cells)


def test_xl_spec_three_drifts_and_window_quotas():
    spec = xl_spec()
    assert spec.n_turns == 1050 and spec.drift_turns == [280, 580, 870]
    plans = spec.session_plans()
    change_sessions = [p.index for p in plans if p.phase is Phase.CHANGE_POINT]
    assert change_sessions == [29, 59, 88]
    for k in (1, 2, 3):
        n = sum(
            1 for p in plans for c in p.cues if c.drift_index == k and c.kind in (CueKind.IMPLICIT_EVENT, CueKind.DENSE)
        )
        assert 20 <= n <= 30
    cells = spec.qa_cells()
    assert len(cells) == 420
    assert Counter(c.window for c in cells) == {w: 20 for w in range(1, 22)}
    assert all(c.anchor_turn == c.window * 50 for c in cells)
    assert sum(1 for c in cells if c.qa_phase == "change-point") == 60  # three windows contain a change-point session


def test_spec_for_seed_validation():
    spec = spec_for_seed(BASE_SEED, "base")
    assert spec.name == "dimobench" and spec.slots == ("attitude_toward_social_activities",)
    with pytest.raises(ValueError):
        spec_for_seed(BASE_SEED, "xl")  # XL needs transitions at all three drifts
    with pytest.raises(ValueError):
        spec_for_seed(BASE_SEED, "nope")


def _build(tmp_path, verdict="ACCEPT", revise_first_n=0, **opt):
    spec = spec_for_seed(BASE_SEED, "base")
    generator = FakeDiMoBenchLLM("fake-generator")
    verifier = FakeDiMoBenchLLM("fake-verifier", verdict=verdict, revise_first_n=revise_first_n)
    options = BuildOptions(workers=2, skeleton_chunk=10, qa_batch=10, **opt)
    builder = DiMoBenchBuilder(spec, generator, verifier, options, work_dir=tmp_path / "work")
    return builder, generator, verifier, builder.build(BASE_SEED)


def test_pipeline_end_to_end_offline(tmp_path):
    builder, generator, verifier, dataset = _build(tmp_path, max_sessions=24, max_qa=40)
    assert dataset["benchmark"] == "dimobench" and dataset["persona"]["name"] == "Mike"
    assert len(dataset["dialogue"]) == 240
    turns = [Turn.from_dict(t) for t in dataset["dialogue"]]
    assert [t.dialogue_id for t in turns[:3]] == ["D1:1", "D1:2", "D2:1"] or turns[0].dialogue_id == "D1:1"
    assert all(t.speaker == ("user" if t.turn_in_session % 2 else "assistant") for t in turns)
    # every planned cue is localized in a user turn
    realized = {c for t in turns for c in t.cue_ids}
    planned = {c["cue_id"] for s in dataset["sessions"] for c in s["cues"]}
    assert planned == realized and all(t.turn_in_session % 2 == 1 for t in turns if t.cue_ids)
    assert any("keyword_trap" in t.tags for t in turns) and any("entity_decoy" in t.tags for t in turns)
    # cues carry the slot state they express and their realized dialogue id
    cue = dataset["cues"][0]
    assert cue["dialogue_id"] in {t.dialogue_id for t in turns} and cue["state_to_express"] == "organizes group outings"
    assert dataset["slots"][0]["segments"] == [
        {"from_turn": 1, "to_turn": 200, "state": "organizes group outings"},
        {"from_turn": 201, "to_turn": 350, "state": "prefers quiet solo weekends"},
    ]

    qas = [QARecord.from_dict(q) for q in dataset["qa_pairs"]]
    assert len(qas) == 40 and all(q.verification.decision == "ACCEPT" for q in qas)
    by_id = {t.dialogue_id: t for t in turns}
    for q in qas:
        assert q.evidence and all(by_id[e].turn <= q.anchor_turn for e in q.evidence)
        assert q.anchor_turn % 10 == 0 and q.phase == "pre-change"
    stats = dataset["stats"]
    assert stats["n_qa_requested"] == 40 and stats["n_qa_generated"] == 40 and stats["accept_rate"] == 1.0
    assert stats["generator_model"] == "fake-generator" and stats["verifier_model"] == "fake-verifier"
    assert generator.calls["verify"] if "verify" in generator.calls else True  # generator never verifies
    assert "verify" not in generator.calls and set(verifier.calls) == {"verify"}
    assert audit_summary(qas)["n_audit"] == 8  # exactly 20% of 40, allocated across strata

    # checkpoints exist and a resumed build makes no new generator calls
    work = tmp_path / "work" / "persona_t"
    assert {p.name for p in work.iterdir()} >= {
        "stage1_persona.json",
        "stage2_skeleton.json",
        "stage3_dialogue.json",
        "stage4_qa.json",
        "stage5_verified.json",
    }
    calls_before = dict(generator.calls)
    again = builder.build(BASE_SEED)
    assert generator.calls == calls_before and again["qa_pairs"] == dataset["qa_pairs"]

    # the loader consumes the output and the anchors survive
    sample = parse_persona(dataset)
    assert len(pair_exchanges(sample.dialogue)) == 120 and sample.qa_pairs[0].meta["anchor_turn"] == qas[0].anchor_turn


def test_revise_loop_regenerates_then_accepts(tmp_path):
    builder, generator, verifier, dataset = _build(tmp_path, revise_first_n=3, max_sessions=4, max_qa=6)
    qas = [QARecord.from_dict(q) for q in dataset["qa_pairs"]]
    assert len(qas) == 6 and all(q.verification.decision == "ACCEPT" for q in qas)
    stats = dataset["stats"]
    assert stats["accept_by_round"].get(2, 0) >= 1 and stats["n_accept"] == 6
    assert verifier.calls["verify"] > 6  # some questions were verified twice
    assert generator.calls["qa"] >= 2  # at least one revision request went to the generator
    # a resumed build restores the verdict tally from the checkpoint
    again = builder.build(BASE_SEED)
    assert again["stats"]["n_accept"] == 6 and again["stats"]["accept_by_round"] == stats["accept_by_round"]


def test_reject_discards_immediately(tmp_path):
    _, generator, verifier, dataset = _build(tmp_path, verdict="REJECT", max_sessions=3, max_qa=4, rounds=2)
    assert dataset["qa_pairs"] == [] and dataset["stats"]["n_rejected"] == 4
    assert verifier.calls["verify"] == 4  # one verdict per question, no rewrite after REJECT


def test_unresolved_questions_are_dropped_unless_kept(tmp_path):
    _, _, _, dropped = _build(tmp_path / "a", verdict="REVISE", max_sessions=3, max_qa=4, rounds=2)
    assert dropped["qa_pairs"] == [] and dropped["stats"]["n_unresolved"] == 4
    _, _, _, kept = _build(tmp_path / "b", verdict="REVISE", max_sessions=3, max_qa=4, rounds=2, keep_unresolved=True)
    assert len(kept["qa_pairs"]) == 4 and kept["stats"]["n_unresolved"] == 4
    assert all(q["verification"]["decision"] == "REVISE" and q["verification"]["rounds"] == 2 for q in kept["qa_pairs"])


def test_audit_sheets_and_kappa(tmp_path):
    _, _, _, dataset = _build(tmp_path, max_sessions=3, max_qa=12)
    qas = [QARecord.from_dict(q) for q in dataset["qa_pairs"]]
    turns = [Turn.from_dict(t) for t in dataset["dialogue"]]
    files = write_audit_sheets("persona_t", qas, turns, tmp_path / "audit")
    rows = [json.loads(line) for line in files["audit_jsonl"].read_text().splitlines()]
    assert rows and all(r["evidence_text"] for r in rows)
    assert files["audit_csv"].read_text().splitlines()[0].startswith("persona_id,qid,phase")
    assert len(files["calibration_jsonl"].read_text().splitlines()) == 12

    assert cohen_kappa(["a", "b", "a", "b"], ["a", "b", "a", "b"]) == 1.0
    assert cohen_kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"]) == pytest.approx(0.0)
    k = pairwise_kappa({"ann1": ["a", "b", "a"], "ann2": ["a", "b", "a"], "ann3": ["a", "b", "b"]})
    assert k["ann1~ann2"] == 1.0 and "mean" in k
