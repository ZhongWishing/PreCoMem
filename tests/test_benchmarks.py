import json

from precomem.benchmarks import (
    BENCHMARKS,
    DialogueTurn,
    DiMoBenchLoader,
    LoCoMoLoader,
    LongMemEvalLoader,
    PersonaMemLoader,
    get_benchmark,
    infer_role,
    pair_exchanges,
)


def _turn(i, speaker, text, role=None, **kw):
    return DialogueTurn(index=i, speaker=speaker, text=text, role=role or infer_role(speaker), **kw)


def test_pair_exchanges_user_assistant_and_consecutive_user_turns():
    turns = [
        _turn(1, "user", "hi"),
        _turn(2, "assistant", "hello"),
        _turn(3, "user", "a"),
        _turn(4, "user", "b"),
        _turn(5, "assistant", "ok"),
    ]
    ex = pair_exchanges(turns)
    assert [(e.user, e.assistant, e.turn_indices) for e in ex] == [
        ("hi", "hello", [1, 2]),
        ("a", "", [3]),
        ("b", "ok", [4, 5]),
    ]
    assert ex[0].memory_input() == "hi"  # role labels are not prefixed


def test_pair_exchanges_two_humans_and_assistant_first():
    turns = [
        _turn(1, "Caroline", "I went hiking", dialogue_id="D1:1", session="session_1", timestamp="8 May 2023"),
        _turn(2, "Melanie", "Nice!", dialogue_id="D1:2", session="session_1", timestamp="8 May 2023"),
    ]
    ex = pair_exchanges(turns)
    assert len(ex) == 2 and all(e.assistant == "" for e in ex)
    assert ex[0].memory_input() == "Caroline: I went hiking" and ex[0].dialogue_id == "D1:1"
    assert ex[1].speaker == "Melanie"
    first_assistant = pair_exchanges([_turn(1, "assistant", "Welcome back!"), _turn(2, "user", "thanks")])
    assert first_assistant[0].user == "" and first_assistant[0].assistant == "Welcome back!"
    assert first_assistant[1].user == "thanks"


def test_locomo_loader(tmp_path):
    data = [
        {
            "sample_id": "conv-1",
            "conversation": {
                "speaker_a": "Caroline",
                "speaker_b": "Melanie",
                "session_2": [{"speaker": "Melanie", "text": "second", "dia_id": "D2:1"}],
                "session_2_date_time": "9 May 2023",
                "session_1": [
                    {"speaker": "Caroline", "text": "first", "dia_id": "D1:1"},
                    {"speaker": "Melanie", "text": "reply", "dia_id": "D1:2"},
                ],
                "session_1_date_time": "8 May 2023",
            },
            "qa": [
                {"question": "q?", "answer": "a", "category": 2, "evidence": ["D1:1"]},
                {"question": "trick?", "answer": "n/a", "category": 5},
                {"question": "state?", "answer": "b", "category": 3},
            ],
        }
    ]
    path = tmp_path / "locomo.json"
    path.write_text(json.dumps(data))
    samples = LoCoMoLoader.load(path)
    s = samples[0]
    assert [t.dialogue_id for t in s.dialogue] == ["D1:1", "D1:2", "D2:1"]  # sessions sorted numerically
    assert all(t.role == "user" for t in s.dialogue) and s.dialogue[0].named_speaker == "Caroline"
    assert s.dialogue[2].timestamp == "9 May 2023"
    # official LoCoMo numbering: 2 = temporal, 3 = open-domain; adversarial (5) dropped by default
    assert [q.category for q in s.qa_pairs] == ["temporal", "open_domain"]
    assert s.qa_pairs[0].evidence == ["D1:1"]
    assert len(LoCoMoLoader.load(path, include_adversarial=True)[0].qa_pairs) == 3
    assert len(pair_exchanges(s.dialogue)) == 3  # every human message is an observation


def test_longmemeval_loader(tmp_path):
    data = [
        {
            "question_id": "q1_abs",
            "question": "what?",
            "answer": "I don't know",
            "question_type": "knowledge-update",
            "haystack_sessions": [
                [{"role": "user", "content": "u1", "has_answer": True}, {"role": "assistant", "content": "a1"}]
            ],
            "haystack_dates": ["2023/05/20 (Sat) 02:21"],
            "haystack_session_ids": ["s-1"],
            "question_date": "2023/06/01",
        }
    ]
    path = tmp_path / "lme.json"
    path.write_text(json.dumps(data))
    s = LongMemEvalLoader.load(path)[0]
    assert s.sample_id == "q1_abs" and s.qa_pairs[0].category == "KU" and s.qa_pairs[0].meta["abstention"] is True
    assert [t.role for t in s.dialogue] == ["user", "assistant"] and s.dialogue[0].session == "s-1"
    assert s.dialogue[0].meta["has_answer"] is True


def test_personamem_loader(tmp_path):
    data = {
        "data": [
            {
                "persona_id": "p1",
                "history": [
                    {"role": "user", "content": "I like jazz", "session": 1},
                    {"role": "assistant", "content": "cool", "session": 1},
                ],
                "questions": [
                    {
                        "question_id": "pq1",
                        "question": "music?",
                        "options": ["A. jazz", "B. rock"],
                        "answer": "A",
                        "category": "preference",
                    }
                ],
            }
        ]
    }
    path = tmp_path / "pm.json"
    path.write_text(json.dumps(data))
    s = PersonaMemLoader.load(path)[0]
    assert s.qa_pairs[0].options == ["A. jazz", "B. rock"] and s.qa_pairs[0].meta["format"] == "MCQ"
    assert s.dialogue[0].session == "1"


def test_dimobench_loader_new_and_legacy_formats(tmp_path):
    new = {
        "benchmark": "dimobench",
        "persona_id": "persona_a",
        "drift_turns": [200],
        "dialogue": [
            {
                "turn": 1,
                "session": 1,
                "turn_in_session": 1,
                "dialogue_id": "D1:1",
                "speaker": "user",
                "text": "hi",
                "phase": "pre",
            },
            {
                "turn": 2,
                "session": 1,
                "turn_in_session": 2,
                "dialogue_id": "D1:2",
                "speaker": "assistant",
                "text": "hello",
                "phase": "pre",
            },
        ],
        "qa_pairs": [
            {
                "qid": "Q0001",
                "question": "q",
                "answer": "a",
                "phase": "pre-change",
                "category": "history recall",
                "interference": "none",
                "evidence": ["D1:1"],
                "anchor_turn": 2,
                "window": 1,
                "audit": True,
            }
        ],
    }
    (tmp_path / "persona_a.json").write_text(json.dumps(new))
    (tmp_path / "suite.json").write_text(json.dumps({"personas": ["persona_a.json"]}))
    s = DiMoBenchLoader.load(tmp_path / "suite.json")[0]
    assert s.sample_id == "persona_a" and s.qa_pairs[0].meta["anchor_turn"] == 2 and s.qa_pairs[0].meta["audit"] is True
    assert s.meta["drift_turns"] == [200]

    legacy = {
        "persona_id": "old",
        "drift_turns": [200],
        "dialogue": [
            {"turn": 1, "speaker": "user", "text": "x", "session": 3, "phase": "pre"},
            {"turn": 2, "speaker": "assistant", "text": "y", "session": 3, "phase": "pre"},
        ],
        "qa_pairs": [{"qid": "Q1", "question": "q", "answer": "a", "phase": "pre-change", "category": "c"}],
    }
    (tmp_path / "old.json").write_text(json.dumps(legacy))
    s2 = DiMoBenchLoader.load(tmp_path / "old.json")[0]
    assert [t.dialogue_id for t in s2.dialogue] == ["D3:1", "D3:2"]
    assert s2.qa_pairs[0].meta["anchor_turn"] is None  # asked at the end of the dialogue


def test_registry():
    assert set(BENCHMARKS) == {"locomo", "longmemeval", "personamem", "dimobench"}
    assert get_benchmark("DiMoBench-XL").name == "dimobench"
    assert get_benchmark("longmemeval").default_backbone == "gpt-4.1"
