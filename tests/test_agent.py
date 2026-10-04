import json

import pytest

from precomem.agent import PreCoMem, TurnMeta
from precomem.gate import Regime
from tests.conftest import FixedVerifier, ScriptedLLM, make_agent


def test_cold_start_routes_first_observation_to_profile(agent: PreCoMem):
    trace = agent.step("I usually start my mornings with coffee.")
    assert trace.regime == "PROFILE" and trace.gate.cold_start is True
    assert trace.surprisal.s_eff == 0.0 and trace.retrieved == []
    hyps = agent.state.belief_store.hypotheses()
    assert len(hyps) == 1 and hyps[0].sources == [trace.trace_id]
    assert hyps[0].topic is not None and len(agent.state.topic_table) == 1
    assert hyps[0].embedding is not None
    assert trace.memory["n_hypotheses"] == 1 and trace.tokens["prompt_tokens"] > 0
    assert set(trace.latency_ms) >= {"retrieve", "respond", "surprisal", "operator", "lifecycle"}


def test_maintain_path_reinforces_and_resets_decay_anchor(offline_config, scripted_llm: ScriptedLLM):
    agent = make_agent(offline_config, scripted_llm, verifier=FixedVerifier(0.0))
    scripted_llm.profile_payload["hypothesis_content"] = "I usually start my mornings with coffee."
    first = agent.step("I usually start my mornings with coffee.")
    item = agent.state.belief_store.get(first.operator["hypothesis"]["item_id"])
    scripted_llm.target_id = item.item_id
    w_before = item.weight
    # Identical text -> d_sem = 0; no contradiction -> S_eff = 0 -> MAINTAIN.
    second = agent.step("I usually start my mornings with coffee.")
    assert second.regime == "MAINTAIN" and second.surprisal.s_eff == pytest.approx(0.0)
    assert item.weight == pytest.approx(w_before + 0.6 * (1 - w_before))
    assert item.last_supported_turn == 2 and second.trace_id in item.sources
    assert second.operator["updates"][0]["id"] == item.item_id


def test_correct_path_with_reliable_contradiction(offline_config, scripted_llm: ScriptedLLM):
    agent = make_agent(offline_config, scripted_llm, verifier=FixedVerifier(1.0))
    first = agent.step("I usually start my mornings with coffee.")
    item = agent.state.belief_store.get(first.operator["hypothesis"]["item_id"])
    scripted_llm.target_id = item.item_id
    trace = agent.step("I really don't drink coffee anymore at all.")
    # One retrieved item -> rho = 1; d_conf = 1 -> S_raw >= 0.5 -> above theta_high once d_sem contributes.
    assert trace.surprisal.rho == pytest.approx(1.0) and trace.surprisal.d_conf == 1.0
    assert trace.regime == "CORRECT"
    assert item.deprecated is True and item.weight < first.operator["initial_weight"]
    facts = agent.state.belief_store.facts()
    assert len(facts) == 1 and facts[0].content == "User prefers tea over coffee now" and facts[0].weight == 0.85
    assert facts[0].sources == [trace.trace_id] and facts[0].topic is not None


def test_read_only_turn_does_not_mutate_memory(agent: PreCoMem):
    agent.step("I usually start my mornings with coffee.")

    def memory_only():
        snap = agent.snapshot()
        snap.pop("turn")  # the dialogue clock still advances on a read-only turn
        return json.dumps(snap, sort_keys=True)

    before = memory_only()
    trace = agent.step("What should I drink?", update_memory=False)
    assert trace.regime == "NONE" and trace.operator == {"noop": True, "reason": "read-only turn"}
    assert memory_only() == before and trace.trace_id == ""
    assert len(agent.short_term) == 2  # still a dialogue turn for the working window


def test_observe_stores_benchmark_metadata_and_recall_hides_weights(agent: PreCoMem):
    meta = TurnMeta(speaker="Caroline", session="session_1", dialogue_id="D1:3", timestamp="8 May 2023")
    trace = agent.observe("Caroline: I went hiking with Mel last weekend.", "", meta=meta)
    stored = agent.state.episodic_cache.get(trace.trace_id)
    assert stored.speaker == "Caroline" and stored.dialogue_id == "D1:3" and stored.timestamp == "8 May 2023"
    evidence = agent.recall("Who went hiking?")
    rendered = evidence.render()
    assert "weight" not in rendered and "w=" not in rendered
    assert "[D1:3] (8 May 2023) Caroline:" in rendered
    assert len(evidence.beliefs) == 1


def test_blank_input_writes_episode_without_belief_update(agent: PreCoMem):
    trace = agent.observe("", "The assistant recommends a 5k run plan.")
    assert trace.regime == "NONE" and len(agent.state.belief_store) == 0
    assert len(agent.state.episodic_cache) == 1


def test_reset_save_and_load(agent: PreCoMem, tmp_path):
    agent.step("I usually start my mornings with coffee.")
    path = agent.save_memory(tmp_path / "mem.json")
    agent.reset()
    assert len(agent.state.belief_store) == 0 and agent.state.turn == 0 and not agent.history
    agent.load_memory(path)
    assert len(agent.state.belief_store) == 1 and agent.state.turn == 1
    agent.dump_history(tmp_path / "hist.jsonl")
    assert (tmp_path / "hist.jsonl").read_text() == ""  # history was cleared by reset


def test_dump_history_json_lines(agent: PreCoMem, tmp_path):
    agent.step("I tried matcha at a new cafe today.")
    agent.step("Matcha lattes are my ritual now.")
    path = agent.dump_history(tmp_path / "h.jsonl")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["turn"] for r in rows] == [1, 2]
    assert rows[1]["regime"] in {r.value for r in Regime}
    assert {"gate", "surprisal", "retrieved", "operator", "lifecycle", "memory"} <= set(rows[1])


def test_query_uses_generic_response_prompt(agent: PreCoMem, scripted_llm: ScriptedLLM):
    agent.step("I usually start my mornings with coffee.")
    answer = agent.query("What do I drink?")
    assert answer == scripted_llm.response_text
    assert "Retrieved beliefs:" in scripted_llm.prompts[-1]
    assert agent.state.turn == 1  # a query never advances the turn counter
