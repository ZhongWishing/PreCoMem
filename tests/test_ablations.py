"""The ablations and controls of Fig. 4(a) and the appendix, as config switches."""

import pytest

from precomem.agent import PreCoMem
from precomem.config import ConfigError, load_config
from precomem.memory import MemoryType
from tests.conftest import FixedVerifier, ScriptedLLM, make_agent


def _cfg(**ablation):
    return load_config(
        overrides={"embedding": {"provider": "hashing"}, "nli": {"model": "heuristic"}, "ablation": ablation}
    )


def _seeded(agent: PreCoMem, llm: ScriptedLLM, text="I usually start my mornings with coffee."):
    """Cold-start one belief whose content equals ``text`` and point the scripted agents at it."""
    llm.profile_payload["hypothesis_content"] = text
    first = agent.step(text)
    item = agent.state.belief_store.get(first.operator["hypothesis"]["item_id"])
    llm.target_id = item.item_id
    return item


def test_es_forces_maintain_once_evidence_exists():
    llm = ScriptedLLM()
    agent = make_agent(_cfg(es=True), llm, verifier=FixedVerifier(1.0))  # strong contradiction would mean CORRECT
    _seeded(agent, llm)
    trace = agent.step("I really don't drink coffee anymore at all.")
    assert trace.regime == "MAINTAIN" and trace.gate.x_t == 0.0
    assert trace.surprisal.s_eff > 0.6  # the signal itself is still computed and logged


def test_rr_sets_rho_to_one():
    llm = ScriptedLLM()
    agent = make_agent(_cfg(rr=True), llm, verifier=FixedVerifier(0.0))
    _seeded(agent, llm, "coffee in the morning")
    agent.step("tea at night is my routine now")  # second belief -> two retrieved items with diffuse scores
    trace = agent.step("what about weekends")
    assert trace.retrieved and len(trace.retrieved) == 2
    assert trace.surprisal.rho == 1.0


def test_cw_overwrites_weights_and_promotes_immediately():
    llm = ScriptedLLM()
    agent = make_agent(_cfg(cw=True), llm)
    trace = agent.step("I usually start my mornings with coffee.")
    items = agent.state.belief_store.items()
    assert len(items) == 1 and items[0].weight == 1.0
    assert (
        items[0].mtype is MemoryType.FACT and trace.lifecycle["promoted"] == 1
    )  # promoted at its first lifecycle step
    assert agent.retriever.use_weights is False


def test_read_time_weighting_control_only_touches_retrieval():
    llm = ScriptedLLM()
    agent = make_agent(_cfg(read_time_weighting=False), llm)
    assert agent.retriever.use_weights is False
    _seeded(agent, llm)
    assert agent.state.belief_store.items()[0].weight < 0.8  # lifecycle confidence intact


def test_without_profile_commits_directly_as_fact():
    llm = ScriptedLLM()
    agent = make_agent(_cfg(without_profile=True), llm)
    trace = agent.step("I tried matcha at a new cafe today.")
    assert trace.regime == "PROFILE"
    item = agent.state.belief_store.items()[0]
    assert item.mtype is MemoryType.FACT and item.weight == 0.85 and trace.operator["initial_weight"] == 0.85


def test_without_correct_and_without_maintain_are_noops():
    llm = ScriptedLLM()
    agent = make_agent(_cfg(without_correct=True, without_maintain=True), llm, verifier=FixedVerifier(1.0))
    item = _seeded(agent, llm)
    w0 = item.weight
    trace = agent.step("I really don't drink coffee anymore at all.")
    assert (
        trace.regime == "CORRECT"
        and trace.operator["noop"] is True
        and "disabled by ablation" in trace.operator["reason"]
    )
    assert (
        item.weight == w0 and item.decay_protection
    )  # untouched: no CORRECT decay, and explicit events do not time-decay
    assert len(agent.state.belief_store.facts()) == 0
    assert len(agent.state.episodic_cache) == 2  # the observation still enters the episodic cache


def test_single_threshold_routes_medium_band_to_correct():
    llm = ScriptedLLM()
    agent = make_agent(_cfg(single_threshold=True), llm, verifier=FixedVerifier(0.5))
    _seeded(agent, llm)
    trace = agent.step("Switching to tea is becoming part of my routine.")
    assert 0.2 < trace.surprisal.s_eff <= 0.6, "test needs a medium-band turn"
    assert trace.regime == "CORRECT"


def test_random_routing_uses_only_the_given_frequencies():
    llm = ScriptedLLM()
    agent = make_agent(_cfg(random_routing=[0.0, 1.0, 0.0]), llm, verifier=FixedVerifier(1.0))
    _seeded(agent, llm)
    regimes = {agent.step(f"message {i} about nothing in particular").regime for i in range(5)}
    assert regimes == {"PROFILE"}


def test_invalid_ablation_combinations_are_rejected():
    with pytest.raises(ConfigError):
        _cfg(random_routing=[0.5, 0.5])
    with pytest.raises(ConfigError):
        _cfg(es=True, single_threshold=True)
    assert _cfg().ablation.any_active is False
    assert _cfg(es=True).ablation.active_switches() == {"es": True}
