import math

import pytest

from precomem.config import load_config
from precomem.memory import BeliefStore, EpisodicCache, HypothesisClass, MemoryItem, MemoryType
from precomem.operators import (
    CorrectOperator,
    CorrectOutcome,
    HypothesisLifecycle,
    MaintainOperator,
    MaintainOutcome,
    NoOpOutcome,
    ProfileOperator,
    ProfileOutcome,
    cumulative_decay,
    initial_weight,
    reinforce,
    soft_decay,
)
from precomem.retrieval import HashingEmbedder, Retriever
from tests.conftest import ScriptedLLM

CFG = load_config(overrides={"embedding": {"provider": "hashing"}})


def _store_with(content: str, weight: float, mtype=MemoryType.FACT):
    store = BeliefStore()
    item = MemoryItem(content=content, weight=weight, mtype=mtype, created_turn=1, last_supported_turn=1)
    store.add(item)
    retrieval = Retriever(HashingEmbedder(dim=64), CFG.retrieval).retrieve(content, store, EpisodicCache())
    return store, item, retrieval


# ----------------------------------------------------------------------
# MAINTAIN (Eq. 7)
# ----------------------------------------------------------------------


def test_maintain_applies_closed_form_and_marks_support():
    store, item, retrieval = _store_with("user loves matcha", 0.5)
    llm = ScriptedLLM()
    llm.target_id = item.item_id
    out = MaintainOperator(llm, CFG.operators.maintain)("matcha again", retrieval, x_t=0.1, store=store, turn=7)
    assert isinstance(out, MaintainOutcome)
    assert item.weight == pytest.approx(reinforce(0.5, 0.6)) == pytest.approx(0.8)
    assert item.last_supported_turn == 7 and out.supported_ids() == [item.item_id]
    assert out.updates[0].scalar == 0.6 and out.updates[0].old_weight == 0.5


def test_maintain_clips_eta_ignores_unknown_ids_and_attaches_refinement():
    store, item, retrieval = _store_with("user loves matcha", 0.5)
    assert item.embedding is not None  # embedded by the retrieval above
    llm = ScriptedLLM()
    llm.target_id = item.item_id
    llm.maintain_payload = {
        "reinforced_memories": [
            {"id": "TARGET", "evidence_strength": 7.0},  # clipped to eta_max = 1.0
            {"id": "not-in-store", "evidence_strength": 0.5},  # ignored
        ],
        "refinement_action": {
            "type": "attribute",
            "target_parent_id": "TARGET",
            "new_content": "drinks it unsweetened",
        },
    }
    out = MaintainOperator(llm, CFG.operators.maintain)("matcha", retrieval, 0.1, store, turn=3)
    assert len(out.updates) == 1 and out.updates[0].scalar == 1.0 and item.weight == pytest.approx(1.0)
    assert out.refinements[0].kind == "attribute"
    assert item.content.endswith("| attribute: drinks it unsweetened")
    assert item.embedding is None  # content changed -> re-embedded lazily


def test_maintain_without_evidence_is_a_noop():
    store = BeliefStore()
    retrieval = Retriever(HashingEmbedder(dim=64), CFG.retrieval).retrieve("x", store, EpisodicCache())
    out = MaintainOperator(ScriptedLLM(), CFG.operators.maintain)("x", retrieval, 0.1, store, turn=1)
    assert isinstance(out, NoOpOutcome)


# ----------------------------------------------------------------------
# PROFILE (Eq. 8)
# ----------------------------------------------------------------------


def test_profile_initialization_matches_appendix_example():
    prior, q, w0 = initial_weight(x_t=0.4, q_t=0.65, band=(0.60, 0.70))
    assert prior == pytest.approx(0.6599, abs=5e-5)
    assert w0 == pytest.approx(0.6549, abs=5e-5)
    # The agent scalar moves w_0 by at most half the band width.
    _, _, lo = initial_weight(0.4, 0.0, (0.60, 0.70))
    _, _, hi = initial_weight(0.4, 1.0, (0.60, 0.70))
    assert hi - lo == pytest.approx(0.05)


def test_profile_writes_hypothesis_with_band_class_and_keywords():
    store, _, retrieval = _store_with("user drinks coffee", 0.9)
    llm = ScriptedLLM()
    out = ProfileOperator(llm, CFG.operators.profile)("I tried matcha today", retrieval, x_t=0.4, store=store, turn=5)
    assert isinstance(out, ProfileOutcome)
    h = out.hypothesis
    assert h.mtype is MemoryType.HYPOTHESIS and h.classification is HypothesisClass.EXPLICIT
    assert h.weight == pytest.approx(0.6549, abs=5e-5) and h.decay_protection is True
    assert h.keywords == ["matcha", "cafe"] and h.created_turn == h.last_supported_turn == 5
    assert h.item_id in store and out.written_ids() == [h.item_id]


def test_profile_latent_defaults_when_agent_output_is_garbled():
    store, _, retrieval = _store_with("user drinks coffee", 0.9)
    llm = ScriptedLLM()
    llm.profile_payload = {"classification": "something odd", "hypothesis_content": "", "initial_confidence": "n/a"}
    out = ProfileOperator(llm, CFG.operators.profile)("feels tired lately", retrieval, x_t=0.3, store=store, turn=2)
    h = out.hypothesis
    assert h.classification is HypothesisClass.LATENT and h.decay_protection is False
    assert h.content == "feels tired lately"  # falls back to the utterance
    assert 0.50 <= h.weight <= 0.59  # inside the latent band


# ----------------------------------------------------------------------
# CORRECT (Eq. 10)
# ----------------------------------------------------------------------


def test_correct_decays_archives_and_writes_fresh_fact():
    store, item, retrieval = _store_with("user loves coffee", 0.85)
    llm = ScriptedLLM()
    llm.target_id = item.item_id
    out = CorrectOperator(llm, CFG.operators.correct)(
        "I don't drink coffee anymore", retrieval, x_t=0.7, store=store, turn=9
    )
    assert isinstance(out, CorrectOutcome)
    assert item.weight == pytest.approx(soft_decay(0.85, 1.0, 0.7, 0.7)) == pytest.approx(0.85 * math.exp(-0.49))
    assert item.deprecated is True and out.retired() == [item]
    assert item.last_supported_turn == 1  # a contradiction is not support
    assert out.new_fact is not None and out.new_fact.weight == 0.85 and out.new_fact.mtype is MemoryType.FACT
    assert out.new_fact.item_id in store and out.written_ids() == [out.new_fact.item_id]


def test_correct_only_touches_retrieved_items():
    store, item, retrieval = _store_with("user loves coffee", 0.85)
    other = store.add(MemoryItem(content="user has a cat", weight=0.9))
    llm = ScriptedLLM()
    llm.correct_payload = {"contradicted_memories": [{"id": other.item_id, "severity": 1.0}], "new_fact_content": ""}
    out = CorrectOperator(llm, CFG.operators.correct)("no more coffee", retrieval, 0.9, store, turn=2)
    assert other.weight == 0.9 and out.updates == [] and out.new_fact is None


# ----------------------------------------------------------------------
# Lifecycle (Eq. 9, promotion, pruning)
# ----------------------------------------------------------------------


def test_compounding_decay_matches_closed_form_and_eight_turn_horizon():
    lc = HypothesisLifecycle(CFG.lifecycle)
    store = BeliefStore()
    h = store.add(
        MemoryItem(content="guess", weight=0.545, mtype=MemoryType.HYPOTHESIS, created_turn=0, last_supported_turn=0)
    )
    pruned_at = None
    for t in range(1, 20):
        stats = lc.step(store, turn=t)
        if h.item_id not in store:
            pruned_at = t
            break
        assert h.weight == pytest.approx(0.545 * cumulative_decay(0.05, t))
    # Appendix: a latent hypothesis started mid-band is pruned after roughly eight unsupported turns.
    assert pruned_at == 8 and stats.pruned == 1


def test_supported_or_protected_hypotheses_do_not_decay_and_promotion_is_one_way():
    lc = HypothesisLifecycle(CFG.lifecycle)
    store = BeliefStore()
    protected = store.add(
        MemoryItem(content="p", weight=0.6, mtype=MemoryType.HYPOTHESIS, decay_protection=True, last_supported_turn=0)
    )
    supported = store.add(MemoryItem(content="s", weight=0.6, mtype=MemoryType.HYPOTHESIS, last_supported_turn=5))
    ripe = store.add(MemoryItem(content="r", weight=0.81, mtype=MemoryType.HYPOTHESIS, last_supported_turn=5))
    fact = store.add(MemoryItem(content="f", weight=0.05, mtype=MemoryType.FACT))
    stats = lc.step(store, turn=5)
    assert protected.weight == 0.6 and supported.weight == 0.6
    assert ripe.mtype is MemoryType.FACT and stats.promoted == 1 and ripe.metadata["promoted_at_turn"] == 5
    assert fact.item_id in store and stats.fact_pruned == 0  # facts are never pruned by default


def test_optional_fact_prune_for_ablation():
    cfg = load_config(overrides={"lifecycle": {"fact_optional_prune": 0.1}})
    store = BeliefStore()
    store.add(MemoryItem(content="old", weight=0.02, mtype=MemoryType.FACT))
    stats = HypothesisLifecycle(cfg.lifecycle).step(store, turn=1)
    assert stats.fact_pruned == 1 and len(store) == 0
