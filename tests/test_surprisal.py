import math

import numpy as np
import pytest

from precomem.config import load_config
from precomem.memory import BeliefStore, EpisodicCache, MemoryItem
from precomem.retrieval import HashingEmbedder, Retriever
from precomem.surprisal import (
    HeuristicVerifier,
    compute_surprisal,
    contradiction_index,
    effective_surprisal,
    raw_surprisal,
    retrieval_reliability,
    semantic_deviation,
)
from tests.conftest import FixedVerifier


def test_reliability_is_one_for_a_single_dominant_item_and_zero_when_uniform():
    assert retrieval_reliability([0.9]) == pytest.approx(1.0)  # log max(1,2) guard: one item is fully reliable
    assert retrieval_reliability([0.5, 0.5]) == pytest.approx(0.0)
    assert retrieval_reliability([0.9, 0.05, 0.05]) > 0.6
    assert retrieval_reliability([]) == 0.0
    # Scale invariance (up to epsilon): the entropy depends only on the normalized scores.
    assert retrieval_reliability([2, 1, 1]) == pytest.approx(retrieval_reliability([0.2, 0.1, 0.1]), abs=1e-6)


def test_raw_and_effective_surprisal_formulas():
    assert raw_surprisal(0.4, 0.6, lam=0.5) == pytest.approx(0.5)
    assert raw_surprisal(0.4, 0.6, lam=1.0) == pytest.approx(0.4)
    assert raw_surprisal(2.0, 2.0, lam=0.5) == 1.0  # clipped to [0, 1]
    assert effective_surprisal(0.5, 0.5) == pytest.approx(0.25)
    assert effective_surprisal(0.0, 1.0) == 0.0  # diffuse recall cannot trigger an update


def test_semantic_deviation_is_zero_for_identical_and_half_for_orthogonal_evidence():
    q = np.array([1.0, 0.0, 0.0])
    same = np.array([[1.0, 0.0, 0.0]])
    orth = np.array([[0.0, 1.0, 0.0]])
    assert semantic_deviation(q, same, [1.0]) == pytest.approx(0.0)
    assert semantic_deviation(q, orth, [1.0]) == pytest.approx(0.5)
    assert semantic_deviation(q, np.zeros((0, 3)), []) == 0.0


def test_heuristic_verifier_fires_on_negated_overlap():
    v = HeuristicVerifier()
    neutral = v.contradict("user drinks coffee every morning", "the weather is lovely today")
    contradiction = v.contradict("user drinks coffee every morning", "I don't drink coffee anymore")
    assert contradiction > neutral and contradiction > 0.5
    assert 0.0 <= neutral <= 1.0


def test_contradiction_index_resolves_label_order():
    assert contradiction_index({0: "contradiction", 1: "entailment", 2: "neutral"}) == 0
    assert contradiction_index({0: "ENTAILMENT", 1: "NEUTRAL", 2: "CONTRADICTION"}) == 2
    assert contradiction_index(None) == 2


def test_compute_surprisal_end_to_end_and_cold_start():
    cfg = load_config(overrides={"embedding": {"provider": "hashing"}}).retrieval
    retriever = Retriever(HashingEmbedder(dim=64), cfg)
    store = BeliefStore()
    store.add(MemoryItem(content="user drinks coffee every morning", weight=0.9))
    retrieval = retriever.retrieve("I quit coffee last week", store, EpisodicCache())
    sig = compute_surprisal("I quit coffee last week", retrieval, FixedVerifier(0.8), lam=0.5)
    assert sig.rho == pytest.approx(1.0)  # a single retrieved item is fully concentrated
    assert sig.d_conf == pytest.approx(0.8)
    assert sig.s_raw == pytest.approx(0.5 * sig.d_sem + 0.5 * 0.8)
    assert sig.s_eff == pytest.approx(sig.rho * sig.s_raw)
    assert sig.per_item_contradiction == (0.8,)
    assert math.isfinite(sig.d_sem) and 0.0 <= sig.d_sem <= 1.0

    cold = compute_surprisal(
        "hello", retriever.retrieve("hello", BeliefStore(), EpisodicCache()), FixedVerifier(1.0), lam=0.5
    )
    assert (cold.rho, cold.d_sem, cold.d_conf, cold.s_raw, cold.s_eff) == (0.0, 0.0, 0.0, 0.0, 0.0)
