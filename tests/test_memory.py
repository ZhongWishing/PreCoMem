import numpy as np
import pytest

from precomem.memory import BeliefStore, EpisodicCache, EpisodicTrace, MemoryItem, MemoryState, MemoryType, TopicTable


def test_belief_store_version_tracks_structure_and_text_but_not_weights():
    store = BeliefStore()
    item = MemoryItem(content="user drinks coffee", weight=0.5)
    item.embedding = np.ones(4, dtype=np.float32)
    v0 = store.version
    store.add(item)
    assert store.version == v0 + 1
    item.set_weight(0.9, turn=3)
    assert store.version == v0 + 1  # weight edits do not invalidate the lexical index
    store.update_content(item.item_id, "user drinks tea", turn=4)
    assert store.version == v0 + 2
    assert item.embedding is None  # stale embedding dropped after a content edit
    assert item.last_updated_turn == 4
    store.remove(item.item_id)
    assert store.version == v0 + 3 and len(store) == 0


def test_item_weight_is_clamped_and_support_anchor_separate_from_updates():
    item = MemoryItem(content="x", weight=1.7, mtype=MemoryType.HYPOTHESIS, created_turn=2, last_supported_turn=2)
    assert item.weight == 1.0
    item.set_weight(0.3, turn=5)  # a CORRECT decay touches last_updated_turn only
    assert item.last_supported_turn == 2 and item.last_updated_turn == 5
    assert item.age_since_support(6) == 4
    item.mark_supported(6)
    assert item.age_since_support(6) == 0


def test_episodic_cache_evicts_oldest_and_resolves_in_order():
    cache = EpisodicCache(capacity=2)
    for t in range(1, 4):
        cache.append(EpisodicTrace(trace_id=f"E{t}", turn=t, user=f"u{t}", assistant=f"a{t}"))
    assert len(cache) == 2 and cache.get("E1") is None
    traces = cache.resolve(["E3", "E1", "E2", "E3"])
    assert [t.trace_id for t in traces] == ["E3", "E2"]
    with pytest.raises(ValueError):
        cache.append(EpisodicTrace(trace_id="E3", turn=9, user="dup"))


def test_episodic_trace_rendering_uses_identifiers_and_speaker():
    tr = EpisodicTrace(
        trace_id="E1",
        turn=1,
        user="hi",
        assistant="hello",
        speaker="Caroline",
        dialogue_id="D1:3",
        timestamp="8 May 2023",
    )
    assert tr.text.startswith("Caroline: hi")
    rendered = tr.render()
    assert "[D1:3] (8 May 2023) Caroline: hi" in rendered and "Assistant: hello" in rendered


def test_topic_table_leader_clustering_and_distribution():
    table = TopicTable(new_threshold=0.9)
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    assert table.assign(a, "coffee habits") == "coffee_habits"
    assert table.assign(b, "coffee habits") == "coffee_habits_2"  # dissimilar -> new topic, label de-duplicated
    assert table.assign(np.array([0.98, 0.02, 0.0]), "ignored") == "coffee_habits"  # joins the nearest prototype
    assert len(table) == 2
    dist = table.distribution(a)
    assert dist.shape == (2,) and dist.argmax() == 0 and abs(dist.sum() - 1.0) < 1e-9
    restored = TopicTable.from_dict(table.to_dict())
    assert restored.labels == table.labels and restored.index_of("coffee_habits_2") == 1


def test_memory_state_json_roundtrip(tmp_path):
    state = MemoryState(episodic_cache=EpisodicCache(capacity=50), topic_table=TopicTable(new_threshold=0.4))
    item = MemoryItem(
        content="user likes hiking", mtype=MemoryType.HYPOTHESIS, weight=0.62, keywords=["hiking"], topic="outdoors"
    )
    item.embedding = np.array([0.1, 0.2], dtype=np.float32)
    state.belief_store.add(item)
    state.episodic_cache.append(EpisodicTrace(trace_id="E1", turn=1, user="I hiked", assistant="nice", session="S1"))
    state.topic_table.assign(np.array([0.1, 0.2]), "outdoors")
    state.turn = 7
    path = state.save(tmp_path / "mem.json", include_embeddings=True)
    loaded = MemoryState.load(path)
    assert loaded.turn == 7 and len(loaded.belief_store) == 1 and len(loaded.episodic_cache) == 1
    restored = loaded.belief_store.get(item.item_id)
    assert restored.mtype is MemoryType.HYPOTHESIS and restored.keywords == ["hiking"] and restored.topic == "outdoors"
    assert np.allclose(restored.embedding, item.embedding)
    assert loaded.episodic_cache.capacity == 50 and loaded.topic_table.labels == ["outdoors"]
    assert "[HYPO]  user likes hiking  (w=0.62)" in loaded.render()
