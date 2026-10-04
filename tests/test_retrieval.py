import numpy as np

from precomem.config import load_config
from precomem.memory import BeliefStore, EpisodicCache, EpisodicTrace, MemoryItem, MemoryType, TopicTable
from precomem.retrieval import BM25Index, HashingEmbedder, Retriever, tokenize


def test_tokenizer_handles_latin_and_cjk():
    assert tokenize("User_1 loves Matcha-lattes") == ["user_1", "loves", "matcha", "lattes"]
    assert tokenize("我爱喝茶") == ["我", "爱", "喝", "茶"]


def test_bm25_ranks_matching_document_first_and_normalises_to_unit_max():
    index = BM25Index(["the user drinks coffee every morning", "the user plays football", "weather talk"])
    scores = index.normalized_score("coffee morning")
    assert scores.argmax() == 0 and abs(scores.max() - 1.0) < 1e-6
    assert scores[2] == 0.0
    assert np.all(index.score("zzz unknown") == 0.0)


def _retriever(top_k=10, topic=0.2):
    cfg = load_config(overrides={"retrieval": {"top_k": top_k, "alpha": {"topic": topic}}}).retrieval
    return Retriever(HashingEmbedder(dim=128), cfg)


def test_confidence_modulates_ranking():
    store = BeliefStore()
    high = MemoryItem(content="user loves coffee in the morning", weight=0.9)
    low = MemoryItem(content="user loves coffee in the morning", weight=0.3)  # identical text, lower confidence
    store.add(high)
    store.add(low)
    result = _retriever().retrieve("coffee morning", store, EpisodicCache())
    assert result.items[0] is high and result.scores[0] > result.scores[1]
    assert abs(result.scores[0] / result.scores[1] - 3.0) < 1e-6  # ratio of weights


def test_keywords_enter_only_the_lexical_term():
    store = BeliefStore()
    plain = MemoryItem(content="user mentioned a new hobby", weight=0.8)
    tagged = MemoryItem(content="user mentioned a new hobby", weight=0.8, keywords=["pottery", "kiln"])
    store.add(plain)
    store.add(tagged)
    result = _retriever().retrieve("pottery kiln", store, EpisodicCache())
    assert result.items[0] is tagged
    i_plain = result.items.index(plain)
    assert result.semantic[0] == result.semantic[i_plain]  # same embedding: keywords never enter e(c(m))
    assert result.lexical[0] > result.lexical[i_plain]


def test_topic_prior_and_provenance_traces():
    retriever = _retriever(top_k=2)
    store = BeliefStore()
    cache = EpisodicCache()
    table = TopicTable(new_threshold=0.0)
    a = MemoryItem(content="user runs every morning before work", weight=0.8, sources=["E1", "E2"])
    b = MemoryItem(content="user runs every morning before work", weight=0.8, sources=["E2", "E9"])
    for m in (a, b):
        store.add(m)
    emb = retriever.ensure_embeddings([a, b])
    a.topic = table.assign(emb[0], "running")
    cache.append(EpisodicTrace(trace_id="E1", turn=1, user="ran 5k", assistant="great"))
    cache.append(EpisodicTrace(trace_id="E2", turn=2, user="ran again", assistant="nice"))
    result = retriever.retrieve("morning run", store, cache, topic_table=table)
    assert result.n == 2 and result.items[0] is a
    assert result.topic[0] > 0.0 and result.topic[1] == 0.0  # only the labeled item gets the topic prior
    assert [t.trace_id for t in result.traces] == ["E1", "E2"]  # dedup, chronological, dangling E9 dropped


def test_lexical_index_is_rebuilt_only_when_store_version_changes():
    retriever = _retriever()
    store = BeliefStore()
    store.add(MemoryItem(content="alpha beta", weight=0.5))
    retriever.retrieve("alpha", store, EpisodicCache())
    first = retriever._bm25
    retriever.retrieve("alpha", store, EpisodicCache())
    assert retriever._bm25 is first
    store.add(MemoryItem(content="gamma", weight=0.5))
    retriever.retrieve("alpha", store, EpisodicCache())
    assert retriever._bm25 is not first


def test_empty_store_and_top_k_cap():
    retriever = _retriever(top_k=1)
    assert retriever.retrieve("anything", BeliefStore(), EpisodicCache()).is_empty
    store = BeliefStore()
    for i in range(3):
        store.add(MemoryItem(content=f"fact {i}", weight=0.5, mtype=MemoryType.FACT))
    assert retriever.retrieve("fact", store, EpisodicCache()).n == 1
