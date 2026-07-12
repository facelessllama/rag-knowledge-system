"""
Tests for vector_db/sparse_encoder.py — the BM25-style sparse vector
construction used for Qdrant hybrid search.
"""
from collections import Counter

from vector_db.sparse_encoder import build_sparse_vector, tokenize


def test_tokenize_strips_stopwords_and_punctuation():
    tokens = tokenize("The Contract, Article 15.1 — clause 3.")
    assert "the" not in tokens
    assert "article" in tokens
    assert "contract" in tokens
    assert "15" in tokens


def test_tokenize_stems_english_only():
    tokens = tokenize("processing documents quickly")
    # "processing" -> "process" (strips "ing"), "documents" -> "document" (strips "s")
    assert "process" in tokens
    assert "document" in tokens


def test_tokenize_drops_single_char_and_stopwords():
    tokens = tokenize("a I to the of")
    assert tokens == []


def test_build_sparse_vector_empty_text_returns_empty_vector():
    sv = build_sparse_vector("")
    assert sv.indices == []
    assert sv.values == []


def test_build_sparse_vector_term_frequency_is_bm25_saturated():
    sv = build_sparse_vector("article article article clause")
    # 2 unique stemmed terms -> 2 sparse dimensions
    assert len(sv.indices) == 2
    values_by_index = dict(zip(sv.indices, sv.values))
    values = sorted(values_by_index.values())
    # "clause" (tf=1) -> (k1+1)*1/(k1+1) = 1.0 exactly; "article" (tf=3) -> < 3.0 (saturated)
    assert values[0] == 1.0
    assert 1.0 < values[1] < 3.0


def test_build_sparse_vector_caps_runaway_term_repetition():
    """A query that accidentally repeats a word many times (e.g. copy-paste
    duplication) must not get a term weight anywhere near linear in the
    repeat count — otherwise it can dominate RRF fusion and pull in unrelated
    documents that merely share that common word."""
    normal = build_sparse_vector("discount amount")
    repeated = build_sparse_vector(" ".join(["discount"] * 20) + " amount")

    normal_by_idx = dict(zip(normal.indices, normal.values))
    repeated_by_idx = dict(zip(repeated.indices, repeated.values))

    shared = set(normal_by_idx) & set(repeated_by_idx)
    assert len(shared) == 2  # "discount" and "amount" both present in both
    discount_weight_normal = max(normal_by_idx[i] for i in shared)
    discount_weight_repeated = max(repeated_by_idx[i] for i in shared)
    # Raw counts would be 1 vs 20 (20x). Saturated, the ratio must be far smaller.
    assert discount_weight_repeated / discount_weight_normal < 3.0


def test_build_sparse_vector_is_stable_across_calls():
    """Indices must be stable across process restarts — built on hashlib, not
    Python's randomized builtin hash()."""
    a = build_sparse_vector("Article 15.1 clause 3 about the Contract")
    b = build_sparse_vector("Article 15.1 clause 3 about the Contract")
    assert a.indices == b.indices
    assert a.values == b.values


def test_build_sparse_vector_no_index_collisions_for_distinct_tokens():
    tokens = tokenize("article clause contract statute penalty jurisdiction remedy breach")
    indices = [build_sparse_vector(t).indices[0] for t in tokens]
    assert len(set(indices)) == len(Counter(tokens))
