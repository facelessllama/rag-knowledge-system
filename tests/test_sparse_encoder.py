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


def test_build_sparse_vector_term_frequency():
    sv = build_sparse_vector("article article article clause")
    # 2 unique stemmed terms -> 2 sparse dimensions
    assert len(sv.indices) == 2
    values_by_index = dict(zip(sv.indices, sv.values))
    counts = sorted(values_by_index.values())
    assert counts == [1.0, 3.0]  # "clause" once, "article" three times


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
