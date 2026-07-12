"""
Tests for rag/retriever.py — multi-query merge/boost logic, per-document
result guarantee, and neighbor expansion. Embedding and Qdrant are faked
(no real GPU/network calls) so these run fast and deterministically.
"""
import pytest

from rag.retriever import HybridRetriever, extract_case_numbers, promote_case_number_matches


class FakeEmbedder:
    def embed_text(self, text):
        return [0.1, 0.2, 0.3]


class FakeVectorStore:
    def __init__(self, results_by_query=None, neighbors_by_doc=None, full_docs_by_id=None):
        self.results_by_query = results_by_query or {}
        self.neighbors_by_doc = neighbors_by_doc or {}
        self.full_docs_by_id = full_docs_by_id or {}  # doc_id -> list[chunk] | None (None = "too long")

    def hybrid_search(self, query_vector, query_text, top_k=5, doc_filter=None, folder_filter=None):
        return [dict(r) for r in self.results_by_query.get(query_text, [])]

    def neighbor_chunks(self, document_id, chunk_indices):
        candidates = self.neighbors_by_doc.get(document_id, [])
        return [dict(n) for n in candidates if n["chunk_index"] in chunk_indices]

    def all_chunks_for_document(self, document_id, limit):
        if document_id not in self.full_docs_by_id:
            return None
        chunks = self.full_docs_by_id[document_id]
        if chunks is None or len(chunks) > limit:
            return None
        return [dict(c) for c in chunks]


def _chunk(chunk_id, doc_id, idx, score, text="text"):
    return {
        "chunk_id": chunk_id, "document_id": doc_id, "chunk_index": idx,
        "score": score, "text": text, "page_num": 1, "filename": "f.pdf", "folder": "",
    }


@pytest.mark.asyncio
async def test_retrieve_expanded_boosts_chunks_found_by_multiple_queries():
    chunk1 = _chunk("c1", "d1", 0, 0.5)
    chunk2 = _chunk("c2", "d1", 1, 0.3)
    vs = FakeVectorStore(results_by_query={
        "q1": [chunk1, chunk2],
        "q2": [_chunk("c1", "d1", 0, 0.4)],  # same chunk_id found again
    })
    retriever = HybridRetriever(FakeEmbedder(), vs, top_k=5)

    results = await retriever.retrieve_expanded(["q1", "q2"], top_k=5)

    by_id = {r["chunk_id"]: r for r in results}
    assert by_id["c1"]["score"] == pytest.approx(0.5 + 0.4 * 0.9 * 0.25)
    assert by_id["c1"]["source"] == "hybrid"
    assert by_id["c2"]["score"] == pytest.approx(0.3)


@pytest.mark.asyncio
async def test_retrieve_expanded_guarantees_one_chunk_per_document():
    chunk_doc1 = _chunk("c1", "d1", 0, 0.9)
    chunk_doc2 = _chunk("c2", "d2", 0, 0.1)  # low score, would be cut by top_k=1
    vs = FakeVectorStore(results_by_query={"q1": [chunk_doc1, chunk_doc2]})
    retriever = HybridRetriever(FakeEmbedder(), vs, top_k=5)

    results = await retriever.retrieve_expanded(["q1"], top_k=1)

    doc_ids = {r["document_id"] for r in results}
    assert doc_ids == {"d1", "d2"}  # d2 pulled back in despite top_k=1


@pytest.mark.asyncio
async def test_retrieve_expands_top_candidates_with_neighbors():
    anchor = _chunk("anchor", "d1", 5, 0.8)
    vs = FakeVectorStore(
        results_by_query={"the query": [anchor]},
        neighbors_by_doc={"d1": [
            {"chunk_id": "n4", "document_id": "d1", "chunk_index": 4, "text": "neighbor",
             "page_num": 1, "filename": "f.pdf", "folder": ""},
        ]},
    )
    retriever = HybridRetriever(FakeEmbedder(), vs, top_k=5)

    results = await retriever.retrieve("the query", top_k=5)

    by_id = {r["chunk_id"]: r for r in results}
    assert "n4" in by_id
    assert by_id["n4"]["source"] == "neighbor"
    assert by_id["n4"]["score"] == pytest.approx(0.8 * 0.6)
    assert by_id["anchor"]["source"] == "hybrid"


@pytest.mark.asyncio
async def test_retrieve_pulls_in_whole_short_document_not_just_index_window():
    """A generic query might only match a short document's caption chunk
    (index 0) and miss its substantive chunk (index 3) — for documents at or
    under SHORT_DOC_CHUNK_LIMIT, the whole document should be pulled in
    rather than just a +/-1 index window around the hit."""
    anchor = _chunk("caption", "d1", 0, 0.8)
    substantive = {"chunk_id": "ruling", "document_id": "d1", "chunk_index": 3,
                    "text": "the actual ruling", "page_num": 2, "filename": "f.pdf", "folder": ""}
    vs = FakeVectorStore(
        results_by_query={"q": [anchor]},
        full_docs_by_id={"d1": [
            {"chunk_id": "caption", "document_id": "d1", "chunk_index": 0, "text": "caption",
             "page_num": 1, "filename": "f.pdf", "folder": ""},
            substantive,
        ]},
    )
    retriever = HybridRetriever(FakeEmbedder(), vs, top_k=5)

    results = await retriever.retrieve("q", top_k=5)

    by_id = {r["chunk_id"]: r for r in results}
    assert "ruling" in by_id
    assert by_id["ruling"]["source"] == "neighbor"
    assert by_id["ruling"]["score"] == pytest.approx(0.8 * 0.6)


@pytest.mark.asyncio
async def test_retrieve_falls_back_to_index_window_for_long_documents():
    """all_chunks_for_document returning None (too many chunks) must fall
    back to the existing +/-1 neighbor-window behavior, not silently drop
    expansion entirely."""
    anchor = _chunk("anchor", "d1", 5, 0.8)
    vs = FakeVectorStore(
        results_by_query={"the query": [anchor]},
        neighbors_by_doc={"d1": [
            {"chunk_id": "n4", "document_id": "d1", "chunk_index": 4, "text": "neighbor",
             "page_num": 1, "filename": "f.pdf", "folder": ""},
        ]},
        full_docs_by_id={"d1": None},  # too long to fully expand
    )
    retriever = HybridRetriever(FakeEmbedder(), vs, top_k=5)

    results = await retriever.retrieve("the query", top_k=5)

    by_id = {r["chunk_id"]: r for r in results}
    assert "n4" in by_id
    assert by_id["n4"]["source"] == "neighbor"


@pytest.mark.asyncio
async def test_retrieve_does_not_duplicate_neighbor_already_in_results():
    chunk_a = _chunk("a", "d1", 4, 0.8)
    chunk_b = _chunk("b", "d1", 5, 0.7)  # already adjacent — should not be re-added as a "neighbor"
    vs = FakeVectorStore(
        results_by_query={"q": [chunk_a, chunk_b]},
        neighbors_by_doc={"d1": [{"chunk_id": "b", "document_id": "d1", "chunk_index": 5,
                                   "text": "text", "page_num": 1, "filename": "f.pdf", "folder": ""}]},
    )
    retriever = HybridRetriever(FakeEmbedder(), vs, top_k=5)

    results = await retriever.retrieve("q", top_k=5)

    assert len([r for r in results if r["chunk_id"] == "b"]) == 1
    assert next(r for r in results if r["chunk_id"] == "b")["source"] == "hybrid"


# ── extract_case_numbers ─────────────────────────────────────────────────────

def test_extract_case_numbers_matches_various_type_prefixes():
    assert extract_case_numbers("C.R.M. 10691 of 2020") == {("10691", "2020")}
    assert extract_case_numbers("WPA 12091 of 2019") == {("12091", "2019")}
    assert extract_case_numbers("F.A.O. No.293 of 2019") == {("293", "2019")}
    assert extract_case_numbers("ABLAPL 1380 of 2020") == {("1380", "2020")}


def test_extract_case_numbers_returns_empty_for_no_match():
    assert extract_case_numbers("This document has no case number in it.") == set()


def test_extract_case_numbers_finds_multiple():
    text = "Related to CRM 100 of 2019 and also WPA 200 of 2020."
    assert extract_case_numbers(text) == {("100", "2019"), ("200", "2020")}


# ── case-number exact-match guarantee ────────────────────────────────────────

@pytest.mark.asyncio
async def test_retrieve_expanded_guarantees_case_number_match_even_with_low_score():
    """These short, heavily-templated interlocutory orders (case caption +
    1-2 sentence order, near-identical boilerplate across different cases)
    are exactly where semantic/BM25 scoring is least reliable — an exact
    case-number match in the query must survive even if its score alone
    wouldn't make top_k."""
    matching_low_score = _chunk("match", "d_target", 0, 0.01,
                                 text="C.R.M. 10691 of 2020 In the matter of Mohan Viswakarma")
    distractor_high_score = _chunk("distractor", "d_other", 0, 0.9,
                                    text="CRR 1374 of 2020 Mainak Ranjan Bakshi")
    vs = FakeVectorStore(results_by_query={
        "What is the case CRM 10691 of 2020 about?": [distractor_high_score, matching_low_score],
    })
    retriever = HybridRetriever(FakeEmbedder(), vs, top_k=1)  # top_k=1 would normally cut the low scorer

    results = await retriever.retrieve_expanded(["What is the case CRM 10691 of 2020 about?"], top_k=1)

    chunk_ids = {r["chunk_id"] for r in results}
    assert "match" in chunk_ids
    matched = next(r for r in results if r["chunk_id"] == "match")
    assert matched["case_number_match"] is True


@pytest.mark.asyncio
async def test_retrieve_expanded_no_case_number_in_query_is_a_no_op():
    chunk1 = _chunk("c1", "d1", 0, 0.5, text="some ordinary text")
    vs = FakeVectorStore(results_by_query={"a generic question": [chunk1]})
    retriever = HybridRetriever(FakeEmbedder(), vs, top_k=5)

    results = await retriever.retrieve_expanded(["a generic question"], top_k=5)

    assert "case_number_match" not in results[0]


# ── promote_case_number_matches ──────────────────────────────────────────────

def _rr_chunk(chunk_id, rerank_score=None, case_number_match=False):
    c = {"chunk_id": chunk_id, "text": "text"}
    if rerank_score is not None:
        c["rerank_score"] = rerank_score
    if case_number_match:
        c["case_number_match"] = True
    return c


def test_promote_case_number_matches_is_a_no_op_without_a_match():
    chunks = [_rr_chunk("a")]
    top_chunks = [_rr_chunk("a", rerank_score=0.5)]
    assert promote_case_number_matches(chunks, top_chunks, relevance_threshold=1.5) == top_chunks


def test_promote_case_number_matches_adds_missing_match_above_threshold():
    matched = _rr_chunk("match", case_number_match=True)  # not yet reranked, no score
    chunks = [matched, _rr_chunk("other", case_number_match=False)]
    top_chunks = [_rr_chunk("other", rerank_score=3.0)]  # reranker's own top pick, no match

    result = promote_case_number_matches(chunks, top_chunks, relevance_threshold=1.5)

    ids = {c["chunk_id"] for c in result}
    assert "match" in ids
    promoted = next(c for c in result if c["chunk_id"] == "match")
    assert promoted["rerank_score"] > 1.5


def test_promote_case_number_matches_boosts_already_included_low_score():
    """If the reranker DID include the match but scored it below threshold,
    raise its score rather than add a duplicate."""
    matched = _rr_chunk("match", case_number_match=True)
    chunks = [matched]
    top_chunks = [_rr_chunk("match", rerank_score=0.1)]  # reranker scored it too low

    result = promote_case_number_matches(chunks, top_chunks, relevance_threshold=1.5)

    assert len(result) == 1
    assert result[0]["rerank_score"] > 1.5


def test_promote_case_number_matches_does_not_drop_the_match_even_over_top_k():
    matched = _rr_chunk("match", case_number_match=True)
    chunks = [matched]
    top_chunks = [_rr_chunk("a", rerank_score=5.0), _rr_chunk("b", rerank_score=4.0)]

    result = promote_case_number_matches(chunks, top_chunks, relevance_threshold=1.5)

    ids = {c["chunk_id"] for c in result}
    assert "match" in ids
    assert len(result) == 3  # exceeds top_k=2 rather than drop the match
