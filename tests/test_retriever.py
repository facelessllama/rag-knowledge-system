"""
Tests for rag/retriever.py — multi-query merge/boost logic, per-document
result guarantee, and neighbor expansion. Embedding and Qdrant are faked
(no real GPU/network calls) so these run fast and deterministically.
"""
import pytest

from rag.retriever import (
    HybridRetriever, extract_case_numbers, extract_celex_ids, extract_citation_numbers,
    extract_party_match, promote_identity_matches, promote_document_opening_chunks,
    extract_structural_references, structural_match_tier, best_structural_chunk,
)


class FakeEmbedder:
    def embed_text(self, text):
        return [0.1, 0.2, 0.3]


class FakeVectorStore:
    def __init__(self, results_by_query=None, neighbors_by_doc=None, full_docs_by_id=None,
                 case_number_index=None, celex_id_index=None, citation_number_index=None,
                 document_chunks=None):
        self.results_by_query = results_by_query or {}
        self.neighbors_by_doc = neighbors_by_doc or {}
        self.full_docs_by_id = full_docs_by_id or {}  # doc_id -> list[chunk] | None (None = "too long")
        self.case_number_index = case_number_index or {}  # (num, year) -> list[chunk]
        self.celex_id_index = celex_id_index or {}  # celex_id -> list[chunk]
        self.citation_number_index = citation_number_index or {}  # (num, year) -> list[chunk]
        self.document_chunks = document_chunks or {}  # doc_id -> list[chunk], for chunks_for_document()
        self.chunks_for_document_calls = []  # tests assert against this — which docs got text-scanned

    def hybrid_search(self, query_vector, query_text, top_k=5, doc_filter=None, folder_filter=None):
        self.last_doc_filter = doc_filter  # tests assert against this to confirm document_ids was forwarded
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

    def chunks_for_document(self, document_id, limit=2000):
        self.chunks_for_document_calls.append(document_id)
        return [dict(c) for c in self.document_chunks.get(document_id, [])]

    def chunks_by_case_number(self, case_number, case_year):
        return [dict(c) for c in self.case_number_index.get((case_number, case_year), [])]

    def chunks_by_celex_id(self, celex_id):
        return [dict(c) for c in self.celex_id_index.get(celex_id, [])]

    def chunks_by_citation_number(self, citation_number, citation_year):
        return [dict(c) for c in self.citation_number_index.get((citation_number, citation_year), [])]


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


# ── extract_celex_ids ────────────────────────────────────────────────────────

def test_extract_celex_ids_matches_plain_and_suffixed_ids():
    assert extract_celex_ids("What is 31997R0955 about?") == {"31997R0955"}
    assert extract_celex_ids("What does 31958D1127(01) concern?") == {"31958D1127(01)"}


def test_extract_celex_ids_finds_multiple():
    text = "Compare 31997R0955 and 32011D0126."
    assert extract_celex_ids(text) == {"31997R0955", "32011D0126"}


def test_extract_celex_ids_returns_empty_for_no_match():
    assert extract_celex_ids("What is the capital of France?") == set()


# ── extract_citation_numbers ─────────────────────────────────────────────────

def test_extract_citation_numbers_parses_four_digit_year():
    assert extract_citation_numbers("What does Regulation (EC) No 1021/2008 concern?") == {("1021", "2008")}


def test_extract_citation_numbers_expands_two_digit_year():
    """"No 480/86" reads as 1986 to a human, not 2086 — must match
    ingestion/chunker.py::extract_citation_number's stored 4-digit year on
    the other side of the lookup, or chunks_by_citation_number() finds
    nothing even though the document is indexed correctly."""
    assert extract_citation_numbers("Council Regulation (EEC) No 480/86") == {("480", "1986")}


def test_extract_citation_numbers_two_digit_year_pivot_at_30():
    assert extract_citation_numbers("No 1/29") == {("1", "2029")}
    assert extract_citation_numbers("No 1/30") == {("1", "1930")}


def test_extract_citation_numbers_requires_slash_not_case_number_shape():
    """Without the slash this is _CASE_NUMBER_RE's "<N> of <YYYY>" shape,
    not a citation number — the two must not collide."""
    assert extract_citation_numbers("Case 100 of 2019") == set()


def test_extract_citation_numbers_finds_multiple():
    text = "Compare Regulation No 1021/2008 and Regulation No. 780/2007."
    assert extract_citation_numbers(text) == {("1021", "2008"), ("780", "2007")}


def test_extract_citation_numbers_returns_empty_for_no_match():
    assert extract_citation_numbers("What is the capital of France?") == set()


# ── extract_party_match ───────────────────────────────────────────────────────

def test_extract_party_match_requires_all_parties_present():
    assert extract_party_match(
        "What is the case LIPU PRADHAN vs STATE OF ODISHA about?",
        ["LIPU PRADHAN", "STATE OF ODISHA"],
    ) is True


def test_extract_party_match_is_case_insensitive():
    assert extract_party_match("what is lipu pradhan vs state of odisha about", ["LIPU PRADHAN", "STATE OF ODISHA"]) is True


def test_extract_party_match_false_when_only_one_party_present():
    """Many filenames share a common party (e.g. '... vs STATE OF ODISHA') —
    matching on just one would over-promote every case involving it."""
    assert extract_party_match(
        "What is the case about STATE OF ODISHA?", ["LIPU PRADHAN", "STATE OF ODISHA"]
    ) is False


def test_extract_party_match_false_for_no_parties():
    assert extract_party_match("any query", []) is False
    assert extract_party_match("any query", None) is False


# ── identity-match guarantee (case number + party name) ────────────────────────

@pytest.mark.asyncio
async def test_retrieve_expanded_guarantees_case_number_match_even_with_low_score():
    """These short, heavily-templated interlocutory orders (case caption +
    1-2 sentence order, near-identical boilerplate across different cases)
    are exactly where semantic/BM25 scoring is least reliable — an exact
    case-number match in the query must survive even if its score alone
    wouldn't make top_k. Matching is via each chunk's own structured
    case_number/case_year metadata (see ingestion/chunker.py::
    extract_case_metadata), not by regexing its text."""
    matching_low_score = _chunk("match", "d_target", 0, 0.01,
                                 text="In the matter of Mohan Viswakarma")
    matching_low_score["case_number"] = "10691"
    matching_low_score["case_year"] = "2020"
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
    assert matched["identity_match"] is True


@pytest.mark.asyncio
async def test_retrieve_expanded_fetches_case_number_chunk_via_index_even_if_absent_from_hybrid_search():
    """The whole point of chunks_by_case_number: a document with a low/zero
    semantic score for this query mustn't be invisible just because hybrid
    search's top-`pool` candidates didn't include it — the direct payload
    index lookup must surface it independently."""
    indexed_chunk = _chunk("indexed", "d_target", 0, 0.0, text="the actual ruling text")
    indexed_chunk["case_number"] = "10691"
    indexed_chunk["case_year"] = "2020"
    vs = FakeVectorStore(
        results_by_query={"What is the case CRM 10691 of 2020 about?": []},  # nothing from hybrid search
        case_number_index={("10691", "2020"): [indexed_chunk]},
    )
    retriever = HybridRetriever(FakeEmbedder(), vs, top_k=5)

    results = await retriever.retrieve_expanded(["What is the case CRM 10691 of 2020 about?"], top_k=5)

    chunk_ids = {r["chunk_id"] for r in results}
    assert "indexed" in chunk_ids
    matched = next(r for r in results if r["chunk_id"] == "indexed")
    assert matched["identity_match"] is True
    assert matched["source"] == "case_number_index"


@pytest.mark.asyncio
async def test_retrieve_expanded_fetches_celex_id_chunk_via_index_even_if_absent_from_hybrid_search():
    """A CELEX ID never appears in its own document's body text (EU
    legislation cites itself in a different format) — hybrid search has no
    literal string to find at all without chunks_by_celex_id()."""
    indexed_chunk = _chunk("indexed", "d_target", 0, 0.0, text="the actual regulation text")
    indexed_chunk["celex_id"] = "31997R0955"
    vs = FakeVectorStore(
        results_by_query={"What is 31997R0955 about?": []},  # nothing from hybrid search
        celex_id_index={"31997R0955": [indexed_chunk]},
    )
    retriever = HybridRetriever(FakeEmbedder(), vs, top_k=5)

    results = await retriever.retrieve_expanded(["What is 31997R0955 about?"], top_k=5)

    chunk_ids = {r["chunk_id"] for r in results}
    assert "indexed" in chunk_ids
    matched = next(r for r in results if r["chunk_id"] == "indexed")
    assert matched["identity_match"] is True
    assert matched["source"] == "celex_id_index"


@pytest.mark.asyncio
async def test_retrieve_expanded_guarantees_celex_id_match_even_with_low_score():
    matching_low_score = _chunk("match", "d_target", 0, 0.01, text="fragment of the regulation")
    matching_low_score["celex_id"] = "31997R0955"
    distractor_high_score = _chunk("distractor", "d_other", 0, 0.9, text="unrelated regulation")
    distractor_high_score["celex_id"] = "32011D0126"
    vs = FakeVectorStore(results_by_query={
        "What is 31997R0955 about?": [distractor_high_score, matching_low_score],
    })
    retriever = HybridRetriever(FakeEmbedder(), vs, top_k=1)

    results = await retriever.retrieve_expanded(["What is 31997R0955 about?"], top_k=1)

    chunk_ids = {r["chunk_id"] for r in results}
    assert "match" in chunk_ids
    matched = next(r for r in results if r["chunk_id"] == "match")
    assert matched["identity_match"] is True


@pytest.mark.asyncio
async def test_retrieve_expanded_fetches_citation_number_chunk_via_index_even_if_absent_from_hybrid_search():
    """The natural short-citation form ("No 1021/2008") recurs across
    different years and gets cited in passing by other documents' body
    text — hybrid search alone isn't reliable for it, same rationale as
    the CELEX ID index above but for the path a real user (who doesn't
    know the CELEX ID) actually relies on."""
    indexed_chunk = _chunk("indexed", "d_target", 0, 0.0, text="the actual regulation text")
    indexed_chunk["citation_number"] = "1021"
    indexed_chunk["citation_year"] = "2008"
    vs = FakeVectorStore(
        results_by_query={"What does Regulation No 1021/2008 concern?": []},  # nothing from hybrid search
        citation_number_index={("1021", "2008"): [indexed_chunk]},
    )
    retriever = HybridRetriever(FakeEmbedder(), vs, top_k=5)

    results = await retriever.retrieve_expanded(["What does Regulation No 1021/2008 concern?"], top_k=5)

    chunk_ids = {r["chunk_id"] for r in results}
    assert "indexed" in chunk_ids
    matched = next(r for r in results if r["chunk_id"] == "indexed")
    assert matched["identity_match"] is True
    assert matched["source"] == "citation_number_index"


@pytest.mark.asyncio
async def test_retrieve_expanded_citation_number_index_expands_two_digit_year():
    """Query says "No 480/86"; the chunk's stored citation_year (from
    ingestion/chunker.py::extract_citation_number) is always the full
    4-digit form — the lookup only succeeds if the query-side 2-digit year
    was expanded to match before hitting the index."""
    indexed_chunk = _chunk("indexed", "d_target", 0, 0.0, text="the actual regulation text")
    indexed_chunk["citation_number"] = "480"
    indexed_chunk["citation_year"] = "1986"
    vs = FakeVectorStore(
        results_by_query={"What does Regulation No 480/86 concern?": []},
        citation_number_index={("480", "1986"): [indexed_chunk]},
    )
    retriever = HybridRetriever(FakeEmbedder(), vs, top_k=5)

    results = await retriever.retrieve_expanded(["What does Regulation No 480/86 concern?"], top_k=5)

    matched = next((r for r in results if r["chunk_id"] == "indexed"), None)
    assert matched is not None
    assert matched["identity_match"] is True


@pytest.mark.asyncio
async def test_retrieve_expanded_guarantees_citation_number_match_even_with_low_score():
    matching_low_score = _chunk("match", "d_target", 0, 0.01, text="fragment of the regulation")
    matching_low_score["citation_number"] = "1021"
    matching_low_score["citation_year"] = "2008"
    distractor_high_score = _chunk("distractor", "d_other", 0, 0.9, text="unrelated regulation")
    distractor_high_score["citation_number"] = "780"
    distractor_high_score["citation_year"] = "2007"
    vs = FakeVectorStore(results_by_query={
        "What does Regulation No 1021/2008 concern?": [distractor_high_score, matching_low_score],
    })
    retriever = HybridRetriever(FakeEmbedder(), vs, top_k=1)

    results = await retriever.retrieve_expanded(["What does Regulation No 1021/2008 concern?"], top_k=1)

    chunk_ids = {r["chunk_id"] for r in results}
    assert "match" in chunk_ids
    matched = next(r for r in results if r["chunk_id"] == "match")
    assert matched["identity_match"] is True


@pytest.mark.asyncio
async def test_retrieve_expanded_guarantees_party_name_match_even_with_low_score():
    matching_low_score = _chunk("match", "d_target", 0, 0.01, text="Heard learned counsel.")
    matching_low_score["parties"] = ["LIPU PRADHAN", "STATE OF ODISHA"]
    distractor_high_score = _chunk("distractor", "d_other", 0, 0.9, text="an unrelated ruling")
    distractor_high_score["parties"] = ["SOMEONE ELSE", "STATE OF ODISHA"]
    query = "What is the case LIPU PRADHAN vs STATE OF ODISHA about?"
    vs = FakeVectorStore(results_by_query={query: [distractor_high_score, matching_low_score]})
    retriever = HybridRetriever(FakeEmbedder(), vs, top_k=1)

    results = await retriever.retrieve_expanded([query], top_k=1)

    chunk_ids = {r["chunk_id"] for r in results}
    assert "match" in chunk_ids
    matched = next(r for r in results if r["chunk_id"] == "match")
    assert matched["identity_match"] is True


@pytest.mark.asyncio
async def test_retrieve_expanded_no_identity_signal_in_query_is_a_no_op():
    chunk1 = _chunk("c1", "d1", 0, 0.5, text="some ordinary text")
    vs = FakeVectorStore(results_by_query={"a generic question": [chunk1]})
    retriever = HybridRetriever(FakeEmbedder(), vs, top_k=5)

    results = await retriever.retrieve_expanded(["a generic question"], top_k=5)

    assert "identity_match" not in results[0]


# ── document_ids scope (the "compare N specific documents" filter) ──────────
# Previously "compare" only ever *hinted* at scope via filenames spelled out
# in the question text plus a folder filter — hybrid search (and the
# reranker) stayed free to pull in other documents from the same folder.

async def test_retrieve_expanded_forwards_document_ids_to_hybrid_search_as_doc_filter():
    vs = FakeVectorStore(results_by_query={"compare a and b": []})
    retriever = HybridRetriever(FakeEmbedder(), vs, top_k=5)

    await retriever.retrieve_expanded(["compare a and b"], top_k=5, document_ids=["d1", "d2"])

    assert sorted(vs.last_doc_filter) == ["d1", "d2"]


async def test_retrieve_expanded_no_document_ids_means_no_doc_filter():
    vs = FakeVectorStore(results_by_query={"a generic question": []})
    retriever = HybridRetriever(FakeEmbedder(), vs, top_k=5)

    await retriever.retrieve_expanded(["a generic question"], top_k=5)

    assert vs.last_doc_filter is None


async def test_retrieve_expanded_document_ids_scopes_hybrid_search_results():
    in_scope = _chunk("c1", "d1", 0, 0.5)
    vs = FakeVectorStore(results_by_query={"q": [in_scope]})
    retriever = HybridRetriever(FakeEmbedder(), vs, top_k=5)

    results = await retriever.retrieve_expanded(["q"], top_k=5, document_ids=["d1", "d2"])

    assert {r["document_id"] for r in results} == {"d1"}


async def test_retrieve_expanded_document_ids_drops_out_of_scope_identity_index_hit():
    """The case/CELEX/citation-number index lookups are exact-key fetches,
    not searches — they don't take a doc filter themselves (see
    vector_db/qdrant_client.py::chunks_by_celex_id and friends), so an
    identity match for a document outside document_ids has to be caught
    after the fact instead. Two documents sharing a CELEX ID doesn't happen
    in practice, but the safety-net filter shouldn't care why an
    out-of-scope hit showed up — it should be dropped either way."""
    out_of_scope = _chunk("indexed", "d_other", 0, 0.0, text="the actual regulation text")
    out_of_scope["celex_id"] = "31997R0955"
    vs = FakeVectorStore(
        results_by_query={"What is 31997R0955 about?": []},
        celex_id_index={"31997R0955": [out_of_scope]},
    )
    retriever = HybridRetriever(FakeEmbedder(), vs, top_k=5)

    results = await retriever.retrieve_expanded(
        ["What is 31997R0955 about?"], top_k=5, document_ids=["d1", "d2"]
    )

    assert "indexed" not in {r["chunk_id"] for r in results}
    assert "d_other" not in {r["document_id"] for r in results}


async def test_retrieve_expanded_document_ids_keeps_in_scope_identity_index_hit():
    in_scope = _chunk("indexed", "d1", 0, 0.0, text="the actual regulation text")
    in_scope["celex_id"] = "31997R0955"
    vs = FakeVectorStore(
        results_by_query={"What is 31997R0955 about?": []},
        celex_id_index={"31997R0955": [in_scope]},
    )
    retriever = HybridRetriever(FakeEmbedder(), vs, top_k=5)

    results = await retriever.retrieve_expanded(
        ["What is 31997R0955 about?"], top_k=5, document_ids=["d1", "d2"]
    )

    assert "indexed" in {r["chunk_id"] for r in results}


# ── structural Figure-N/Table-N reference lookup ─────────────────────────────
# Narrow, targeted mechanism (unlike the reverted two-stage pool-widening
# attempt — see project memory): only fires when the query itself names a
# figure/table explicitly, text-scans at most STRUCTURAL_TOP_DOCS documents,
# and adds at most one chunk per document.

def test_extract_structural_references_parses_multiple_styles():
    assert extract_structural_references("What does Figure 7 show?") == {("Figure", "7")}
    assert extract_structural_references("See Fig. 3 for details.") == {("Figure", "3")}
    assert extract_structural_references("Fig 12 illustrates this.") == {("Figure", "12")}
    assert extract_structural_references("What does Table 4 report?") == {("Table", "4")}
    assert extract_structural_references("Compare Figure 3a and Figure 3b.") == {("Figure", "3a"), ("Figure", "3b")}


def test_extract_structural_references_empty_when_none_present():
    assert extract_structural_references("What is the main contribution of this paper?") == set()


def test_structural_match_tier_colon_is_strongest():
    text = "Figure 7: Comparison of methods across benchmarks."
    assert structural_match_tier(text, "Figure", "7") == 0


def test_structural_match_tier_pipe_is_also_strongest():
    text = "Table 3 | Ablation results for the full model."
    assert structural_match_tier(text, "Table", "3") == 0


def test_structural_match_tier_period_is_medium():
    text = "Fig. 3. Comparison of methods across benchmarks."
    assert structural_match_tier(text, "Figure", "3") == 1


def test_structural_match_tier_bare_is_weakest():
    text = "Fig. 1 Mechanistic World Models shift AI from forecasting to discovery."
    assert structural_match_tier(text, "Figure", "1") == 2


def test_structural_match_tier_rejects_a_plain_in_text_reference():
    text = "As shown in Figure 7, the results are consistent with our hypothesis."
    assert structural_match_tier(text, "Figure", "7") is None


def test_structural_match_tier_rejects_discourse_connective_after_punctuation():
    """The exact false-positive shape the golden-builder rewrite also
    guards against: punctuation that looks like a caption opener but is
    actually just a paragraph continuing after an in-text citation."""
    text = "As illustrated in Figure 1. Finally, we discuss the capabilities of this approach."
    assert structural_match_tier(text, "Figure", "1") is None


def test_structural_match_tier_wrong_number_does_not_match():
    text = "Figure 7: Comparison of methods."
    assert structural_match_tier(text, "Figure", "3") is None


def test_structural_match_tier_wrong_kind_does_not_match():
    text = "Table 7: Comparison of methods."
    assert structural_match_tier(text, "Figure", "7") is None


def test_best_structural_chunk_picks_the_strongest_tier_among_candidates():
    weak = {"chunk_id": "c1", "text": "As shown in Figure 7, results improve."}
    strong = {"chunk_id": "c2", "text": "Figure 7: Comparison of methods across benchmarks."}
    assert best_structural_chunk([weak, strong], "Figure", "7") is strong


def test_best_structural_chunk_returns_none_when_nothing_matches():
    chunks = [{"chunk_id": "c1", "text": "Unrelated body text with no figure references at all."}]
    assert best_structural_chunk(chunks, "Figure", "7") is None


@pytest.mark.asyncio
async def test_retrieve_expanded_adds_the_structural_chunk_when_found():
    stage1_hit = _chunk("c1", "d1", 0, 0.9, text="unrelated top-scoring chunk")
    caption_chunk = {"chunk_id": "caption", "document_id": "d1", "chunk_index": 40,
                      "text": "Figure 7: Comparison of diagonal-QFI and full-QFI QEWC.",
                      "page_num": 12, "filename": "f.pdf", "folder": ""}
    vs = FakeVectorStore(
        results_by_query={"What does Figure 7 show?": [stage1_hit]},
        document_chunks={"d1": [caption_chunk]},
    )
    retriever = HybridRetriever(FakeEmbedder(), vs, top_k=5)

    results = await retriever.retrieve_expanded(["What does Figure 7 show?"], top_k=5)

    by_id = {r["chunk_id"]: r for r in results}
    assert "caption" in by_id
    assert by_id["caption"]["source"] == "structural_reference"
    assert by_id["caption"]["identity_match"] is True
    assert by_id["caption"]["score"] == 1.0


@pytest.mark.asyncio
async def test_retrieve_expanded_structural_lookup_is_a_noop_without_a_query_reference():
    stage1_hit = _chunk("c1", "d1", 0, 0.9)
    vs = FakeVectorStore(
        results_by_query={"What is the main contribution?": [stage1_hit]},
        document_chunks={"d1": [{"chunk_id": "caption", "document_id": "d1", "chunk_index": 40,
                                  "text": "Figure 7: Comparison of diagonal-QFI and full-QFI QEWC.",
                                  "page_num": 12, "filename": "f.pdf", "folder": ""}]},
    )
    retriever = HybridRetriever(FakeEmbedder(), vs, top_k=5)

    results = await retriever.retrieve_expanded(["What is the main contribution?"], top_k=5)

    assert "caption" not in {r["chunk_id"] for r in results}
    assert vs.chunks_for_document_calls == []


@pytest.mark.asyncio
async def test_retrieve_expanded_structural_lookup_applies_within_doc_scope():
    """Regression test: this used to be skipped entirely whenever
    document_ids was set (the compare/"user already has a document open"
    flow), on the assumption that documents already being known made the
    scan redundant. Measured wrong: a title-free, document-scoped caption
    question dropped Evidence-chunk Recall from 91.5% to 69.5% purely
    because of this skip (project memory / eval/mixed_corpus/README.md's
    title-free robustness check) — a document being explicitly selected
    doesn't mean its caption was already found by hybrid search."""
    stage1_hit = _chunk("c1", "d1", 0, 0.9)
    vs = FakeVectorStore(
        results_by_query={"What does Figure 7 show?": [stage1_hit]},
        document_chunks={"d1": [{"chunk_id": "caption", "document_id": "d1", "chunk_index": 40,
                                  "text": "Figure 7: Comparison of diagonal-QFI and full-QFI QEWC.",
                                  "page_num": 12, "filename": "f.pdf", "folder": ""}]},
    )
    retriever = HybridRetriever(FakeEmbedder(), vs, top_k=5)

    results = await retriever.retrieve_expanded(
        ["What does Figure 7 show?"], top_k=5, document_ids=["d1"]
    )

    by_id = {r["chunk_id"]: r for r in results}
    assert "caption" in by_id
    assert by_id["caption"]["source"] == "structural_reference"
    assert by_id["caption"]["identity_match"] is True
    assert vs.chunks_for_document_calls == ["d1"]


@pytest.mark.asyncio
async def test_retrieve_expanded_structural_lookup_scans_every_scoped_document():
    """Scoped documents are already user-confirmed relevant, not a stage-1
    guess — unlike the unscoped case, scanning isn't capped at
    STRUCTURAL_TOP_DOCS. Uses more scoped documents than that cap to prove
    it."""
    n = HybridRetriever.STRUCTURAL_TOP_DOCS + 2
    doc_ids = [f"d{i}" for i in range(n)]
    stage1_hits = [_chunk(f"c{i}", doc_ids[i], 0, 1.0 - i * 0.1) for i in range(n)]
    vs = FakeVectorStore(results_by_query={"What does Figure 7 show?": stage1_hits})
    retriever = HybridRetriever(FakeEmbedder(), vs, top_k=5)

    await retriever.retrieve_expanded(["What does Figure 7 show?"], top_k=5, document_ids=doc_ids)

    assert set(vs.chunks_for_document_calls) == set(doc_ids)


@pytest.mark.asyncio
async def test_retrieve_expanded_structural_lookup_scoped_never_touches_an_out_of_scope_document():
    """Comparison-flow safety: a document outside doc_scope must never be
    scanned or have a chunk injected, even if it would otherwise be the
    strongest stage-1 candidate."""
    in_scope_hit = _chunk("c1", "d1", 0, 0.5)
    out_of_scope_hit = _chunk("c2", "d2", 0, 0.99)  # scores higher, but not selected
    vs = FakeVectorStore(
        results_by_query={"What does Figure 7 show?": [in_scope_hit, out_of_scope_hit]},
        document_chunks={
            "d1": [{"chunk_id": "cap1", "document_id": "d1", "chunk_index": 40,
                     "text": "Figure 7: In-scope caption.", "page_num": 12, "filename": "f.pdf", "folder": ""}],
            "d2": [{"chunk_id": "cap2", "document_id": "d2", "chunk_index": 40,
                     "text": "Figure 7: Out-of-scope caption.", "page_num": 12, "filename": "g.pdf", "folder": ""}],
        },
    )
    retriever = HybridRetriever(FakeEmbedder(), vs, top_k=5)

    results = await retriever.retrieve_expanded(
        ["What does Figure 7 show?"], top_k=5, document_ids=["d1"]
    )

    assert vs.chunks_for_document_calls == ["d1"]
    assert "cap2" not in {r["chunk_id"] for r in results}


@pytest.mark.asyncio
async def test_retrieve_expanded_structural_lookup_adds_at_most_one_chunk_per_document():
    stage1_hit = _chunk("c1", "d1", 0, 0.9)
    doc_chunks = [
        {"chunk_id": "cap_a", "document_id": "d1", "chunk_index": 10,
         "text": "Figure 7: First mention that looks like a caption.",
         "page_num": 5, "filename": "f.pdf", "folder": ""},
        {"chunk_id": "cap_b", "document_id": "d1", "chunk_index": 40,
         "text": "Figure 7: Second, unrelated recurrence of the same label.",
         "page_num": 20, "filename": "f.pdf", "folder": ""},
    ]
    vs = FakeVectorStore(
        results_by_query={"What does Figure 7 show?": [stage1_hit]},
        document_chunks={"d1": doc_chunks},
    )
    retriever = HybridRetriever(FakeEmbedder(), vs, top_k=5)

    results = await retriever.retrieve_expanded(["What does Figure 7 show?"], top_k=5)

    structural_hits = [r for r in results if r.get("source") == "structural_reference"]
    assert len(structural_hits) == 1


@pytest.mark.asyncio
async def test_retrieve_expanded_structural_lookup_bounded_to_top_candidate_documents():
    n = HybridRetriever.STRUCTURAL_TOP_DOCS + 2
    chunks = [_chunk(f"c{i}", f"d{i}", 0, 1.0 - i * 0.1) for i in range(n)]
    vs = FakeVectorStore(results_by_query={"What does Figure 7 show?": chunks})
    retriever = HybridRetriever(FakeEmbedder(), vs, top_k=5)

    await retriever.retrieve_expanded(["What does Figure 7 show?"], top_k=5)

    assert len(vs.chunks_for_document_calls) == HybridRetriever.STRUCTURAL_TOP_DOCS
    assert set(vs.chunks_for_document_calls) == {f"d{i}" for i in range(HybridRetriever.STRUCTURAL_TOP_DOCS)}


@pytest.mark.asyncio
async def test_retrieve_expanded_structural_lookup_survives_a_low_rerank_score_via_top_k():
    """identity_match=True guarantees survival into the returned top_k_
    chunks even when the structural chunk's score wouldn't otherwise have
    made a small top_k cut — same guarantee case/CELEX/citation-number
    matches already get."""
    stage1_hits = [_chunk(f"c{i}", "d1", i, 0.9 - i * 0.01) for i in range(5)]
    caption_chunk = {"chunk_id": "caption", "document_id": "d1", "chunk_index": 40,
                      "text": "Figure 7: Comparison of diagonal-QFI and full-QFI QEWC.",
                      "page_num": 12, "filename": "f.pdf", "folder": ""}
    vs = FakeVectorStore(
        results_by_query={"What does Figure 7 show?": stage1_hits},
        document_chunks={"d1": [caption_chunk]},
    )
    retriever = HybridRetriever(FakeEmbedder(), vs, top_k=1)  # tiny top_k

    results = await retriever.retrieve_expanded(["What does Figure 7 show?"], top_k=1)

    assert "caption" in {r["chunk_id"] for r in results}


@pytest.mark.asyncio
async def test_retrieve_expanded_promotes_a_structural_match_already_found_by_hybrid_search():
    """Regression test for a real bug found by the caption-miss taxonomy
    (project memory: 112/127 real misses traced to this): the structural
    injection used to fire only 'if key not in all_chunks', so a chunk
    hybrid search had ALREADY found — however weakly — never got the
    identity-match promotion. Earlier, weaker detection silently defeated
    the later, stronger structural signal instead of deferring to it.
    All competing chunks below belong to the SAME document so the
    per-document diversity guarantee can't accidentally rescue this case
    and mask a regression."""
    caption_text = "Figure 7: Comparison of diagonal-QFI and full-QFI QEWC."
    caption_low_score = _chunk("caption", "d1", 40, 0.05, text=caption_text)
    stronger_same_doc_hits = [_chunk(f"c{i}", "d1", i, 0.9 - i * 0.01) for i in range(4)]
    vs = FakeVectorStore(
        results_by_query={"What does Figure 7 show?": [caption_low_score] + stronger_same_doc_hits},
        document_chunks={"d1": [dict(caption_low_score)]},
    )
    retriever = HybridRetriever(FakeEmbedder(), vs, top_k=1)  # tiny top_k — only identity_match survives this

    results = await retriever.retrieve_expanded(["What does Figure 7 show?"], top_k=1)

    result_ids = [r["chunk_id"] for r in results]
    by_id = {r["chunk_id"]: r for r in results}
    assert by_id["caption"]["identity_match"] is True
    assert by_id["caption"]["source"] == "structural_reference"
    assert by_id["caption"]["score"] == 1.0
    assert result_ids.count("caption") == 1


@pytest.mark.asyncio
async def test_retrieve_expanded_structural_promotion_equivalent_whether_or_not_hybrid_found_it_first():
    """The invariant: the promoted chunk must look identical whether
    hybrid search missed the caption chunk entirely, or found the exact
    same chunk first with a weak score."""
    caption_text = "Figure 7: Comparison of diagonal-QFI and full-QFI QEWC."
    stronger_same_doc_hits = [_chunk(f"c{i}", "d1", i, 0.9 - i * 0.01) for i in range(4)]

    # Scenario A: hybrid search never surfaces the caption chunk at all.
    vs_absent = FakeVectorStore(
        results_by_query={"What does Figure 7 show?": stronger_same_doc_hits},
        document_chunks={"d1": [{"chunk_id": "caption", "document_id": "d1", "chunk_index": 40,
                                  "text": caption_text, "page_num": 12, "filename": "f.pdf", "folder": ""}]},
    )
    retriever_absent = HybridRetriever(FakeEmbedder(), vs_absent, top_k=1)
    results_absent = await retriever_absent.retrieve_expanded(["What does Figure 7 show?"], top_k=1)

    # Scenario B: hybrid search already found the SAME chunk, weakly.
    caption_low_score = _chunk("caption", "d1", 40, 0.05, text=caption_text)
    vs_weak = FakeVectorStore(
        results_by_query={"What does Figure 7 show?": [caption_low_score] + stronger_same_doc_hits},
        document_chunks={"d1": [dict(caption_low_score)]},
    )
    retriever_weak = HybridRetriever(FakeEmbedder(), vs_weak, top_k=1)
    results_weak = await retriever_weak.retrieve_expanded(["What does Figure 7 show?"], top_k=1)

    for results in (results_absent, results_weak):
        by_id = {r["chunk_id"]: r for r in results}
        assert "caption" in by_id
        assert by_id["caption"]["identity_match"] is True
        assert by_id["caption"]["source"] == "structural_reference"
        assert by_id["caption"]["score"] == 1.0


# ── promote_identity_matches ─────────────────────────────────────────────────

def _rr_chunk(chunk_id, rerank_score=None, identity_match=False):
    c = {"chunk_id": chunk_id, "text": "text"}
    if rerank_score is not None:
        c["rerank_score"] = rerank_score
    if identity_match:
        c["identity_match"] = True
    return c


def test_promote_identity_matches_is_a_no_op_without_a_match():
    chunks = [_rr_chunk("a")]
    top_chunks = [_rr_chunk("a", rerank_score=0.5)]
    assert promote_identity_matches(chunks, top_chunks, relevance_threshold=1.5) == top_chunks


def test_promote_identity_matches_adds_missing_match_above_threshold():
    matched = _rr_chunk("match", identity_match=True)  # not yet reranked, no score
    chunks = [matched, _rr_chunk("other", identity_match=False)]
    top_chunks = [_rr_chunk("other", rerank_score=3.0)]  # reranker's own top pick, no match

    result = promote_identity_matches(chunks, top_chunks, relevance_threshold=1.5)

    ids = {c["chunk_id"] for c in result}
    assert "match" in ids
    promoted = next(c for c in result if c["chunk_id"] == "match")
    assert promoted["rerank_score"] > 1.5


def test_promote_identity_matches_boosts_already_included_low_score():
    """If the reranker DID include the match but scored it below threshold,
    raise its score rather than add a duplicate."""
    matched = _rr_chunk("match", identity_match=True)
    chunks = [matched]
    top_chunks = [_rr_chunk("match", rerank_score=0.1)]  # reranker scored it too low

    result = promote_identity_matches(chunks, top_chunks, relevance_threshold=1.5)

    assert len(result) == 1
    assert result[0]["rerank_score"] > 1.5


def test_promote_identity_matches_does_not_drop_the_match_even_over_top_k():
    matched = _rr_chunk("match", identity_match=True)
    chunks = [matched]
    top_chunks = [_rr_chunk("a", rerank_score=5.0), _rr_chunk("b", rerank_score=4.0)]

    result = promote_identity_matches(chunks, top_chunks, relevance_threshold=1.5)

    ids = {c["chunk_id"] for c in result}
    assert "match" in ids
    assert len(result) == 3  # exceeds top_k=2 rather than drop the match


# ── promote_document_opening_chunks ─────────────────────────────────────────────

def test_promote_document_opening_chunks_adds_missing_opening_chunk():
    """The reranker often scores a document's title/citation chunk (index 0)
    lower than surrounding content-specific chunks, cutting it from the
    final top_k even though a later chunk continuing its sentence reads as
    a fragment without it — see eval/README.md, "Known issue: multi-doc
    near-duplicate-title flakiness", root-caused via this exact scenario."""
    opening = _chunk("c0", "d1", 0, 0.5, text="This Order may be cited as X and shall come into force on...")
    content = _chunk("c2", "d1", 2, 0.9, text="...the day after the day on which it is made.")
    chunks = [opening, content]
    top_chunks = [content]  # reranker kept only the fragment, dropped the opening

    result = promote_document_opening_chunks(chunks, top_chunks)

    ids = {c["chunk_id"] for c in result}
    assert ids == {"c0", "c2"}


def test_promote_document_opening_chunks_is_a_no_op_when_opening_already_present():
    opening = _chunk("c0", "d1", 0, 0.9, text="Opening.")
    chunks = [opening]
    top_chunks = [opening]

    result = promote_document_opening_chunks(chunks, top_chunks)

    assert result == top_chunks


def test_promote_document_opening_chunks_is_a_no_op_when_opening_not_in_pool():
    content = _chunk("c2", "d1", 2, 0.9, text="Fragment.")
    chunks = [content]  # opening chunk never made it into the candidate pool at all
    top_chunks = [content]

    result = promote_document_opening_chunks(chunks, top_chunks)

    assert result == top_chunks


def test_promote_document_opening_chunks_ignores_documents_not_in_top_chunks():
    """A document's opening chunk sitting in the broader candidate pool
    shouldn't be pulled in just because it exists — only documents already
    represented in top_chunks get their opening chunk added."""
    other_doc_opening = _chunk("c0", "d2", 0, 0.9, text="Unrelated document's opening.")
    chunks = [other_doc_opening]
    top_chunks = [_chunk("c5", "d1", 5, 0.8, text="d1 content, not d2")]

    result = promote_document_opening_chunks(chunks, top_chunks)

    assert result == top_chunks


def test_promote_document_opening_chunks_handles_multiple_documents():
    d1_opening = _chunk("d1c0", "d1", 0, 0.5, text="d1 opening")
    d1_content = _chunk("d1c3", "d1", 3, 0.9, text="d1 content")
    d2_opening = _chunk("d2c0", "d2", 0, 0.5, text="d2 opening")
    d2_content = _chunk("d2c1", "d2", 1, 0.8, text="d2 content")
    chunks = [d1_opening, d1_content, d2_opening, d2_content]
    top_chunks = [d1_content, d2_content]

    result = promote_document_opening_chunks(chunks, top_chunks)

    ids = {c["chunk_id"] for c in result}
    assert ids == {"d1c0", "d1c3", "d2c0", "d2c1"}
