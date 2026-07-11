"""
Tests for rag/retriever.py — multi-query merge/boost logic, per-document
result guarantee, and neighbor expansion. Embedding and Qdrant are faked
(no real GPU/network calls) so these run fast and deterministically.
"""
import pytest

from rag.retriever import HybridRetriever


class FakeEmbedder:
    def embed_text(self, text):
        return [0.1, 0.2, 0.3]


class FakeVectorStore:
    def __init__(self, results_by_query=None, neighbors_by_doc=None):
        self.results_by_query = results_by_query or {}
        self.neighbors_by_doc = neighbors_by_doc or {}

    def hybrid_search(self, query_vector, query_text, top_k=5, doc_filter=None, folder_filter=None):
        return [dict(r) for r in self.results_by_query.get(query_text, [])]

    def neighbor_chunks(self, document_id, chunk_indices):
        candidates = self.neighbors_by_doc.get(document_id, [])
        return [dict(n) for n in candidates if n["chunk_index"] in chunk_indices]


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
