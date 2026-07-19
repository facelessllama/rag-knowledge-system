"""
Tests for vector_db/qdrant_client.py's upsert_chunks():
1. chunks/vectors length must match — zip() silently drops the mismatched
   remainder otherwise, so an embedding backend returning fewer vectors than
   chunks would vanish chunks from Qdrant with no error, while Postgres'
   chunk count (set independently) keeps recording the full number.
2. Point IDs are deterministic (point_id_for_chunk, derived from chunk_id),
   not a fresh uuid4() per call — re-upserting the same chunk_id overwrites
   its existing point instead of creating a duplicate.

The real QdrantClient is never constructed — VectorStore.__init__ requires
a live connection attempt via qdrant_client's own constructor, so these
tests build a bare instance and monkeypatch `.client` with a fake recorder.
"""
import pytest

from ingestion.chunker import TextChunk
from vector_db.qdrant_client import VectorStore, point_id_for_chunk


def _chunk(chunk_id, document_id="doc1", page_num=1, chunk_index=0, text="hello world"):
    return TextChunk(
        chunk_id=chunk_id, text=text, page_num=page_num, chunk_index=chunk_index,
        char_count=len(text), has_ocr=False, document_id=document_id,
        char_start=0, char_end=len(text),
    )


class _FakeQdrantClient:
    def __init__(self):
        self.upsert_calls = []

    def upsert(self, collection_name, points):
        self.upsert_calls.append((collection_name, points))


@pytest.fixture
def vs():
    store = object.__new__(VectorStore)  # bypass __init__ — no real connection needed
    store.client = _FakeQdrantClient()
    store.collection = "knowledge_base"
    store.timeout = 5.0
    return store


# ── length mismatch ──────────────────────────────────────────────────────────

def test_upsert_chunks_rejects_more_chunks_than_vectors(vs):
    chunks = [_chunk("c1"), _chunk("c2"), _chunk("c3")]
    vectors = [[0.1], [0.2]]  # one short — simulates a truncated embedding batch
    with pytest.raises(ValueError, match="3 chunks but 2 vectors"):
        vs.upsert_chunks(chunks, vectors)
    assert vs.client.upsert_calls == []  # must fail before ever calling Qdrant


def test_upsert_chunks_rejects_more_vectors_than_chunks(vs):
    chunks = [_chunk("c1")]
    vectors = [[0.1], [0.2]]
    with pytest.raises(ValueError, match="1 chunks but 2 vectors"):
        vs.upsert_chunks(chunks, vectors)
    assert vs.client.upsert_calls == []


def test_upsert_chunks_rejects_duplicate_chunk_id_in_same_batch(vs):
    """point_id_for_chunk() is a pure function of chunk_id — two chunks
    sharing one chunk_id would map to the same point ID, and Qdrant would
    silently keep only one of them. Same class of element loss as the
    length mismatch above, just via a duplicate key instead of a short
    list."""
    chunks = [_chunk("c1"), _chunk("c1")]  # same chunk_id twice
    vectors = [[0.1], [0.2]]
    with pytest.raises(ValueError, match="duplicate chunk_id"):
        vs.upsert_chunks(chunks, vectors)
    assert vs.client.upsert_calls == []  # must fail before ever calling Qdrant


def test_upsert_chunks_accepts_matching_lengths(vs):
    chunks = [_chunk("c1"), _chunk("c2")]
    vectors = [[0.1], [0.2]]
    vs.upsert_chunks(chunks, vectors)  # must not raise
    assert len(vs.client.upsert_calls) == 1
    _, points = vs.client.upsert_calls[0]
    assert len(points) == 2


# ── deterministic point IDs ──────────────────────────────────────────────────

def test_point_id_for_chunk_is_deterministic():
    assert point_id_for_chunk("doc1_p1_c0") == point_id_for_chunk("doc1_p1_c0")


def test_point_id_for_chunk_differs_across_chunk_ids():
    assert point_id_for_chunk("doc1_p1_c0") != point_id_for_chunk("doc1_p1_c1")


def test_point_id_for_chunk_is_a_valid_uuid_string():
    import uuid
    # Qdrant point IDs must be an unsigned int or a valid UUID — chunk_id
    # itself (e.g. "doc1_p1_c0") isn't one, so this must actually parse.
    uuid.UUID(point_id_for_chunk("doc1_p1_c0"))


def test_upsert_chunks_reuses_the_same_point_id_for_a_repeated_chunk_id(vs):
    """The core idempotency fix: upserting the same chunk_id twice (e.g. a
    retried upload after a partial failure) must produce the SAME point ID
    both times, so Qdrant overwrites in place instead of leaving a
    duplicate point behind."""
    chunk = _chunk("doc1_p1_c0")
    vs.upsert_chunks([chunk], [[0.1]])
    vs.upsert_chunks([chunk], [[0.1]])

    first_points = vs.client.upsert_calls[0][1]
    second_points = vs.client.upsert_calls[1][1]
    assert first_points[0].id == second_points[0].id == point_id_for_chunk("doc1_p1_c0")


def test_upsert_chunks_gives_different_chunks_different_point_ids(vs):
    chunks = [_chunk("c1"), _chunk("c2", chunk_index=1)]
    vs.upsert_chunks(chunks, [[0.1], [0.2]])
    _, points = vs.client.upsert_calls[0]
    assert points[0].id != points[1].id
    assert points[0].id == point_id_for_chunk("c1")
    assert points[1].id == point_id_for_chunk("c2")
