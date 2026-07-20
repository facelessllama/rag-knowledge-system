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
import vector_db.qdrant_client as qdrant_client_module


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


# ── size-based upsert batching ────────────────────────────────────────────────
# Qdrant's default max request size is 32 MB (REST and gRPC alike). A single
# document's chunks were observed exceeding that in one upsert() call on the
# two largest documents of a real 762-doc corpus — well within the app's own
# 50 MB per-file upload limit, so a reachable production failure.

class _FakeSizedPoint:
    """Minimal stand-in for PointStruct exposing only what
    _point_size_bytes() needs, so batch-size math can be tested with exact,
    deterministic sizes instead of depending on real embedding dimensions."""
    def __init__(self, json_str):
        self._json_str = json_str

    def model_dump_json(self):
        return self._json_str


def test_batch_points_by_size_splits_when_over_threshold():
    points = [_FakeSizedPoint("x" * 10) for _ in range(5)]  # 10 bytes each
    batches = qdrant_client_module._batch_points_by_size(points, max_bytes=25)
    assert sum(len(b) for b in batches) == 5  # no point lost
    assert [len(b) for b in batches] == [2, 2, 1]  # 20<=25 but 30>25 per batch


def test_batch_points_by_size_keeps_everything_in_one_batch_when_under_threshold():
    points = [_FakeSizedPoint("x" * 10) for _ in range(5)]
    batches = qdrant_client_module._batch_points_by_size(points, max_bytes=1000)
    assert len(batches) == 1
    assert len(batches[0]) == 5


def test_batch_points_by_size_gives_an_oversized_single_point_its_own_batch():
    """A point larger than max_bytes on its own must still be sent (nothing
    more can shrink it) rather than dropped or merged into an over-limit
    batch."""
    oversized = _FakeSizedPoint("x" * 100)
    points = [_FakeSizedPoint("x" * 10), oversized, _FakeSizedPoint("x" * 10)]
    batches = qdrant_client_module._batch_points_by_size(points, max_bytes=50)
    assert sum(len(b) for b in batches) == 3
    oversized_batch = next(b for b in batches if oversized in b)
    assert oversized_batch == [oversized]


def test_upsert_chunks_issues_one_upsert_call_when_under_the_batch_threshold(vs):
    chunks = [_chunk("c1"), _chunk("c2"), _chunk("c3")]
    vectors = [[0.1], [0.2], [0.3]]
    vs.upsert_chunks(chunks, vectors)
    assert len(vs.client.upsert_calls) == 1
    assert len(vs.client.upsert_calls[0][1]) == 3


def test_upsert_chunks_issues_multiple_upsert_calls_when_over_the_batch_threshold(vs, monkeypatch):
    """End-to-end: upsert_chunks() itself must split into multiple Qdrant
    calls once real serialized points exceed the configured threshold, and
    every chunk must still arrive exactly once across the split calls."""
    monkeypatch.setattr(qdrant_client_module, "_MAX_UPSERT_BATCH_BYTES", 1)  # forces one point per batch
    chunks = [_chunk("c1"), _chunk("c2"), _chunk("c3")]
    vectors = [[0.1], [0.2], [0.3]]
    vs.upsert_chunks(chunks, vectors)
    assert len(vs.client.upsert_calls) == 3
    all_points = [p for _, batch_points in vs.client.upsert_calls for p in batch_points]
    assert {p.id for p in all_points} == {point_id_for_chunk(c) for c in ("c1", "c2", "c3")}
