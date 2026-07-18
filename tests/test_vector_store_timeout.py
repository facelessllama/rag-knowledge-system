"""
VectorStore must pass its `timeout` through to QdrantClient explicitly
(rather than relying on qdrant_client's own undocumented-in-this-codebase
default) — this is the actual bound api/main.py's metadata-mutation
reconciliation helpers rely on: their asyncio.wait_for() backstop
(QDRANT_RECONCILE_TIMEOUT_SECONDS) is only a safe no-op-under-normal-
conditions guard *because* this client-level timeout is real and shorter.
See api/main.py's QDRANT_REQUEST_TIMEOUT_SECONDS/QDRANT_RECONCILE_TIMEOUT_
SECONDS comments.
"""
import vector_db.qdrant_client as vs_module


def test_vector_store_passes_timeout_to_qdrant_client(monkeypatch):
    captured = {}

    def _fake_qdrant_client(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(vs_module, "QdrantClient", _fake_qdrant_client)

    vs_module.VectorStore(url="http://example:6333", collection="c", api_key=None, timeout=7.5)

    assert captured["timeout"] == 7.5


def test_vector_store_timeout_defaults_to_five_seconds(monkeypatch):
    captured = {}

    def _fake_qdrant_client(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(vs_module, "QdrantClient", _fake_qdrant_client)

    vs_module.VectorStore(url="http://example:6333", collection="c")

    assert captured["timeout"] == 5.0
