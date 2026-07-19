"""
VectorStore.create_collection() must treat "this name already resolves to
something" (via get_collection(), which works for both a physical
collection AND an alias) as "nothing to create" — not "is this name in
get_collections()'s list", which only ever lists physical collections.
Verified live against a real Qdrant instance that creating a collection
whose name is already an alias fails with a 400 ("Alias with the same name
already exists"); the old membership check would have walked straight into
that on every startup() after scripts/migrate_to_hybrid_schema.py's alias
cutover.
"""
from vector_db.qdrant_client import VectorStore


class _FakeQdrantClient:
    def __init__(self, existing_names=()):
        self._existing = set(existing_names)
        self.create_collection_calls = []
        self.create_payload_index_calls = []

    def get_collection(self, name):
        if name not in self._existing:
            raise Exception(f"collection or alias '{name}' not found (simulated 404)")
        return object()

    def create_collection(self, **kwargs):
        self.create_collection_calls.append(kwargs)

    def create_payload_index(self, **kwargs):
        self.create_payload_index_calls.append(kwargs)


def _store(client, collection="knowledge_base"):
    store = object.__new__(VectorStore)
    store.client = client
    store.collection = collection
    store.timeout = 5.0
    return store


def test_create_collection_creates_when_name_does_not_exist():
    client = _FakeQdrantClient(existing_names=())
    store = _store(client)
    store.create_collection(vector_size=1024)
    assert len(client.create_collection_calls) == 1
    assert client.create_collection_calls[0]["collection_name"] == "knowledge_base"


def test_create_collection_skips_when_name_already_resolves():
    """This is the alias case: 'knowledge_base' is not itself a physical
    collection name, but get_collection() resolves it (through the alias)
    without raising — must NOT attempt to create it."""
    client = _FakeQdrantClient(existing_names=("knowledge_base",))
    store = _store(client)
    store.create_collection(vector_size=1024)
    assert client.create_collection_calls == []


def test_create_collection_always_ensures_payload_indexes_either_way():
    for existing_names in ((), ("knowledge_base",)):
        client = _FakeQdrantClient(existing_names=existing_names)
        store = _store(client)
        store.create_collection(vector_size=1024)
        assert len(client.create_payload_index_calls) > 0
