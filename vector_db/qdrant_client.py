"""
Qdrant Vector Database Client

Hybrid search (dense + BM25-style sparse) runs entirely server-side:
Qdrant maintains sparse-vector IDF statistics incrementally as points are
added/removed (Modifier.IDF), and RRF fusion of dense+sparse candidates
happens in a single query_points call. No client-side keyword index, no
full-corpus rebuild on upload or restart.
"""
import logging
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue, MatchAny,
    SparseVectorParams, Modifier, Prefetch, FusionQuery, Fusion,
    PayloadSchemaType,
)

from vector_db.sparse_encoder import build_sparse_vector

logger = logging.getLogger(__name__)

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "bm25"

_PAYLOAD_INDEXES = (
    ("document_id", PayloadSchemaType.KEYWORD),
    ("folder", PayloadSchemaType.KEYWORD),
    ("chunk_index", PayloadSchemaType.INTEGER),
)


class VectorStore:
    def __init__(self, url: str = "http://localhost:6333", collection: str = "knowledge_base"):
        self.client = QdrantClient(url=url)
        self.collection = collection
        logger.info(f"Connected to Qdrant | collection: {collection}")

    def create_collection(self, vector_size: int = 1024):
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection not in existing:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config={DENSE_VECTOR_NAME: VectorParams(size=vector_size, distance=Distance.COSINE)},
                sparse_vectors_config={SPARSE_VECTOR_NAME: SparseVectorParams(modifier=Modifier.IDF)},
            )
            logger.info(f"Collection '{self.collection}' created (dense+sparse hybrid)")
        else:
            logger.info(f"Collection '{self.collection}' already exists")
        self._ensure_payload_indexes()

    def _ensure_payload_indexes(self):
        for field, schema in _PAYLOAD_INDEXES:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection, field_name=field, field_schema=schema
                )
            except Exception as e:
                logger.debug(f"Payload index '{field}' already present or failed to create: {e}")

    def upsert_chunks(self, chunks: list, vectors: list[list[float]]):
        points = []
        for chunk, vector in zip(chunks, vectors):
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    DENSE_VECTOR_NAME: vector,
                    SPARSE_VECTOR_NAME: build_sparse_vector(chunk.text),
                },
                payload={
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "page_num": chunk.page_num,
                    "document_id": chunk.document_id,
                    "has_ocr": chunk.has_ocr,
                    "char_count": chunk.char_count,
                    "filename": getattr(chunk, "filename", "unknown"),
                    "chunk_index": getattr(chunk, "chunk_index", 0),
                    "pages": getattr(chunk, "pages", 0),
                    "folder": getattr(chunk, "folder", ""),
                    "char_start": getattr(chunk, "char_start", None),
                    "char_end": getattr(chunk, "char_end", None),
                }
            ))
        self.client.upsert(collection_name=self.collection, points=points)
        logger.info(f"Stored {len(points)} chunks in Qdrant (dense+sparse)")

    def hybrid_search(
        self,
        query_vector: list[float],
        query_text: str,
        top_k: int = 5,
        doc_filter: str = None,
        folder_filter: str = None,
    ) -> list[dict]:
        """Single server-side call: dense + sparse candidates fused via RRF."""
        search_filter = self._build_filter(doc_filter, folder_filter)

        sparse_vector = build_sparse_vector(query_text)
        prefetch = [Prefetch(query=query_vector, using=DENSE_VECTOR_NAME, limit=top_k, filter=search_filter)]
        if sparse_vector.indices:
            prefetch.append(Prefetch(query=sparse_vector, using=SPARSE_VECTOR_NAME, limit=top_k, filter=search_filter))

        results = self.client.query_points(
            collection_name=self.collection,
            prefetch=prefetch,
            query=FusionQuery(fusion=Fusion.RRF),
            query_filter=search_filter,
            limit=top_k,
            with_payload=True,
        ).points
        return self._to_result_dicts(results)

    def search(self, query_vector: list[float], top_k: int = 5, doc_filter: str = None, folder_filter: str = None) -> list[dict]:
        """Dense-only search — kept for callers without query text (e.g. pure similarity lookups)."""
        search_filter = self._build_filter(doc_filter, folder_filter)
        results = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            using=DENSE_VECTOR_NAME,
            limit=top_k,
            query_filter=search_filter,
            with_payload=True,
        ).points
        return self._to_result_dicts(results)

    def neighbor_chunks(self, document_id: str, chunk_indices: list[int]) -> list[dict]:
        """Fetch specific chunks of a document by index — used to expand top hits with adjacent context."""
        if not chunk_indices:
            return []
        search_filter = Filter(must=[
            FieldCondition(key="document_id", match=MatchValue(value=document_id)),
            FieldCondition(key="chunk_index", match=MatchAny(any=chunk_indices)),
        ])
        results, _ = self.client.scroll(
            collection_name=self.collection,
            scroll_filter=search_filter,
            limit=max(len(chunk_indices) * 2, 10),
            with_payload=True,
            with_vectors=False,
        )
        out = []
        for r in results:
            p = r.payload or {}
            if not p.get("text") or not p.get("document_id"):
                continue
            out.append({
                "text": p["text"],
                "page_num": p.get("page_num"),
                "document_id": p["document_id"],
                "chunk_id": p.get("chunk_id"),
                "filename": p.get("filename", ""),
                "folder": p.get("folder", ""),
                "chunk_index": p.get("chunk_index", 0),
                "char_start": p.get("char_start"),
                "char_end": p.get("char_end"),
            })
        return out

    def _build_filter(self, doc_filter: str = None, folder_filter: str = None) -> Filter | None:
        must_conditions = []
        if doc_filter:
            must_conditions.append(FieldCondition(key="document_id", match=MatchValue(value=doc_filter)))
        if folder_filter:
            must_conditions.append(FieldCondition(key="folder", match=MatchValue(value=folder_filter)))
        return Filter(must=must_conditions) if must_conditions else None

    def _to_result_dicts(self, results) -> list[dict]:
        out = []
        for r in results:
            p = r.payload or {}
            if not p.get("text") or not p.get("document_id"):
                logger.warning(f"Skipping malformed Qdrant result id={r.id}: missing required fields")
                continue
            out.append({
                "text": p["text"],
                "page_num": p.get("page_num"),
                "document_id": p["document_id"],
                "score": r.score,
                "chunk_id": p.get("chunk_id"),
                "filename": p.get("filename", ""),
                "folder": p.get("folder", ""),
                "chunk_index": p.get("chunk_index", 0),
                "char_start": p.get("char_start"),
                "char_end": p.get("char_end"),
            })
        return out

    def get_collection_info(self) -> dict:
        try:
            info = self.client.get_collection(self.collection)
            return {
                "total_vectors": info.points_count,
                "collection": self.collection
            }
        except Exception as e:
            return {"collection": self.collection, "error": str(e)}
