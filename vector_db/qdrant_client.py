"""
Qdrant Vector Database Client
"""
import logging
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue
)
import uuid

logger = logging.getLogger(__name__)

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
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )
            logger.info(f"Collection '{self.collection}' created")
        else:
            logger.info(f"Collection '{self.collection}' already exists")

    def upsert_chunks(self, chunks: list, vectors: list[list[float]]):
        points = []
        for chunk, vector in zip(chunks, vectors):
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
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
                }
            ))
        self.client.upsert(collection_name=self.collection, points=points)
        logger.info(f"Stored {len(points)} chunks in Qdrant")

    def search(self, query_vector: list[float], top_k: int = 5, doc_filter: str = None, folder_filter: str = None):
        must_conditions = []
        if doc_filter:
            must_conditions.append(FieldCondition(key="document_id", match=MatchValue(value=doc_filter)))
        if folder_filter:
            must_conditions.append(FieldCondition(key="folder", match=MatchValue(value=folder_filter)))
        search_filter = Filter(must=must_conditions) if must_conditions else None
        results = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            limit=top_k,
            query_filter=search_filter,
            with_payload=True
        ).points
        return [
            {
                "text": r.payload["text"],
                "page_num": r.payload["page_num"],
                "document_id": r.payload["document_id"],
                "score": r.score,
                "chunk_id": r.payload["chunk_id"]
            }
            for r in results
        ]

    def get_collection_info(self) -> dict:
        try:
            info = self.client.get_collection(self.collection)
            return {
                "total_vectors": info.points_count,
                "collection": self.collection
            }
        except Exception as e:
            return {"collection": self.collection, "error": str(e)}
