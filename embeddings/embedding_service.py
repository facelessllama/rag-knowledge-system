"""
Embedding Service
Converts text to vectors using BGE-M3 model (multilingual, runs locally)
"""
import logging
from sentence_transformers import SentenceTransformer
import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Wraps sentence-transformers for text vectorization.
    
    Why BGE-M3?
    - Best multilingual model (Russian + English)
    - Runs fully offline on GPU
    - 1024-dimensional vectors = high quality search
    """

    def __init__(self, model_name: str = "BAAI/bge-m3"):
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.vector_size = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model ready | vector size: {self.vector_size}")

    def embed_text(self, text: str) -> list[float]:
        """Embed a single text string"""
        vector = self.model.encode(text, normalize_embeddings=True)
        return vector.tolist()

    def embed_batch(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """
        Embed multiple texts efficiently.
        Batch processing is much faster than one-by-one.
        """
        logger.info(f"Embedding {len(texts)} texts in batches of {batch_size}")
        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True
        )
        return vectors.tolist()

    def get_vector_size(self) -> int:
        return self.vector_size
