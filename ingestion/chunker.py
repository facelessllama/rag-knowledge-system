"""
Smart Text Chunker
Splits documents into semantically meaningful chunks with overlap
"""
import re
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    """A single chunk of text with metadata"""
    chunk_id: str
    text: str
    page_num: int
    chunk_index: int
    char_count: int
    has_ocr: bool
    document_id: Optional[str] = None
    filename: Optional[str] = None
    pages: int = 0
    folder: str = ""


class SmartChunker:
    """
    Splits text into overlapping chunks.
    Respects sentence and paragraph boundaries.
    
    Why not fixed-size chunking?
    Fixed size cuts sentences in half, losing semantic meaning.
    Smart chunking preserves context.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        min_chunk_size: int = 100
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        logger.info(
            f"SmartChunker ready | size={chunk_size} "
            f"overlap={chunk_overlap}"
        )

    def chunk_document(self, pages: list[dict], doc_id: str) -> list[TextChunk]:
        """Process all pages of a document into chunks"""
        all_chunks = []
        chunk_index = 0

        for page in pages:
            page_chunks = self._chunk_text(
                text=page["text"],
                page_num=page["page_num"],
                has_ocr=page["has_ocr"],
                doc_id=doc_id,
                start_index=chunk_index
            )
            all_chunks.extend(page_chunks)
            chunk_index += len(page_chunks)

        logger.info(f"Document {doc_id}: {len(all_chunks)} chunks created")
        return all_chunks

    def _chunk_text(
        self,
        text: str,
        page_num: int,
        has_ocr: bool,
        doc_id: str,
        start_index: int
    ) -> list[TextChunk]:
        """Split a single page text into chunks"""
        if not text or len(text) < self.min_chunk_size:
            return []

        # Split into sentences first
        sentences = self._split_into_sentences(text)
        chunks = []
        current_chunk = []
        current_size = 0

        for sentence in sentences:
            sentence_size = len(sentence)

            # If adding this sentence exceeds chunk_size — save current chunk
            if current_size + sentence_size > self.chunk_size and current_chunk:
                chunk_text = " ".join(current_chunk).strip()
                if len(chunk_text) >= self.min_chunk_size:
                    chunks.append(TextChunk(
                        chunk_id=f"{doc_id}_p{page_num}_c{start_index + len(chunks)}",
                        text=chunk_text,
                        page_num=page_num,
                        chunk_index=start_index + len(chunks),
                        char_count=len(chunk_text),
                        has_ocr=has_ocr,
                        document_id=doc_id
                    ))

                # Overlap: keep last N chars for context continuity
                overlap_text = chunk_text[-self.chunk_overlap:]
                current_chunk = [overlap_text, sentence]
                current_size = len(overlap_text) + sentence_size
            else:
                current_chunk.append(sentence)
                current_size += sentence_size

        # Save the last chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk).strip()
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append(TextChunk(
                    chunk_id=f"{doc_id}_p{page_num}_c{start_index + len(chunks)}",
                    text=chunk_text,
                    page_num=page_num,
                    chunk_index=start_index + len(chunks),
                    char_count=len(chunk_text),
                    has_ocr=has_ocr,
                    document_id=doc_id
                ))

        return chunks

    def _split_into_sentences(self, text: str) -> list[str]:
        """Split text into sentences respecting Russian and English"""
        # Clean text first
        text = re.sub(r'\s+', ' ', text).strip()
        # Split on sentence endings
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
