"""
Smart Text Chunker
Splits documents into semantically meaningful chunks with overlap
"""
import re
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def normalize_whitespace(text: str) -> str:
    """Collapse all whitespace runs to single spaces. Chunk char_start/char_end
    are offsets into this normalized form — callers that need to slice back
    into a chunk's source text (e.g. a TXT-document viewer) must normalize
    the same way first, or offsets won't line up."""
    return re.sub(r'\s+', ' ', text).strip()


def chunk_context_text(chunk) -> str:
    """Prepends a short document-level context (the descriptive filename,
    underscores -> spaces) to a chunk's text — 'contextual retrieval'
    (Anthropic). Many chunks, especially short heavily-templated legal
    boilerplate ("Heard learned counsel for the petitioner...", near-
    identical across many different case documents), carry almost no
    signal on their own distinguishing which document they came from. Use
    this for what gets embedded/indexed/reranked, never for what gets
    stored/shown — chunk.text itself stays untouched so char_start/char_end
    offsets and displayed excerpts remain exact.

    Accepts both TextChunk-like objects (ingestion) and the plain dicts
    retrieval/reranking pass around (Qdrant hits carry "filename"/"text"
    keys, not attributes)."""
    if isinstance(chunk, dict):
        filename = chunk.get("filename", "") or ""
        text = chunk["text"]
    else:
        filename = getattr(chunk, "filename", "") or ""
        text = chunk.text
    title = Path(filename).stem.replace("_", " ") if filename else ""
    return f"{title}: {text}" if title else text


# This corpus's court-case PDFs follow "<TYPE>_<NUM>_<YEAR>_<PARTY1>_vs_
# <PARTY2>.pdf" (see generate_legal_docs.py) — e.g. "ABLAPL_3648_2020_
# LIPU_PRADHAN_vs_STATE_OF_ODISHA.pdf". Same convention eval/
# build_golden_dataset.py parses to generate case_lookup_by_number/
# case_summary questions. Type prefix itself may contain underscores (e.g.
# "CR._MISC."), so match it non-greedily up to the first "_<digits>_
# <4-digit year>_" run instead of constraining its character set.
_CASE_FILENAME_RE = re.compile(r'^(.+?)_(\d+)_(\d{4})_(.+?)_vs_(.+)$')


def extract_case_metadata(filename: str) -> dict:
    """Parses a case-file's structured identity (case number, year, party
    names) from its filename at ingestion time, for storage on every chunk
    of the document (see vector_db/qdrant_client.py). This lets query-time
    matching (rag/retriever.py) compare against a stable per-document
    identity instead of re-deriving it by regex-scanning each candidate
    chunk's own text — which misses whenever the number/parties don't
    happen to land in that particular chunk (a chunk deep in a ruling's
    body won't always repeat the caption). Returns {} for filenames that
    don't match this convention (e.g. the plain-text UK statutory
    instruments, which have no case number)."""
    m = _CASE_FILENAME_RE.match(Path(filename).stem)
    if not m:
        return {}
    _case_type, num, year, p1, p2 = m.groups()
    return {
        "case_number": num,
        "case_year": year,
        "parties": [p1.replace("_", " "), p2.replace("_", " ")],
    }


# EU legislation (EURLEX57K-style corpus) filenames are "<CELEX_ID> -
# <description>.pdf" — e.g. "31958D1127(01) - EEC Council Rules of the
# Transport Committee.pdf". A CELEX ID is sector digit(1) + year(4) + type
# letter(1) + number(4), optional "(NN)" suffix — e.g. "31997R0955",
# "32011D0126". Unlike the court-case corpus's case number, the CELEX ID
# itself never appears in the document's own body text (documents cite
# themselves as "1997/955/EC"-style, a different format entirely) — see
# rag/retriever.py's use of this for why that makes it retrieval-hard
# without structured metadata.
_CELEX_ID_RE = re.compile(r'^(\d{5}[A-Z]\d{4}(?:\(\d+\))?)\s*-\s*')


def extract_celex_id(filename: str) -> str | None:
    """Parses the CELEX ID prefix from an EU-legislation filename at
    ingestion time, for storage on every chunk of the document (see
    vector_db/qdrant_client.py) — same role as extract_case_metadata()
    plays for the court-case corpus. Returns None for filenames that don't
    match this convention."""
    m = _CELEX_ID_RE.match(Path(filename).stem)
    return m.group(1) if m else None


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
    # Offsets into normalize_whitespace(page_text) — page-relative, not
    # document-relative. Populated for every chunk; only actually consumed
    # by the TXT viewer today (PDF highlighting still uses PyMuPDF search_for).
    char_start: Optional[int] = None
    char_end: Optional[int] = None


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

        # Normalize once; all offsets below are relative to this string.
        normalized = normalize_whitespace(text)
        sentences = self._split_sentences_with_offsets(normalized)
        chunks = []
        current: list[tuple[int, int, str]] = []  # (start, end, text) triples
        current_size = 0

        def emit(pieces: list[tuple[int, int, str]]) -> None:
            chunk_start = pieces[0][0]
            chunk_end = pieces[-1][1]
            chunk_text = normalized[chunk_start:chunk_end]
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append(TextChunk(
                    chunk_id=f"{doc_id}_p{page_num}_c{start_index + len(chunks)}",
                    text=chunk_text,
                    page_num=page_num,
                    chunk_index=start_index + len(chunks),
                    char_count=len(chunk_text),
                    has_ocr=has_ocr,
                    document_id=doc_id,
                    char_start=chunk_start,
                    char_end=chunk_end,
                ))

        for sent_start, sent_end, sentence in sentences:
            sentence_size = sent_end - sent_start

            # If adding this sentence exceeds chunk_size — save current chunk
            if current_size + sentence_size > self.chunk_size and current:
                emit(current)

                # Overlap: keep last N chars for context continuity — this is
                # a contiguous slice of `normalized` ending exactly where the
                # emitted chunk ended, so it composes cleanly with the next
                # sentence (no gap, no re-search needed).
                prev_end = current[-1][1]
                overlap_start = max(current[0][0], prev_end - self.chunk_overlap)
                current = [(overlap_start, prev_end, normalized[overlap_start:prev_end]),
                           (sent_start, sent_end, sentence)]
                current_size = (prev_end - overlap_start) + sentence_size
            else:
                current.append((sent_start, sent_end, sentence))
                current_size += sentence_size

        # Save the last chunk
        if current:
            emit(current)

        return chunks

    def _split_sentences_with_offsets(self, normalized: str) -> list[tuple[int, int, str]]:
        """Split already-normalized text into (start, end, sentence) triples,
        respecting Russian and English sentence endings."""
        if not normalized:
            return []
        spans = []
        pos = 0
        for m in re.finditer(r'(?<=[.!?])\s+', normalized):
            sep_start, sep_end = m.span()
            if sep_start > pos:
                spans.append((pos, sep_start, normalized[pos:sep_start]))
            pos = sep_end
        if pos < len(normalized):
            spans.append((pos, len(normalized), normalized[pos:]))
        return self._hard_wrap_oversized_spans(normalized, spans)

    def _hard_wrap_oversized_spans(
        self, normalized: str, spans: list[tuple[int, int, str]]
    ) -> list[tuple[int, int, str]]:
        """Fallback for run-on prose with no '.'/'!'/'?' for thousands of chars
        (semicolon-separated clause lists are routine in Russian statutes) —
        without this, such a stretch is one atomic "sentence" span many times
        chunk_size, emitted as a single oversized chunk downstream. That's both
        far slower to embed (attention cost grows faster than linearly with
        sequence length) and semantically diluted for retrieval. Any span over
        2x chunk_size gets hard-wrapped at whitespace boundaries instead."""
        limit = self.chunk_size * 2
        result = []
        for start, end, text in spans:
            if end - start <= limit:
                result.append((start, end, text))
                continue
            piece_start = start
            while piece_start < end:
                piece_end = min(piece_start + self.chunk_size, end)
                if piece_end < end:
                    ws = normalized.rfind(' ', piece_start, piece_end)
                    if ws > piece_start:
                        piece_end = ws
                result.append((piece_start, piece_end, normalized[piece_start:piece_end]))
                piece_start = piece_end
                while piece_start < end and normalized[piece_start] == ' ':
                    piece_start += 1
        return result
