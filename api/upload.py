"""
/upload and /upload-batch endpoints, plus the ingestion helpers only they
use. Split out of api/main.py purely to shrink that file — see the
refactor plan.

Every reference to api/main.py's shared state (UPLOAD_DIR, MAX_UPLOAD_BYTES,
MAX_BATCH_FILES, file_hashes/folders_registry/documents_registry, parser/
chunker/embedder/vector_store, LANGFUSE_ENABLED/langfuse, db_save_ingestion/
db_conn, DuplicateFileHashError, PARSERS_BY_EXT) goes through a LAZY
`import api.main as m` done INSIDE each function, at the point of use —
never a top-level `from api.main import X`. Same two reasons as api/
documents.py/api/health.py/api/query.py: `api.main` imports this module to
wire up its router (a top-level `import api.main` here would be circular),
and tests do `monkeypatch.setattr(api.main, "X", fake)` against the
`api.main` module object itself — a binding captured once at import time
elsewhere would freeze at whatever value existed then.

`_ingestion_semaphore`/`_active_ingestions`/`MAX_CONCURRENT_INGESTIONS`/
`_FORMAT_MAGIC_BYTES` are the one exception: nothing outside this file (and
nothing in the test suite) ever touches them, so they're defined here
directly rather than staying in api/main.py behind a lazy lookup — genuinely
self-contained, unlike everything else in this module.

The backup-exclusion dependency (`Depends(require_not_backing_up)`) below
comes from `api/dependencies.py`, shared with api/documents.py — it has
zero dependency on api.main's state, so both modules import the same
implementation directly rather than each keeping their own copy.
"""
import asyncio
import contextlib
import hashlib
import os
import time
import uuid
from pathlib import Path
from typing import Optional

import aiofiles
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from qdrant_client.models import FieldCondition, Filter, MatchValue

from api.dependencies import require_not_backing_up
from ingestion.chunker import chunk_context_text
from rag.executors import run_on_gpu

import logging

logger = logging.getLogger(__name__)

router = APIRouter()

# Upload/ingestion limits. MAX_CONCURRENT_INGESTIONS only needs to be small
# — embedding is already serialized onto one GPU worker by run_on_gpu
# (rag/executors.py), but PDF parsing/OCR is CPU-bound and runs via plain
# asyncio.to_thread, so nothing else limits how many of those could run in
# parallel without this.
MAX_CONCURRENT_INGESTIONS = int(os.getenv("MAX_CONCURRENT_INGESTIONS", "2"))
_ingestion_semaphore = asyncio.Semaphore(MAX_CONCURRENT_INGESTIONS)
_active_ingestions = 0  # only ever mutated between an `await` boundary and the next, see _ingestion_slot() — safe without a separate lock under asyncio's single-threaded cooperative scheduling

# Every supported format needs an entry here for the streaming upload
# helper (below) to reject an obvious mismatch (a renamed .exe as
# "report.pdf", say) before it ever reaches PyMuPDF/the parser — checked
# against just the first chunk read off the wire, not the whole file.
# TXT has no fixed signature, so any content passes for it; ".txt" is a
# labeling convention, not a format PyMuPDF or anything else parses
# structurally, so there is nothing meaningful to check.
_FORMAT_MAGIC_BYTES = {"pdf": b"%PDF-"}


@contextlib.asynccontextmanager
async def _ingestion_slot(doc_id: str):
    """Wraps _ingestion_semaphore with observability: logs how many
    ingestion jobs are actually concurrently in this block right now, not
    just that a semaphore object exists — the thing worth being able to
    verify from logs under real concurrent load, not just trust from
    reading the semaphore size."""
    global _active_ingestions
    async with _ingestion_semaphore:
        _active_ingestions += 1
        logger.info(f"Ingestion slot acquired for {doc_id} ({_active_ingestions}/{MAX_CONCURRENT_INGESTIONS} in use)")
        try:
            yield
        finally:
            _active_ingestions -= 1
            logger.info(f"Ingestion slot released for {doc_id} ({_active_ingestions}/{MAX_CONCURRENT_INGESTIONS} in use)")


def _validate_signature(doc_format: str, header: bytes) -> bool:
    magic = _FORMAT_MAGIC_BYTES.get(doc_format)
    return magic is None or header.startswith(magic)


async def _stream_upload_to_disk(file: UploadFile, dest: Path, doc_format: str, max_bytes: int) -> str:
    """Streams the upload into a per-attempt temp file in fixed-size chunks,
    hashing incrementally — never holds more than one chunk in memory
    regardless of the file's actual size, unlike the previous
    `content = await file.read()` (which bought a full copy into RAM before
    anything else even looked at it). Only `os.replace()`s the temp file
    onto `dest` (atomic on the same filesystem — same directory guarantees
    that) once the whole upload has validated clean; any failure path only
    ever unlinks the temp file, never `dest` itself. Two things this
    protects against that writing straight to `dest` wouldn't:
      - `dest` never exists in a partially-written state for anything else
        (a directory listing, a concurrent request) to observe.
      - cleanup on failure can never delete a file that isn't this
        attempt's own — the temp name is unique per call, so there is no
        path under which "roll back this failed upload" could touch
        something that existed before it started.

    Dedup (`file_hash in file_hashes`) has to happen AFTER this — the full
    hash needs the full file, and there's no way to know it without reading
    everything, which is exactly the read this function makes safe to do."""
    tmp_path = dest.parent / f".upload_{uuid.uuid4().hex}.tmp"
    hasher = hashlib.md5()
    total = 0
    chunk_size = 1024 * 1024
    first_chunk = True
    error: Optional[HTTPException] = None

    try:
        async with aiofiles.open(tmp_path, "wb") as f:
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                if first_chunk:
                    first_chunk = False
                    if not _validate_signature(doc_format, chunk):
                        error = HTTPException(400, f"File content doesn't look like a {doc_format.upper()} file")
                        break
                total += len(chunk)
                if total > max_bytes:
                    error = HTTPException(413, f"File exceeds the {max_bytes // (1024 * 1024)}MB upload limit")
                    break
                hasher.update(chunk)
                await f.write(chunk)

        if error is None and total == 0:
            error = HTTPException(422, "Uploaded file is empty")
        if error is not None:
            raise error

        os.replace(tmp_path, dest)  # atomic — dest either doesn't exist yet or is fully this upload's content, never a partial write
    except BaseException:
        # BaseException, not Exception — a client disconnect while this
        # request is awaiting `file.read()` surfaces as asyncio.CancelledError,
        # which does NOT subclass Exception (since Python 3.8). An `except
        # Exception:` here would silently leak the temp file on every
        # cancelled upload; the bare `raise` below still propagates
        # cancellation (and anything else) to the caller unchanged —
        # cleanup runs, nothing is swallowed.
        tmp_path.unlink(missing_ok=True)
        raise
    return hasher.hexdigest()


def pick_parser(filename: str):
    """Returns (parser_instance, format) for a supported extension, or (None, None)."""
    import api.main as m

    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    return m.PARSERS_BY_EXT.get(ext, (None, None))


def validate_folder(folder: str) -> str:
    """`folder` is a raw client-controlled Form field, unlike PDF/TXT text
    content (which gets NUL-stripped at extraction time — see
    ingestion/pdf_parser.py's parse()/_extract_metadata() and chunker.py's
    normalize_whitespace()). A NUL byte here hits the exact same Postgres
    "string literal cannot contain NUL" rejection (db_save_ingestion's
    `documents`/`folders` INSERTs), but as unvalidated input rather than a
    parsing artifact it should fail fast with a clear 422, not surface as
    an opaque database error several steps later."""
    if "\x00" in folder:
        raise HTTPException(422, "folder name cannot contain a NUL character")
    return folder


async def _rollback_upload(doc_id: str, file_path: Path):
    """Best-effort compensating cleanup for an ingestion that failed
    somewhere along parse -> chunk -> embed -> upsert -> db_save. Called
    from every one of those failure paths, so it can't assume how far a
    given attempt actually got — Qdrant points, Postgres rows, and the file
    might each be present or absent in any combination. All three are
    therefore attempted unconditionally and independently (a DELETE/
    unlink for something that was never written is just a no-op), each in
    its own try/except logged separately.

    None of the three re-raises: this runs from inside the caller's
    `except` block for the REAL failure, and letting a cleanup error
    propagate from here would replace that original exception with a
    confusing one about cleanup instead — the caller's own logging/
    HTTPException already captured what actually went wrong before calling
    this."""
    import api.main as m

    try:
        await asyncio.to_thread(
            m.vector_store.client.delete,
            collection_name=m.vector_store.collection,
            points_selector=Filter(must=[FieldCondition(key="document_id", match=MatchValue(value=doc_id))]),
        )
    except Exception as e:
        logger.error(f"Rollback: Qdrant cleanup failed for {doc_id}: {e}")

    try:
        def _delete_doc_rows():
            with m.db_conn() as conn:
                cur = conn.cursor()
                cur.execute("DELETE FROM file_hashes WHERE doc_id = %s", (doc_id,))
                # document_pages cascades via its FK's ON DELETE CASCADE (init_db()).
                cur.execute("DELETE FROM documents WHERE doc_id = %s", (doc_id,))
        await asyncio.to_thread(_delete_doc_rows)
    except Exception as e:
        logger.error(f"Rollback: Postgres cleanup failed for {doc_id}: {e}")

    try:
        file_path.unlink(missing_ok=True)
    except Exception as e:
        logger.error(f"Rollback: file cleanup failed for {doc_id} ({file_path}): {e}")


@router.post("/upload", dependencies=[Depends(require_not_backing_up)])
async def upload_document(file: UploadFile = File(...), folder: str = Form("")):
    import api.main as m

    folder = validate_folder(folder)
    safe_filename = Path(file.filename).name  # strip any path components
    doc_parser, doc_format = pick_parser(safe_filename)
    if doc_parser is None:
        raise HTTPException(400, "Only PDF and TXT files are supported")

    doc_id = str(uuid.uuid4())
    file_path = m.UPLOAD_DIR / f"{doc_id}_{safe_filename}"

    # Streams straight to disk (see _stream_upload_to_disk) instead of
    # buffering the whole upload in RAM first — dedup still needs the full
    # hash, so it's checked here, after the file is already safely on disk.
    file_hash = await _stream_upload_to_disk(file, file_path, doc_format, m.MAX_UPLOAD_BYTES)

    if file_hash in m.file_hashes:
        file_path.unlink(missing_ok=True)
        existing_id = m.file_hashes[file_hash]
        existing = m.documents_registry.get(existing_id, {})
        raise HTTPException(409, f"File already uploaded as '{existing.get('filename', existing_id)}' (id: {existing_id})")

    # Everything from here through the Qdrant upsert is one unit: any
    # failure — parse, chunk, embed, or upsert — must leave neither an
    # orphaned file nor orphaned Qdrant points behind, so it's all one
    # try/except around _rollback_upload rather than the empty-chunks case
    # having its own bespoke cleanup and everything else having none.
    # The semaphore bounds concurrent CPU-bound parse/OCR + GPU-bound embed
    # work — embedding is already serialized onto one GPU worker by
    # run_on_gpu, but nothing previously capped how many uploads could be
    # parsing/OCR-ing in parallel via plain asyncio.to_thread.
    try:
        async with _ingestion_slot(doc_id):
            t_parse = time.time()
            parsed = await asyncio.to_thread(doc_parser.parse, str(file_path))
            parse_ms = int((time.time() - t_parse) * 1000)
            chunks = m.chunker.chunk_document(parsed.pages, doc_id)

            if not chunks:
                raise ValueError("Could not extract text from document")

            for c in chunks:
                c.filename = safe_filename
                c.pages = parsed.total_pages
                c.folder = folder or ""

            texts = [chunk_context_text(c) for c in chunks]
            t_embed = time.time()
            vectors = await run_on_gpu(m.embedder.embed_batch, texts)
            embed_ms = int((time.time() - t_embed) * 1000)
            await asyncio.to_thread(m.vector_store.upsert_chunks, chunks, vectors)
    except asyncio.CancelledError:
        # CancelledError does NOT subclass Exception (since Python 3.8) —
        # a client disconnect or server shutdown cancelling this task while
        # it's mid-parse/embed/upsert would otherwise skip both except
        # clauses below entirely and leave the file/Qdrant points orphaned.
        # Must re-raise bare (not as an HTTPException) — swallowing
        # cancellation here would make the task look like it completed
        # normally instead of actually being cancelled.
        logger.warning(f"Ingestion cancelled for {doc_id} ({safe_filename}) — rolling back")
        await _rollback_upload(doc_id, file_path)
        raise
    except ValueError as e:
        await _rollback_upload(doc_id, file_path)
        raise HTTPException(422, str(e))
    except Exception as e:
        logger.error(f"Ingestion failed for {doc_id} ({safe_filename}): {e}")
        await _rollback_upload(doc_id, file_path)
        raise HTTPException(500, f"Failed to process document: {e}")

    ocr_pages = sum(1 for p in parsed.pages if p.get("has_ocr"))
    logger.info(
        f"Ingestion: {safe_filename} | pages={parsed.total_pages} ocr={ocr_pages} "
        f"chunks={len(chunks)} parse_ms={parse_ms} embed_ms={embed_ms}"
    )
    if m.LANGFUSE_ENABLED:
        try:
            m.langfuse.trace(name="doc_ingestion", input=safe_filename, tags=["upload"],
                           metadata={"doc_id": doc_id, "pages": parsed.total_pages,
                                     "ocr_pages": ocr_pages, "chunks": len(chunks),
                                     "size_kb": parsed.file_size_kb, "folder": folder or "",
                                     "parse_ms": parse_ms, "embed_ms": embed_ms})
            m.langfuse.flush()
        except Exception:
            pass

    doc_meta = {
        "doc_id": doc_id,
        "filename": safe_filename,
        "pages": parsed.total_pages,
        "chunks": len(chunks),
        "size_kb": parsed.file_size_kb,
        "metadata": parsed.metadata,
        "folder": folder or "",
        "format": doc_format,
    }
    # file_hashes/folders_registry/documents_registry are only updated in
    # memory AFTER db_save_ingestion() commits — updating them earlier (the
    # old code set file_hashes[file_hash] before this could even fail) left
    # a hash entry with nothing behind it if the save failed: every retry
    # of the same file then hit the file_hash-seen check below and got a
    # false 409, even though the document/file/Qdrant points were all gone.
    try:
        await asyncio.to_thread(m.db_save_ingestion, doc_meta, parsed.pages, file_hash)
    except m.DuplicateFileHashError:
        logger.warning(f"Concurrent upload race for {doc_id} (file_hash already claimed) — discarding this attempt")
        await _rollback_upload(doc_id, file_path)
        existing_id = m.file_hashes.get(file_hash)
        existing = m.documents_registry.get(existing_id, {}) if existing_id else {}
        raise HTTPException(409, f"File already uploaded as '{existing.get('filename', existing_id or file_hash)}'")
    except Exception as e:
        logger.error(f"DB save failed for {doc_id}, rolling back Qdrant points + uploaded file: {e}")
        await _rollback_upload(doc_id, file_path)
        raise HTTPException(500, f"Failed to save document — upload did not complete: {e}")
    m.file_hashes[file_hash] = doc_id
    if folder:
        m.folders_registry.add(folder)
    m.documents_registry[doc_id] = doc_meta

    return {
        "doc_id": doc_id,
        "filename": safe_filename,
        "pages": parsed.total_pages,
        "chunks_created": len(chunks),
        "status": "indexed"
    }


@router.post("/upload-batch", dependencies=[Depends(require_not_backing_up)])
async def upload_batch(files: list[UploadFile] = File(...), folder: str = Form("")):
    import api.main as m

    folder = validate_folder(folder)
    if len(files) > m.MAX_BATCH_FILES:
        raise HTTPException(413, f"Batch exceeds the {m.MAX_BATCH_FILES}-file limit ({len(files)} files sent)")

    results = []
    for file in files:
        doc_id = None   # reset each iteration — a stale doc_id/file_path from
        file_path = None  # a previous file must never be used for this file's cleanup
        try:
            safe_name = Path(file.filename).name
            doc_parser, doc_format = pick_parser(safe_name)
            if doc_parser is None:
                results.append({"filename": safe_name, "status": "error", "error": "Only PDF and TXT files are supported"})
                continue

            doc_id = str(uuid.uuid4())
            file_path = m.UPLOAD_DIR / f"{doc_id}_{safe_name}"

            # Streams straight to disk (see _stream_upload_to_disk) instead
            # of buffering the whole upload in RAM — a size/signature
            # failure here already cleaned up its own file, so it's reported
            # per-file (not a whole-batch abort) without any rollback call.
            try:
                file_hash = await _stream_upload_to_disk(file, file_path, doc_format, m.MAX_UPLOAD_BYTES)
            except HTTPException as e:
                results.append({"filename": safe_name, "status": "error", "error": e.detail})
                continue

            if file_hash in m.file_hashes:
                file_path.unlink(missing_ok=True)
                existing_id = m.file_hashes[file_hash]
                existing = m.documents_registry.get(existing_id, {})
                results.append({"filename": safe_name, "status": "skipped", "error": f"Already uploaded as '{existing.get('filename', existing_id)}'"})
                continue

            # Bounds concurrent CPU-bound parse/OCR + GPU-bound embed work
            # across all in-flight uploads (single + batch share the same
            # semaphore) — see MAX_CONCURRENT_INGESTIONS above.
            async with _ingestion_slot(doc_id):
                parsed = await asyncio.to_thread(doc_parser.parse, str(file_path))
                chunks = m.chunker.chunk_document(parsed.pages, doc_id)

                if not chunks:
                    raise ValueError("Could not extract text")

                for c in chunks:
                    c.filename = safe_name
                    c.pages = parsed.total_pages
                    c.folder = folder or ""

                texts = [chunk_context_text(c) for c in chunks]
                vectors = await run_on_gpu(m.embedder.embed_batch, texts)
                await asyncio.to_thread(m.vector_store.upsert_chunks, chunks, vectors)

            doc_meta = {
                "doc_id": doc_id,
                "filename": safe_name,
                "pages": parsed.total_pages,
                "chunks": len(chunks),
                "size_kb": parsed.file_size_kb,
                "metadata": parsed.metadata,
                "folder": folder or "",
                "format": doc_format,
            }
            # file_hashes/folders_registry/documents_registry only updated in
            # memory after a successful commit — see db_save_ingestion().
            await asyncio.to_thread(m.db_save_ingestion, doc_meta, parsed.pages, file_hash)
            m.file_hashes[file_hash] = doc_id
            if folder:
                m.folders_registry.add(folder)
            m.documents_registry[doc_id] = doc_meta
            results.append({"doc_id": doc_id, "filename": safe_name, "status": "indexed",
                            "pages": parsed.total_pages, "chunks_created": len(chunks)})

        except m.DuplicateFileHashError:
            logger.warning(f"Concurrent upload race for {doc_id} (file_hash already claimed) — discarding this attempt")
            if doc_id and file_path:
                await _rollback_upload(doc_id, file_path)
            existing_id = m.file_hashes.get(file_hash)
            existing = m.documents_registry.get(existing_id, {}) if existing_id else {}
            results.append({"filename": safe_name, "status": "skipped",
                            "error": f"Already uploaded as '{existing.get('filename', existing_id or file_hash)}'"})

        except asyncio.CancelledError:
            # Does NOT subclass Exception (Python 3.8+) — must stay its own
            # clause or a disconnect mid-file would fall through to the
            # generic handler below, get reported as a normal per-file
            # "error" result, and let the loop carry on to the next file
            # instead of actually stopping. Rolls back this file only, then
            # re-raises to abort the whole batch — the caller is gone, there
            # is no results list left to return it to.
            logger.warning(f"Batch ingestion cancelled for {file.filename} — rolling back and aborting batch")
            if doc_id and file_path:
                await _rollback_upload(doc_id, file_path)
            raise

        except Exception as e:
            logger.error(f"Batch error {file.filename}: {e}")
            if doc_id and file_path:
                await _rollback_upload(doc_id, file_path)
            results.append({"filename": file.filename, "status": "error", "error": str(e)})

    indexed = sum(1 for r in results if r["status"] == "indexed")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    errors = sum(1 for r in results if r["status"] == "error")
    return {"total": len(results), "indexed": indexed, "skipped": skipped, "errors": errors, "results": results}
