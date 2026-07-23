"""
Document/folder management endpoints, plus the shared "active document"
helpers query.py also uses. Split out of api/main.py purely to shrink that
file — see the refactor plan.

Every reference to api/main.py's shared state (documents_registry,
folders_registry, the db_* functions, _metadata_mutation_lock,
_reconcile_document_folder_sync/_reconcile_document_deletion, UPLOAD_DIR,
API_KEY, db_conn) goes through a LAZY `import api.main as m` done INSIDE
each function, at the point of use — never a top-level `from api.main
import X`. Two independent reasons: `api.main` imports this module to wire
up its routers (a top-level `import api.main` here would be circular), and
tests do `monkeypatch.setattr(api.main, "X", fake)` / `setattr(api.main,
"documents_registry", {...})` against the `api.main` module object itself —
a binding captured once at import time elsewhere would freeze at whatever
value existed then and silently stop seeing the patch.

`_get_active_document`/`_validate_query_document_scope`/`_active_chunks`/
`_resolve_scope_document_ids` are imported directly by name from
api/query.py (safe — nothing monkeypatches these four by their own
name/identity) but their bodies still follow the same lazy-lookup rule for
`documents_registry`.

Two routers: `protected_router` for everything gated by the API key
(`/documents`, `/folders/*`, `DELETE /documents/{id}`, the page-text
endpoint); `public_router` for `/pdf/{doc_id}` alone, which does its own
manual header/query-key check instead of `Depends(require_api_key)` (PDF.js
needs URL-based access) — keeping it on a separate router means mounting
`protected_router` under the API-key dependency can never accidentally
change `/pdf/{doc_id}`'s auth, and vice versa.

The backup-exclusion dependency (`Depends(require_not_backing_up)`) on
every mutating route below comes from `api/dependencies.py`, shared with
api/upload.py — it has zero dependency on api.main's state, so both
modules import the same implementation directly rather than each keeping
their own copy.
"""
import asyncio
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException

from api.dependencies import require_not_backing_up
from api.schemas import QueryRequest
from ingestion.chunker import normalize_whitespace
from ingestion.txt_parser import decode_text_file

protected_router = APIRouter()
public_router = APIRouter()


def _get_active_document(doc_id: str) -> Optional[dict]:
    """Return a document only while it is logically present."""
    import api.main as m

    doc = m.documents_registry.get(doc_id)
    if doc is None or doc.get("status", "active") != "active":
        return None
    return doc


def _validate_query_document_scope(request: QueryRequest):
    """Reject explicit references to missing/tombstoned documents."""
    requested = []
    if request.document_id:
        requested.append(request.document_id)
    if request.document_ids:
        requested.extend(request.document_ids)
    missing = list(dict.fromkeys(doc_id for doc_id in requested if _get_active_document(doc_id) is None))
    if missing:
        raise HTTPException(404, f"Document(s) not found: {', '.join(missing)}")


def _resolve_scope_document_ids(request: QueryRequest) -> Optional[list[str]]:
    """Canonical retrieval-scope list for a request — the ONE value
    api/query.py's endpoints thread into _augment_compare_queries(),
    retrieve_expanded(), promote_missing_compare_documents(), and the
    relevance-threshold bypass, instead of each of those (across both
    /query and /query/stream) reading request.document_ids directly.

    `document_id` used to be accepted and 404-validated (see
    _validate_query_document_scope above) but never actually reached
    retrieval — only request.document_ids did — so a caller scoping via
    the singular field alone got a silent, unscoped, whole-knowledge-base
    answer instead of an error. Merging the two here, once, means every
    current and future reader of "the scope" only has one field to read
    and cannot independently forget to normalize.

    A document_id that duplicates an entry already in document_ids is
    accepted as redundant. A document_id naming a document NOT in
    document_ids is rejected outright (422) rather than silently unioned
    in — the two fields disagreeing about scope is a malformed request,
    not something to guess through by expanding it.
    """
    if request.document_id and request.document_ids:
        if request.document_id not in request.document_ids:
            raise HTTPException(
                422,
                f"document_id {request.document_id!r} conflicts with document_ids "
                f"{request.document_ids!r} — pass a consistent scope, not both.",
            )
        return request.document_ids
    if request.document_ids:
        return request.document_ids
    if request.document_id:
        return [request.document_id]
    return None


def _active_chunks(chunks: list[dict]) -> list[dict]:
    """Drop stale Qdrant points belonging to logical tombstones."""
    return [chunk for chunk in chunks if _get_active_document(str(chunk.get("document_id", ""))) is not None]


@protected_router.get("/documents")
async def list_documents():
    import api.main as m

    # status='deleting' documents are durably-tombstoned but not yet fully
    # torn down (see delete_document()) — hidden here so a client never sees
    # something it can't act on, even though they still exist in
    # documents_registry for _reconcile_document_deletion()/the startup
    # sweep to find and finish.
    visible = [d for d in m.documents_registry.values() if d.get("status", "active") == "active"]
    return {"total": len(visible), "documents": visible,
            "folders": sorted(m.folders_registry)}


@protected_router.patch("/documents/{doc_id}/folder", dependencies=[Depends(require_not_backing_up)])
async def update_document_folder(doc_id: str, body: dict):
    import api.main as m

    async with m._metadata_mutation_lock:
        doc = _get_active_document(doc_id)
        if doc is None:
            raise HTTPException(404, f"Document {doc_id} not found")
        folder = body.get("folder", "")

        # Postgres first, durably, before touching memory. Folder creation is
        # part of this same transaction, so failure leaves both registries
        # and the folders table untouched.
        await asyncio.to_thread(m.db_update_document_folder, doc_id, folder)
        if folder:
            m.folders_registry.add(folder)
        doc["folder"] = folder
        doc["folder_synced"] = False  # db_update_document_folder() just set this in Postgres too

        # Qdrant's copy is a derived index, synced best-effort — a failure
        # leaves folder_synced=False for later reconciliation.
        await m._reconcile_document_folder_sync(doc_id)
        return {"doc_id": doc_id, "folder": folder, "qdrant_synced": doc["folder_synced"]}


@protected_router.get("/folders")
async def list_folders():
    import api.main as m

    # Merge folders from registry and from documents
    doc_folders = {
        d["folder"] for d in m.documents_registry.values()
        if d.get("status", "active") == "active" and d.get("folder")
    }
    all_folders = sorted(m.folders_registry | doc_folders)
    return {"folders": all_folders}


@protected_router.post("/folders", dependencies=[Depends(require_not_backing_up)])
async def create_folder(body: dict):
    import api.main as m

    name = (body.get("name") or "").strip()
    if not name:
        raise HTTPException(400, "Folder name required")
    # DB first — db_save_folder() now raises instead of swallowing, so a
    # failure here 500s before folders_registry is touched at all.
    async with m._metadata_mutation_lock:
        await asyncio.to_thread(m.db_save_folder, name)
        m.folders_registry.add(name)
        return {"name": name}


@protected_router.delete("/folders/{name}", dependencies=[Depends(require_not_backing_up)])
async def delete_folder(name: str):
    import api.main as m

    # Same ordering as create_folder(): DB first, memory only after it
    # durably succeeds.
    async with m._metadata_mutation_lock:
        await asyncio.to_thread(m.db_delete_folder, name)
        m.folders_registry.discard(name)
        return {"deleted": name}


@protected_router.patch("/folders/{name}", dependencies=[Depends(require_not_backing_up)])
async def rename_folder(name: str, body: dict):
    import api.main as m

    new_name = (body.get("name") or "").strip()
    if not new_name:
        raise HTTPException(400, "New name required")

    # Durable, atomic (one Postgres transaction — see db_rename_folder())
    # rename FIRST: this IS the operation's source of truth. It also marks
    # every affected document folder_synced=FALSE in that same transaction,
    # so even a crash right after this leaves a durable record of exactly
    # which documents still need their Qdrant payload updated — nothing
    # here depends on the per-document loop below completing.
    async with m._metadata_mutation_lock:
        await asyncio.to_thread(m.db_rename_folder, name, new_name)

        m.folders_registry.discard(name)
        m.folders_registry.add(new_name)
        affected_doc_ids = [
            doc_id for doc_id, doc in m.documents_registry.items()
            if doc.get("status", "active") == "active" and doc.get("folder") == name
        ]
        for doc_id in affected_doc_ids:
            m.documents_registry[doc_id]["folder"] = new_name
            m.documents_registry[doc_id]["folder_synced"] = False

        # Best-effort from here — each document is reconciled independently,
        # and the mutation lock remains held through the derived-index writes.
        qdrant_sync_pending = []
        for doc_id in affected_doc_ids:
            await m._reconcile_document_folder_sync(doc_id)
            if not m.documents_registry[doc_id].get("folder_synced", True):
                qdrant_sync_pending.append(doc_id)

        if qdrant_sync_pending:
            m.logger.warning(
                f"Folder rename {name!r} -> {new_name!r}: Postgres committed for all "
                f"{len(affected_doc_ids)} documents, but Qdrant sync failed for "
                f"{len(qdrant_sync_pending)} of them — will retry on next access/restart."
            )

        return {"old": name, "new": new_name, "documents_updated": len(affected_doc_ids),
                "qdrant_sync_pending": qdrant_sync_pending}


@protected_router.delete("/documents/{doc_id}", dependencies=[Depends(require_not_backing_up)])
async def delete_document(doc_id: str):
    """Deletes a document across Qdrant + Postgres + disk. Not a single
    atomic operation (there is no cross-store transaction that could make it
    one) — instead, status='deleting' in Postgres (see db_mark_document_
    deleting()) is written FIRST and durably, before anything else is
    touched, so a crash between any of the steps below always leaves a
    trail: either this call never got past the tombstone write (nothing
    happened, retry from scratch is correct), or it did, in which case
    _reconcile_document_deletion() picks up from wherever it left off —
    Qdrant delete-by-filter and unlink(missing_ok=True) are both safe to
    repeat on content that's already gone. Calling this endpoint again on a
    doc_id already marked 'deleting' is exactly that retry, and is also what
    the startup reconciliation sweep does automatically for anything a
    previous crash left unfinished."""
    import api.main as m

    async with m._metadata_mutation_lock:
        doc = m.documents_registry.get(doc_id)
        if doc is None:
            raise HTTPException(404, f"Document {doc_id} not found")

        if doc.get("status", "active") == "active":
            await asyncio.to_thread(m.db_mark_document_deleting, doc_id)
            doc["status"] = "deleting"
        elif doc.get("status") != "deleting":
            raise HTTPException(404, f"Document {doc_id} not found")

        return await m._reconcile_document_deletion(doc_id)


@protected_router.get("/documents/{doc_id}/pages/{page_num}")
async def get_document_page(doc_id: str, page_num: int):
    """Canonical source-viewing endpoint for both TXT and PDF documents —
    same response shape for both {document_id, format, page, total_pages,
    text, has_ocr} so the frontend needs no format branching — but the two
    formats source that text differently:

    - TXT: decoded + normalize_whitespace()'d straight off the uploaded
      file on every request, exactly like the old (now-removed)
      /documents/{doc_id}/content. TxtParser's output is a pure function of
      the file's bytes with no external library version to drift, so there
      is nothing here that a persisted copy would protect against — storing
      it in document_pages too would only add a failure mode (a TXT
      document that was never backfilled 404s) for no correctness benefit.
    - PDF: read from document_pages, persisted once at ingestion time (see
      db_save_ingestion()) — PyMuPDF/Tesseract extraction is NOT
      a pure function of just the file (library/config versions matter),
      so this is re-derived at ingestion and never again, keeping it in
      lockstep with the char_start/char_end offsets /query sources hand out.

    Superseded /documents/{doc_id}/content (TXT-only) and /pdf/{doc_id}/
    highlights (PyMuPDF page.search_for() against the PDF's own text layer)
    — the latter had no fallback for OCR'd pages (no embedded text layer to
    search) and, even on clean digital PDFs, could land boxes on visually
    blank space wherever the page's internal reading order didn't match its
    visual layout (e.g. text positioned near an inserted image). Offset-
    based lookup has none of that: it never touches the PDF's rendering."""
    import api.main as m

    doc = _get_active_document(doc_id)
    if doc is None:
        raise HTTPException(404, "Document not found")
    doc_format = doc.get("format", "pdf")
    total_pages = 1 if doc_format == "txt" else doc.get("pages", 1)
    if page_num < 1 or page_num > total_pages:
        raise HTTPException(404, f"Page {page_num} out of range (1-{total_pages})")

    if doc_format == "txt":
        txt_file = next(m.UPLOAD_DIR.glob(f"{doc_id}_*"), None)
        if not txt_file:
            raise HTTPException(404, "File not found on disk")

        def _read_normalized():
            return normalize_whitespace(decode_text_file(txt_file.read_bytes()))

        text = await asyncio.to_thread(_read_normalized)
        has_ocr = False
    else:
        def _fetch():
            with m.db_conn() as conn:
                cur = conn.cursor()
                cur.execute(
                    "SELECT normalized_text, has_ocr FROM document_pages WHERE document_id = %s AND page_num = %s",
                    (doc_id, page_num),
                )
                return cur.fetchone()

        row = await asyncio.to_thread(_fetch)
        if not row:
            raise HTTPException(404, "Page text not indexed for this document — re-upload to backfill")
        text, has_ocr = row
        has_ocr = bool(has_ocr)

    return {
        "document_id": doc_id,
        "format": doc_format,
        "page": page_num,
        "total_pages": total_pages,
        "text": text,
        "has_ocr": has_ocr,
    }


@public_router.get("/pdf/{doc_id}")
async def get_pdf(doc_id: str, key: Optional[str] = None, x_api_key: Optional[str] = Header(default=None)):
    """PDF.js fetches this via XHR/fetch (not a raw browser navigation), so it
    can send X-API-Key like every other endpoint — prefer that over the
    ?key= query param, which lands in browser history and server access
    logs. The query param is kept only as a fallback for any caller that
    can't set a header (e.g. a plain <a href> opened in a new tab)."""
    import api.main as m
    from fastapi.responses import FileResponse

    if m.API_KEY and key != m.API_KEY and x_api_key != m.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    doc = _get_active_document(doc_id)
    if doc is None:
        raise HTTPException(404, "Document not found")
    for f in m.UPLOAD_DIR.glob(f"{doc_id}_*"):
        return FileResponse(path=str(f), media_type="application/pdf",
                          filename=doc["filename"])
    raise HTTPException(404, "PDF file not found on disk")
