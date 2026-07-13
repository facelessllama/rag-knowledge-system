#!/usr/bin/env python3
"""
One-off backfill: adds case_number/case_year/parties payload fields (see
ingestion/chunker.py::extract_case_metadata, vector_db/qdrant_client.py) to
points ingested before this feature existed.

Every field this needs (the filename) is already in each point's payload,
so this does NOT re-parse PDFs, re-embed, or re-chunk anything — it scrolls
existing points and patches in the three new fields via set_payload. New
ingestion (vector_db/qdrant_client.py::upsert_chunks) already writes them
going forward; this only catches up the existing corpus.

Usage:
    source venv/bin/activate
    python scripts/backfill_case_metadata.py
"""
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("backfill_case_metadata")

SCROLL_BATCH = 500


def main():
    from qdrant_client import QdrantClient

    from ingestion.chunker import extract_case_metadata
    from vector_db.qdrant_client import VectorStore

    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    collection = os.getenv("QDRANT_COLLECTION", "knowledge_base")

    client = QdrantClient(url=qdrant_url)

    # Make sure the case_number/case_year payload indexes exist before we
    # start writing to those fields (harmless no-op if already present).
    VectorStore(url=qdrant_url, collection=collection)._ensure_payload_indexes()

    updated = 0
    skipped_no_match = 0
    offset = None
    while True:
        points, next_offset = client.scroll(
            collection_name=collection, limit=SCROLL_BATCH, offset=offset,
            with_payload=True, with_vectors=False,
        )
        if not points:
            break

        by_metadata: dict[tuple, list[str]] = {}
        for point in points:
            payload = point.payload or {}
            if "case_number" in payload and payload["case_number"]:
                continue  # already backfilled (or ingested post-feature)
            meta = extract_case_metadata(payload.get("filename", ""))
            if not meta:
                skipped_no_match += 1
                continue
            key = (meta["case_number"], meta["case_year"], tuple(meta["parties"]))
            by_metadata.setdefault(key, []).append(point.id)

        for (num, year, parties), ids in by_metadata.items():
            client.set_payload(
                collection_name=collection,
                payload={"case_number": num, "case_year": year, "parties": list(parties)},
                points=ids,
            )
            updated += len(ids)

        logger.info(f"  scrolled batch of {len(points)} — {updated} updated so far, "
                    f"{skipped_no_match} skipped (no case-number filename match)")

        if next_offset is None:
            break
        offset = next_offset

    logger.info(f"Done. {updated} points backfilled, {skipped_no_match} skipped "
                f"(non-case documents, e.g. UK statutory instruments — expected).")


if __name__ == "__main__":
    main()
