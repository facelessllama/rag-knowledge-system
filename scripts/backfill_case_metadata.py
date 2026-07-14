#!/usr/bin/env python3
"""
One-off backfill: adds case_number/case_year/parties/celex_id/
citation_number/citation_year payload fields (see ingestion/chunker.py::
extract_case_metadata/extract_celex_id/extract_citation_number,
vector_db/qdrant_client.py) to points ingested before those fields
existed.

Every field this needs (the filename) is already in each point's payload,
so this does NOT re-parse PDFs, re-embed, or re-chunk anything — it scrolls
existing points and patches in the new fields via set_payload. New
ingestion (vector_db/qdrant_client.py::upsert_chunks) already writes them
going forward; this only catches up documents ingested before that.
Targets whichever collection QDRANT_COLLECTION points at (see .env / the
second API instance used for the EU scale test).

citation_number/citation_year is backfilled independently of case_number/
celex_id (a point can already have celex_id set from an earlier backfill
pass and still be missing citation_number — it was added later, see
eval/README.md "ID-free spot-check @ 57,000") — don't gate one on the
other's presence.

Usage:
    source venv/bin/activate
    python scripts/backfill_case_metadata.py
    QDRANT_COLLECTION=eu_scale_test python scripts/backfill_case_metadata.py
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

    from ingestion.chunker import extract_case_metadata, extract_celex_id, extract_citation_number
    from vector_db.qdrant_client import VectorStore

    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    collection = os.getenv("QDRANT_COLLECTION", "knowledge_base")

    client = QdrantClient(url=qdrant_url)

    # Make sure all payload indexes exist before we start writing to those
    # fields (harmless no-op if already present).
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

        by_case_metadata: dict[tuple, list[str]] = {}
        by_celex_id: dict[str, list[str]] = {}
        by_citation: dict[tuple, list[str]] = {}
        for point in points:
            payload = point.payload or {}
            filename = payload.get("filename", "")

            already_case = payload.get("case_number") or payload.get("celex_id")
            if not already_case:
                case_meta = extract_case_metadata(filename)
                celex_id = extract_celex_id(filename)
                if case_meta:
                    key = (case_meta["case_number"], case_meta["case_year"], tuple(case_meta["parties"]))
                    by_case_metadata.setdefault(key, []).append(point.id)
                elif celex_id:
                    by_celex_id.setdefault(celex_id, []).append(point.id)
                else:
                    skipped_no_match += 1

            if not payload.get("citation_number"):
                citation_meta = extract_citation_number(filename)
                if citation_meta:
                    key = (citation_meta["citation_number"], citation_meta["citation_year"])
                    by_citation.setdefault(key, []).append(point.id)

        for (num, year, parties), ids in by_case_metadata.items():
            client.set_payload(
                collection_name=collection,
                payload={"case_number": num, "case_year": year, "parties": list(parties)},
                points=ids,
            )
            updated += len(ids)

        for celex_id, ids in by_celex_id.items():
            client.set_payload(
                collection_name=collection,
                payload={"celex_id": celex_id},
                points=ids,
            )
            updated += len(ids)

        for (num, year), ids in by_citation.items():
            client.set_payload(
                collection_name=collection,
                payload={"citation_number": num, "citation_year": year},
                points=ids,
            )
            updated += len(ids)

        logger.info(f"  scrolled batch of {len(points)} — {updated} updated so far, "
                    f"{skipped_no_match} skipped (no recognized filename convention)")

        if next_offset is None:
            break
        offset = next_offset

    logger.info(f"Done. {updated} points backfilled, {skipped_no_match} skipped "
                f"(documents with no recognized identity convention — expected, e.g. UK statutory instruments).")


if __name__ == "__main__":
    main()
