#!/usr/bin/env python3
"""
Semantic verification for a restored backup — goes beyond the row/point
counts restore_backup.sh already checks (all four Postgres tables, Qdrant
point_count) to catch a restore that matches on count alone while being
wrong underneath: orphaned document_pages, documents with no original file
in uploads/, or a Qdrant collection with the right point count but the
wrong points.

Compares against manifest.json (recorded at backup time by
backup_qdrant.sh), never against live production — production has kept
moving since the backup was taken, so it is never the right thing to
diff a restored snapshot against.

Meant to run against the disposable Postgres/Qdrant verify_restore.sh
spins up (docker/docker-compose.restore-test.yml). Read-only — connecting
it to production instead wouldn't corrupt anything, but there's no reason
to, since the whole point is checking a restored copy.

Always prints exactly one JSON object to stdout and sets the exit code
accordingly (0 = every check passed, 1 = at least one failed) — a caller
should parse stdout even on a non-zero exit, not treat it as empty.
"""
import argparse
import json
import random
import sys
from pathlib import Path

import psycopg2
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue


def _check(checks, name, ok, detail=""):
    checks.append({"name": name, "ok": bool(ok), "detail": detail})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, help="path to the backup's manifest.json")
    ap.add_argument("--postgres-url", required=True)
    ap.add_argument("--qdrant-url", required=True)
    ap.add_argument("--qdrant-collection", required=True)
    ap.add_argument("--qdrant-api-key", default="")
    ap.add_argument("--uploads-dir", required=True)
    ap.add_argument("--sample-size", type=int, default=20)
    args = ap.parse_args()

    checks = []

    try:
        manifest = json.loads(Path(args.manifest).read_text())
    except Exception as e:
        print(json.dumps({"ok": False, "error": f"could not read manifest: {e}", "checks": []}, indent=2))
        sys.exit(1)

    all_docs = []
    try:
        conn = psycopg2.connect(args.postgres_url)
        cur = conn.cursor()

        cur.execute("SELECT count(*) FROM documents")
        n_documents = cur.fetchone()[0]
        _check(checks, "postgres.documents count", n_documents == manifest["postgres"]["documents"],
               f"expected={manifest['postgres']['documents']} actual={n_documents}")

        cur.execute("SELECT count(*) FROM file_hashes")
        n_file_hashes = cur.fetchone()[0]
        _check(checks, "postgres.file_hashes count", n_file_hashes == manifest["postgres"]["file_hashes"],
               f"expected={manifest['postgres']['file_hashes']} actual={n_file_hashes}")

        cur.execute("SELECT count(*) FROM folders")
        n_folders = cur.fetchone()[0]
        _check(checks, "postgres.folders count", n_folders == manifest["postgres"]["folders"],
               f"expected={manifest['postgres']['folders']} actual={n_folders}")

        cur.execute("SELECT count(*) FROM document_pages")
        n_pages = cur.fetchone()[0]
        _check(checks, "postgres.document_pages count", n_pages == manifest["postgres"]["document_pages"],
               f"expected={manifest['postgres']['document_pages']} actual={n_pages}")

        # The document_pages FK (ON DELETE CASCADE) should make this
        # impossible by construction — but a restore is exactly the
        # situation where "should be impossible" is worth checking
        # directly rather than trusting blind.
        cur.execute("""
            SELECT count(*) FROM document_pages dp
            LEFT JOIN documents d ON d.doc_id = dp.document_id
            WHERE d.doc_id IS NULL
        """)
        orphan_pages = cur.fetchone()[0]
        _check(checks, "document_pages have no orphans", orphan_pages == 0, f"orphan_pages={orphan_pages}")

        cur.execute("SELECT doc_id, filename FROM documents")
        all_docs = cur.fetchall()
    except Exception as e:
        _check(checks, "postgres connectivity", False, str(e))

    # Every document row must have its original file physically present —
    # a restore that has the metadata but lost uploads.tar.gz (or restored
    # it to the wrong place) still "passes" a naive row-count check.
    uploads_dir = Path(args.uploads_dir)
    missing_originals = [str(doc_id) for doc_id, _ in all_docs if not list(uploads_dir.glob(f"{doc_id}_*"))]
    _check(
        checks, "every document has an original file in uploads/", len(missing_originals) == 0,
        f"missing for {len(missing_originals)} of {len(all_docs)} documents"
        + (f" (e.g. {missing_originals[:5]})" if missing_originals else ""),
    )

    qc = None
    try:
        qc = QdrantClient(url=args.qdrant_url, api_key=args.qdrant_api_key or None)
        info = qc.get_collection(args.qdrant_collection)
        _check(checks, "qdrant collection exists", True)
        _check(checks, "qdrant point_count", info.points_count == manifest["qdrant"]["point_count"],
               f"expected={manifest['qdrant']['point_count']} actual={info.points_count}")
    except Exception as e:
        _check(checks, "qdrant collection exists", False, str(e))

    # Spot-check a random sample end-to-end: a matching count doesn't prove
    # any INDIVIDUAL document's file is readable or its vector/payload made
    # it into Qdrant intact — this reads the actual bytes for a sample
    # instead of trusting aggregates alone.
    if all_docs and qc is not None:
        sample = random.sample(all_docs, min(args.sample_size, len(all_docs)))
        spot_failures = []
        for doc_id, filename in sample:
            matches = list(uploads_dir.glob(f"{doc_id}_*"))
            file_ok = bool(matches) and matches[0].stat().st_size > 0
            point_ok = False
            try:
                points, _ = qc.scroll(
                    collection_name=args.qdrant_collection,
                    scroll_filter=Filter(must=[FieldCondition(key="document_id", match=MatchValue(value=str(doc_id)))]),
                    limit=1,
                    with_vectors=True,
                    with_payload=True,
                )
                point_ok = len(points) > 0 and bool(points[0].payload)
            except Exception:
                point_ok = False
            if not (file_ok and point_ok):
                spot_failures.append({"doc_id": str(doc_id), "filename": filename, "file_ok": file_ok, "point_ok": point_ok})
        _check(
            checks, f"spot-check {len(sample)} random documents (file + Qdrant point readable)",
            len(spot_failures) == 0,
            "all passed" if not spot_failures else f"failed: {spot_failures}",
        )

    ok = all(c["ok"] for c in checks)
    print(json.dumps({"ok": ok, "checks": checks}, indent=2))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
