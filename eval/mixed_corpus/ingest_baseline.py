#!/usr/bin/env python3
"""Resumable baseline ingestion of the mixed corpus into an isolated instance.

Unlike scripts/bulk_ingest_pdfs.py (flat --source-dir, single implicit
category), this reads eval/mixed_corpus/manifest.json directly so each
document's real category/split travels with it — /upload-batch's `folder`
tag is set to the manifest category, so a later per-category breakdown (see
extraction_verdict.md) doesn't require re-deriving it from the filename.

Point --api-url at the isolated instance only (see eval/mixed_corpus/README.md
"Isolation" — port 8002, `mixed_corpus_test` Qdrant collection, separate
Postgres DB `ragdb_mixed_corpus_test`). This intentionally does NOT filter
out documents extraction_probe.json predicted would fail — the whole point
of a baseline run is to confirm the real API's behavior matches the
prediction, not to pre-clean the input.

Usage:
    python eval/mixed_corpus/ingest_baseline.py
    python eval/mixed_corpus/ingest_baseline.py --api-url http://localhost:8002
    python eval/mixed_corpus/ingest_baseline.py --api-url http://localhost:8003 \\
        --progress-path eval/mixed_corpus/baseline_ingest_progress_v2.json \\
        --report-path eval/mixed_corpus/baseline_ingest_report_v2.json
"""
import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx

CORPUS_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = CORPUS_DIR / "manifest.json"


def load_progress(progress_path: Path) -> dict:
    if progress_path.exists():
        return json.loads(progress_path.read_text(encoding="utf-8"))
    return {}


def save_progress(progress_path: Path, done: dict):
    progress_path.write_text(json.dumps(done, indent=2, ensure_ascii=False), encoding="utf-8")


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api-url", default="http://localhost:8002")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--batch-size", type=int, default=20)
    ap.add_argument("--progress-path", default=str(CORPUS_DIR / "baseline_ingest_progress.json"),
                     help="Resumable per-document progress file. MUST be distinct per target "
                          "collection — reusing another run's progress file makes this think "
                          "every document is already done and skip uploading anything, even "
                          "against an empty collection.")
    ap.add_argument("--report-path", default=str(CORPUS_DIR / "baseline_ingest_report.json"))
    args = ap.parse_args()
    progress_path = Path(args.progress_path)
    report_path = Path(args.report_path)

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    done = load_progress(progress_path)  # relative_path -> result dict

    todo = [m for m in manifest if m["relative_path"] not in done]
    print(f"{len(done)} already done, {len(todo)} remaining of {len(manifest)} total")

    headers = {"X-API-Key": args.api_key} if args.api_key else {}
    t0 = time.time()
    async with httpx.AsyncClient(base_url=args.api_url, headers=headers, timeout=180.0) as client:
        r = await client.get("/health")
        r.raise_for_status()
        health = r.json()
        print(f"Target: {args.api_url} | collection={health['vector_store']['collection']} "
              f"| vectors_before={health['vector_store']['total_vectors']}")
        assert health["vector_store"]["collection"] != "knowledge_base", \
            "Refusing to ingest into the main 'knowledge_base' collection — check --api-url"

        for i in range(0, len(todo), args.batch_size):
            batch = todo[i:i + args.batch_size]
            files = []
            for entry in batch:
                path = CORPUS_DIR / entry["relative_path"]
                files.append(("files", (Path(entry["relative_path"]).name, path.read_bytes(), "application/pdf")))
            # Every doc in one manifest batch may span different categories;
            # /upload-batch takes one folder tag per call, so tag with the
            # majority category here and rely on per-doc entries in `done`
            # (keyed by relative_path, which retains the real category) for
            # anything that needs to be exact later.
            folder = batch[0]["category"]
            t_batch = time.time()
            try:
                resp = await client.post("/upload-batch", files=files, data={"folder": folder})
                resp.raise_for_status()
                result = resp.json()
            except httpx.HTTPError as exc:
                # A transport-level failure (timeout, connection reset) — not
                # a per-file rejection, which /upload-batch reports inside
                # result["results"] instead. Leave this batch out of `done`
                # entirely so a re-run retries it.
                print(f"  batch {i}-{i+len(batch)} TRANSPORT ERROR: {exc} — will retry next run")
                continue

            by_name = {r["filename"]: r for r in result["results"]}
            for entry in batch:
                fname = Path(entry["relative_path"]).name
                r = by_name.get(fname, {"status": "unknown", "error": "not in response"})
                done[entry["relative_path"]] = {
                    "category": entry["category"], "split": entry["split"],
                    # /upload-batch's real status vocabulary (api/main.py):
                    # "indexed" | "skipped" (duplicate content hash) | "error"
                    "status": r.get("status"), "error": r.get("error"),
                    "chunks": r.get("chunks_created"), "doc_id": r.get("doc_id"),
                }
            save_progress(progress_path, done)

            elapsed = time.time() - t0
            batch_ms = int((time.time() - t_batch) * 1000)
            n_done = len(done)
            rate = n_done / elapsed if elapsed > 0 else 0
            print(f"  {n_done}/{len(manifest)} | batch={folder} errors={result.get('errors', 0)} "
                  f"batch_ms={batch_ms} | {rate:.2f} docs/sec overall")

    # ── final report ──
    ok = [v for v in done.values() if v["status"] == "indexed"]
    skipped = [v for v in done.values() if v["status"] == "skipped"]
    failed = [v for v in done.values() if v["status"] not in ("indexed", "skipped")]
    by_cat: dict[str, dict[str, int]] = {}
    for v in done.values():
        c = v["category"]
        by_cat.setdefault(c, {"ok": 0, "skipped": 0, "failed": 0})
        key = "ok" if v["status"] == "indexed" else ("skipped" if v["status"] == "skipped" else "failed")
        by_cat[c][key] += 1

    report = {
        "total": len(done), "ok": len(ok), "skipped": len(skipped), "failed": len(failed),
        "by_category": by_cat,
        "elapsed_s": round(time.time() - t0, 1),
    }
    report_path.write_text(json.dumps({"summary": report, "detail": done}, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{'='*60}\nBASELINE INGEST — {len(done)}/{len(manifest)} documents\n{'='*60}")
    for cat in sorted(by_cat):
        c = by_cat[cat]
        print(f"{cat:20s} ok={c['ok']:4d} skipped={c['skipped']:4d} failed={c['failed']:4d}")
    pct = f"{100*len(ok)/len(done):.1f}%" if done else "n/a"
    print(f"\nTotal: ok={len(ok)} skipped={len(skipped)} failed={len(failed)} ({pct} indexed)")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    asyncio.run(main())
