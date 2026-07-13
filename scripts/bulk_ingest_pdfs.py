#!/usr/bin/env python3
"""
Resumable bulk ingestion of a real PDF corpus, in a fixed, reproducible
order — built for the EU-legislation scale test (see eval/README.md,
"Scale test: EU legislation corpus"), but generic to any --source-dir.

The whole point is a stable ordering: on first run, lists every PDF in
--source-dir, shuffles it once with --seed, and persists that order to
--manifest-path. Every later invocation reads the same manifest instead
of re-shuffling, so "--up-to 5000" always means "the first 5000 files
of the same fixed order" regardless of which run it's called from —
each scale-ladder checkpoint is a strict superset of the previous one,
and re-running with the same --up-to is a safe no-op (progress is
tracked separately in --progress-path so a batch failure partway through
doesn't lose the count).

Talks to a running instance's /upload-batch (same endpoint and batching
pattern as scripts/scale_test.py) — point --api-url at an isolated
instance (see eval/README.md for how the EU scale test runs a second API
process against a separate Qdrant collection + Postgres database, so this
never touches the calibrated golden/heldout corpus on the main instance).

Usage:
    python scripts/bulk_ingest_pdfs.py --up-to 1000
    python scripts/bulk_ingest_pdfs.py --up-to 5000   # only uploads the new slice
"""
import argparse
import asyncio
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx
from dotenv import load_dotenv

load_dotenv()

DEFAULT_SOURCE_DIR = "/mnt/c/Users/serg/Downloads/PDFS LEGAL RAG TEST/pdfs"
EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"


def build_or_load_manifest(source_dir: Path, manifest_path: Path, seed: int) -> list[str]:
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        print(f"Loaded existing manifest: {len(manifest)} files ({manifest_path})")
        return manifest

    print(f"No manifest at {manifest_path} — building one from {source_dir} ...")
    files = sorted(p.name for p in source_dir.glob("*.pdf"))
    print(f"Found {len(files)} PDFs")
    random.Random(seed).shuffle(files)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(files))
    print(f"Manifest written (seed={seed}): {manifest_path}")
    return files


def load_progress(progress_path: Path) -> int:
    if progress_path.exists():
        return json.loads(progress_path.read_text())["uploaded"]
    return 0


def save_progress(progress_path: Path, uploaded: int):
    progress_path.write_text(json.dumps({"uploaded": uploaded}))


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-dir", default=DEFAULT_SOURCE_DIR)
    ap.add_argument("--api-url", default=os.getenv("SCALE_TEST_API_URL", "http://localhost:8001"))
    ap.add_argument("--api-key", default=os.getenv("API_KEY", ""))
    ap.add_argument("--folder", default="eu-scale-test")
    ap.add_argument("--batch-size", type=int, default=20)
    ap.add_argument("--manifest-path", default=str(EVAL_DIR / "eu_manifest.json"))
    ap.add_argument("--progress-path", default=str(EVAL_DIR / "eu_ingest_progress.json"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--up-to", type=int, required=True, help="Target cumulative document count")
    args = ap.parse_args()

    source_dir = Path(args.source_dir)
    manifest_path = Path(args.manifest_path)
    progress_path = Path(args.progress_path)

    manifest = build_or_load_manifest(source_dir, manifest_path, args.seed)
    if args.up_to > len(manifest):
        print(f"WARNING: --up-to {args.up_to} exceeds manifest size {len(manifest)}, clamping.")
        args.up_to = len(manifest)

    already = load_progress(progress_path)
    if already >= args.up_to:
        print(f"Already at {already} >= --up-to {args.up_to} — nothing to do.")
        return

    todo = manifest[already:args.up_to]
    print(f"Uploading {len(todo)} new files ({already} -> {args.up_to}) in batches of {args.batch_size} ...")

    headers = {"X-API-Key": args.api_key} if args.api_key else {}
    uploaded = already
    t0 = time.time()
    async with httpx.AsyncClient(base_url=args.api_url, headers=headers, timeout=180.0) as client:
        r = await client.get("/health")
        r.raise_for_status()

        for i in range(0, len(todo), args.batch_size):
            batch_names = todo[i:i + args.batch_size]
            files = []
            for name in batch_names:
                path = source_dir / name
                files.append(("files", (name, path.read_bytes(), "application/pdf")))
            resp = await client.post("/upload-batch", files=files, data={"folder": args.folder})
            resp.raise_for_status()
            result = resp.json()
            uploaded += len(batch_names)
            save_progress(progress_path, uploaded)
            if result["errors"]:
                bad = [x for x in result["results"] if x["status"] == "error"]
                print(f"  batch: {result['errors']} errors — {bad[:2]}")
            elapsed = time.time() - t0
            rate = (uploaded - already) / elapsed if elapsed > 0 else 0
            print(f"  {uploaded}/{args.up_to} total | {elapsed:.1f}s elapsed this run | {rate:.2f} docs/sec")

    ingest_s = time.time() - t0
    net = uploaded - already
    print(f"\nDone: +{net} docs in {ingest_s:.1f}s ({net / ingest_s:.2f} docs/sec) — now at {uploaded}/{len(manifest)}")


if __name__ == "__main__":
    asyncio.run(main())
