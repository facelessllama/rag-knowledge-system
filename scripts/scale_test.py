#!/usr/bin/env python3
"""
Scale test: ingest a synthetic corpus of N documents (default 1000) against
a running instance, report ingest throughput, and spot-check exact-term
retrieval (the whole point of hybrid/BM25-style search — semantic search
alone tends to lose specific codes/numbers).

Does NOT (and can't) measure server restart time or process memory from
outside the process — after running this, manually restart the API and
check the startup log line. It should show a fast Postgres-based document
count, not the old slow full-Qdrant-scroll restore (see api/main.py
init_db() / startup()).

Usage:
    python scripts/scale_test.py --count 1000
    python scripts/scale_test.py --count 100 --api-url http://localhost:8000
"""
import argparse
import asyncio
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fitz
import httpx
from dotenv import load_dotenv

load_dotenv()

TOPICS_RU = ["аренды", "поставки", "оказания услуг", "подряда", "займа", "трудовой договор"]

FILLER = (
    "Стороны обязуются исполнять условия настоящего договора добросовестно. "
    "Любые изменения вносятся по письменному соглашению сторон. "
    "Споры разрешаются в порядке, установленном законодательством. "
    "The parties shall perform their obligations under this agreement in good faith. "
    "Any amendments require written consent of both parties. "
    "Disputes shall be resolved in accordance with applicable law. "
)


def _wrap(text: str, width: int):
    words = text.split()
    line = []
    for w in words:
        line.append(w)
        if len(" ".join(line)) > width:
            yield " ".join(line)
            line = []
    if line:
        yield " ".join(line)


def _make_doc_pdf(path: Path, doc_index: int, article_num: int, needle: str):
    doc = fitz.open()
    topic = TOPICS_RU[doc_index % len(TOPICS_RU)]
    for page_num in range(3):
        page = doc.new_page()
        y = 72
        page.insert_text((72, y), f"Договор {topic} № {doc_index} — Article {article_num}", fontsize=13)
        y += 26
        text = f"Статья {article_num}. Section {article_num}.\n" + FILLER * 6
        if page_num == 1 and needle:
            text += f"\n{needle}\n"
        for line in _wrap(text, 95):
            if y > 780:
                break
            page.insert_text((72, y), line, fontsize=9)
            y += 13
    doc.save(str(path))
    doc.close()


async def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--api-url", default=os.getenv("SCALE_TEST_API_URL", "http://localhost:8000"))
    parser.add_argument("--api-key", default=os.getenv("API_KEY", ""))
    parser.add_argument("--folder", default="scale-test")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--out-dir", default="/tmp/scale_test_pdfs")
    parser.add_argument("--keep-files", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    headers = {"X-API-Key": args.api_key} if args.api_key else {}

    needle_doc_index = args.count // 2
    needle_term = f"zzqx{random.randint(10000, 99999)}"
    needle_sentence = f"The special reference code for this audit is {needle_term}."

    print(f"Generating {args.count} synthetic PDFs in {out_dir} ...")
    t0 = time.time()
    paths = []
    for i in range(args.count):
        p = out_dir / f"doc_{i:05d}.pdf"
        needle = needle_sentence if i == needle_doc_index else ""
        _make_doc_pdf(p, i, article_num=(i % 40) + 1, needle=needle)
        paths.append(p)
    print(f"Generated {len(paths)} PDFs in {(time.time() - t0) * 1000:.0f}ms")

    async with httpx.AsyncClient(base_url=args.api_url, headers=headers, timeout=120.0) as client:
        r = await client.get("/health")
        r.raise_for_status()
        print(f"Health check OK: {r.json()['status']}")

        print(f"Uploading {len(paths)} documents in batches of {args.batch_size} ...")
        t0 = time.time()
        indexed = 0
        for i in range(0, len(paths), args.batch_size):
            batch = paths[i:i + args.batch_size]
            files = [("files", (p.name, p.read_bytes(), "application/pdf")) for p in batch]
            resp = await client.post("/upload-batch", files=files, data={"folder": args.folder})
            resp.raise_for_status()
            result = resp.json()
            indexed += result["indexed"]
            if result["errors"]:
                bad = [x for x in result["results"] if x["status"] == "error"]
                print(f"  batch {i}-{i + len(batch)}: {result['errors']} errors — {bad[:2]}")
            elapsed = time.time() - t0
            rate = indexed / elapsed if elapsed > 0 else 0
            print(f"  {indexed}/{len(paths)} indexed | {elapsed:.1f}s elapsed | {rate:.2f} docs/sec")
        ingest_s = time.time() - t0
        print(f"\nIngest complete: {indexed}/{len(paths)} docs in {ingest_s:.1f}s "
              f"({indexed / ingest_s:.2f} docs/sec)")

        docs_resp = await client.get("/documents")
        print(f"/documents now reports {docs_resp.json()['total']} total documents")

        print(f"\nSpot-checking exact-term retrieval for needle '{needle_term}' ...")
        t0 = time.time()
        q = await client.post("/query", json={
            "question": f"What is the special reference code {needle_term}?",
            "folder": args.folder, "top_k": 5,
        })
        q.raise_for_status()
        query_ms = (time.time() - t0) * 1000
        result = q.json()
        found = needle_term in result["answer"] or any(
            needle_term in s.get("chunk_text", "") for s in result["sources"]
        )
        print(f"Query took {query_ms:.0f}ms | needle found in answer/sources: {found}")
        if not found:
            print("WARNING: exact-term needle not retrieved — check hybrid sparse-vector search "
                  "(vector_db/sparse_encoder.py, vector_db/qdrant_client.py hybrid_search).")

    if not args.keep_files:
        for p in paths:
            p.unlink(missing_ok=True)
        print(f"\nCleaned up {len(paths)} temp PDFs (pass --keep-files to retain them).")

    print("\nNext step (manual): restart the API process and check the startup log —")
    print("it should show a fast Postgres-based document count, not a slow full Qdrant scroll.")


if __name__ == "__main__":
    asyncio.run(main())
