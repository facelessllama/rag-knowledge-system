#!/usr/bin/env python3
"""
Concurrency test: fires N simultaneous /query requests and compares
wall-clock time against a sequential baseline, to verify that CPU/GPU-bound
calls (embedding, reranking) no longer block the event loop and serialize
requests regardless of MAX_CONCURRENT_QUERIES (see rag/executors.py,
rag/retriever.py, api/main.py).

Requires at least one document already ingested — run scripts/scale_test.py
first, or upload manually, so /query has something to retrieve.

Note: MAX_CONCURRENT_QUERIES (default 3, see .env) caps how many requests
are admitted at once — requests beyond that get an immediate 429, which is
intentional backpressure, not a bug. To see real parallelism across all N
requests, set MAX_CONCURRENT_QUERIES >= --requests before starting the API.

Usage:
    python scripts/concurrency_test.py --requests 10
"""
import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx
from dotenv import load_dotenv

load_dotenv()


async def _one_query(client: httpx.AsyncClient, question: str, folder: str | None):
    t0 = time.time()
    payload = {"question": question, "top_k": 5}
    if folder:
        payload["folder"] = folder
    r = await client.post("/query", json=payload)
    elapsed = time.time() - t0
    if r.status_code == 429:
        return ("rejected_429", elapsed)
    r.raise_for_status()
    return ("ok", elapsed)


async def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--requests", type=int, default=10)
    parser.add_argument("--api-url", default=os.getenv("SCALE_TEST_API_URL", "http://localhost:8000"))
    parser.add_argument("--api-key", default=os.getenv("API_KEY", ""))
    parser.add_argument("--folder", default=None)
    parser.add_argument("--question", default="What are the main obligations of the parties?")
    args = parser.parse_args()

    headers = {"X-API-Key": args.api_key} if args.api_key else {}

    async with httpx.AsyncClient(base_url=args.api_url, headers=headers, timeout=120.0) as client:
        r = await client.get("/documents")
        r.raise_for_status()
        total = r.json()["total"]
        if total == 0:
            print("No documents ingested yet — run scripts/scale_test.py first.")
            return
        print(f"{total} documents in knowledge base.")

        print("\n--- Sequential baseline (3 requests) ---")
        seq_times = []
        for _ in range(3):
            status, elapsed = await _one_query(client, args.question, args.folder)
            seq_times.append(elapsed)
        avg_seq = sum(seq_times) / len(seq_times)
        print(f"Sequential avg latency: {avg_seq * 1000:.0f}ms ({[f'{t * 1000:.0f}ms' for t in seq_times]})")

        print(f"\n--- {args.requests} concurrent requests ---")
        t0 = time.time()
        results = await asyncio.gather(
            *[_one_query(client, args.question, args.folder) for _ in range(args.requests)],
            return_exceptions=True,
        )
        wall_time = time.time() - t0

        ok = [r for r in results if isinstance(r, tuple) and r[0] == "ok"]
        rejected = [r for r in results if isinstance(r, tuple) and r[0] == "rejected_429"]
        errors = [r for r in results if not isinstance(r, tuple)]
        naive_sequential_estimate = avg_seq * len(ok) if ok else 0

        print(f"Concurrent wall time: {wall_time * 1000:.0f}ms for {args.requests} requests "
              f"({len(ok)} ok, {len(rejected)} rejected with 429, {len(errors)} errors)")
        if rejected:
            print(f"({len(rejected)} requests hit the MAX_CONCURRENT_QUERIES semaphore — "
                  f"raise it in .env to test more parallelism at once)")
        if ok:
            print(f"Naive sequential estimate for the {len(ok)} admitted requests "
                  f"(avg_seq * {len(ok)}): {naive_sequential_estimate * 1000:.0f}ms")
            speedup = naive_sequential_estimate / wall_time if wall_time > 0 else 0
            print(f"Speedup vs naive sequential: {speedup:.2f}x")
            if speedup < 1.5:
                print("\nWARNING: little to no concurrency gain — check that asyncio.to_thread/"
                      "run_on_gpu wrapping is actually in place on the hot path.")
            else:
                print("\nConcurrency confirmed: admitted requests are NOT serializing on the event loop.")

        if errors:
            print(f"\nUnexpected errors ({len(errors)}):")
            for e in errors[:5]:
                print(f"  {e}")


if __name__ == "__main__":
    asyncio.run(main())
