#!/usr/bin/env python3
"""Builds a corpus-wide index of which documents contain a caption-tier
match for each (kind, num) Figure/Table label — e.g. {"Figure", "10"} ->
every document whose indexed chunks contain something that reads like an
actual "Figure 10" caption, not just an in-text reference to it.

This is the missing piece for measuring cross-document caption-number
COLLISIONS (see project memory, "Diagnostic follow-up: why chunk recall
plateaus at ~28%" — a same-numbered "Figure 10" caption in a completely
unrelated paper was confirmed, by hand, to have displaced the correct
document from the structural lookup's top-3 candidate window on one
case). Without a corpus-wide index, that's anecdotal; with it,
analyze_caption_misses.py can report how often it actually happens.

Reuses the exact detection function retrieve_expanded() itself uses
(rag.retriever.structural_match_tier) against every chunk of every
document, via chunks_for_document() — a deterministic Qdrant payload
scroll, not an embedding search, so unlike hybrid_search this has none of
the cross-process nondeterminism documented in project memory. Safe to
run once and cache.

Usage:
    python eval/mixed_corpus/build_caption_label_index.py \\
        --api-url http://localhost:8003 --qdrant-collection mixed_corpus_v2 \\
        --output eval/mixed_corpus/caption_label_index.json
"""
import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from vector_db.qdrant_client import VectorStore
from rag.retriever import extract_structural_references, structural_match_tier

CORPUS_DIR = Path(__file__).resolve().parent

# Loose, corpus-wide scan for CANDIDATE (kind, num) pairs per chunk —
# extract_structural_references() already does exactly this (it's designed
# for query text, but a chunk's text works the same way): every literal
# "Figure N"/"Table N" mention becomes a candidate, then structural_match_
# tier() (the same function retrieve_expanded() calls) decides whether any
# of them is actually caption-strength, not just an in-text reference.


def captions_in_text(text: str) -> set[tuple[str, str]]:
    found = set()
    for kind, num in extract_structural_references(text):
        if structural_match_tier(text, kind, num) is not None:
            found.add((kind, num))
    return found


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api-url", default=os.getenv("EVAL_API_URL", "http://localhost:8003"))
    ap.add_argument("--api-key", default=os.getenv("API_KEY", ""))
    ap.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL", "http://localhost:6333"))
    ap.add_argument("--qdrant-collection", required=True,
                     help="No default on purpose — same guard as capture_generation_contexts.py, "
                          "forgetting this would silently scan the wrong collection.")
    ap.add_argument("--qdrant-api-key", default=os.getenv("QDRANT_API_KEY"))
    ap.add_argument("--output", default=str(CORPUS_DIR / "caption_label_index.json"))
    args = ap.parse_args()

    assert args.qdrant_collection != "knowledge_base", (
        "Refusing to run against 'knowledge_base' (the main production collection) — "
        "pass --qdrant-collection explicitly (e.g. mixed_corpus_v2)."
    )

    headers = {"X-API-Key": args.api_key} if args.api_key else {}
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{args.api_url}/documents", headers=headers, timeout=30.0)
        resp.raise_for_status()
        documents = resp.json()["documents"]

    vector_store = VectorStore(url=args.qdrant_url, collection=args.qdrant_collection, api_key=args.qdrant_api_key)

    index: dict[str, list[list[str]]] = {}
    for i, d in enumerate(documents):
        doc_id = d["doc_id"]
        chunks = await asyncio.to_thread(vector_store.chunks_for_document, doc_id)
        labels: set[tuple[str, str]] = set()
        for c in chunks:
            labels |= captions_in_text(c.get("text", ""))
        if labels:
            index[doc_id] = sorted(labels)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(documents)} documents scanned", flush=True)

    Path(args.output).write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")
    total_labels = sum(len(v) for v in index.values())
    print(f"\n{len(index)}/{len(documents)} documents have at least one caption-tier label; "
          f"{total_labels} (document, label) pairs total.")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
