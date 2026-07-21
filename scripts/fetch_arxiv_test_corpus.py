#!/usr/bin/env python3
"""Download a batch of real arXiv PDFs for RAG stress-testing and bundle them into one zip.

Uses the public arXiv API (export.arxiv.org) to list recent papers in the given
categories, then downloads each PDF directly from arxiv.org/pdf/. Respects
arXiv's rate-limit guidance (no more than ~1 request/3s to the API, small
delay between PDF fetches).
"""
import argparse
import re
import sys
import time
import zipfile
from pathlib import Path
from urllib.parse import urlencode
from xml.etree import ElementTree as ET

import requests

ARXIV_API = "http://export.arxiv.org/api/query"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}
PAGE_SIZE = 100


def list_ids(categories: list[str], target: int) -> list[tuple[str, str]]:
    """Return (arxiv_id, title) pairs across the given categories, newest first."""
    query = " OR ".join(f"cat:{c}" for c in categories)
    results: list[tuple[str, str]] = []
    seen: set[str] = set()
    start = 0
    while len(results) < target:
        params = {
            "search_query": query,
            "start": start,
            "max_results": PAGE_SIZE,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        resp = requests.get(f"{ARXIV_API}?{urlencode(params)}", timeout=30)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        entries = root.findall("atom:entry", ATOM_NS)
        if not entries:
            break
        for entry in entries:
            id_url = entry.find("atom:id", ATOM_NS).text
            arxiv_id = id_url.rsplit("/", 1)[-1]
            arxiv_id = re.sub(r"v\d+$", "", arxiv_id)
            if arxiv_id in seen:
                continue
            seen.add(arxiv_id)
            title = entry.find("atom:title", ATOM_NS).text.strip().replace("\n", " ")
            results.append((arxiv_id, title))
            if len(results) >= target:
                break
        start += PAGE_SIZE
        time.sleep(3)  # arXiv API etiquette
    return results


def safe_filename(arxiv_id: str, title: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", title).strip("_")[:80]
    return f"{arxiv_id}_{slug}.pdf"


def download_pdfs(papers: list[tuple[str, str]], outdir: Path, delay: float) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for i, (arxiv_id, title) in enumerate(papers, 1):
        dest = outdir / safe_filename(arxiv_id, title)
        if dest.exists() and dest.stat().st_size > 0:
            saved.append(dest)
            continue
        url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            saved.append(dest)
            print(f"[{i}/{len(papers)}] OK  {arxiv_id}  {title[:60]}")
        except requests.RequestException as exc:
            print(f"[{i}/{len(papers)}] FAIL {arxiv_id}: {exc}", file=sys.stderr)
        time.sleep(delay)
    return saved


def make_zip(files: list[Path], zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            zf.write(f, arcname=f.name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--categories",
        default="cs.CL,cs.IR,cs.AI",
        help="Comma-separated arXiv categories (default: cs.CL,cs.IR,cs.AI)",
    )
    parser.add_argument("--count", type=int, default=600, help="Target number of PDFs (500-700 recommended)")
    parser.add_argument("--outdir", default="uploads/arxiv_test_corpus", help="Directory to save PDFs into")
    parser.add_argument("--zip-path", default="uploads/arxiv_test_corpus.zip", help="Output zip archive path")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay in seconds between PDF downloads")
    args = parser.parse_args()

    categories = [c.strip() for c in args.categories.split(",") if c.strip()]
    print(f"Listing ~{args.count} papers from categories: {categories}")
    papers = list_ids(categories, args.count)
    print(f"Got {len(papers)} candidate papers, downloading PDFs...")

    files = download_pdfs(papers, Path(args.outdir), args.delay)
    print(f"Downloaded {len(files)} PDFs into {args.outdir}")

    make_zip(files, Path(args.zip_path))
    print(f"Zipped {len(files)} PDFs into {args.zip_path}")


if __name__ == "__main__":
    main()
