#!/usr/bin/env python3
"""Build a frozen, heterogeneous PDF corpus for cross-domain RAG evaluation.

The first run discovers candidates from public sources and freezes the exact
selection in ``selection.json``. Later runs reuse that selection rather than
silently sampling a newer source listing. Downloads are resumable and every
accepted file is checked for a PDF signature, size/page limits, and SHA-256.

The resulting PDFs are intentionally gitignored. Commit the small selection
and manifest files so another person can reproduce the corpus from its URLs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import quote, urlencode, urljoin
from xml.etree import ElementTree as ET

import httpx

try:
    import fitz
except ImportError:  # Discovery/download still works outside the app venv.
    fitz = None


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "eval" / "mixed_corpus"
USER_AGENT = "rag-knowledge-system-mixed-corpus/1.0 (local research evaluation)"
ARXIV_API = "https://export.arxiv.org/api/query"
HF_TREE = "https://huggingface.co/api/datasets/allenai/olmOCR-bench/tree/main/bench_data/pdfs"
HF_RESOLVE = "https://huggingface.co/datasets/allenai/olmOCR-bench/resolve/main"
DAILYMED_API = "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls.json"
DAILYMED_PDF = "https://dailymed.nlm.nih.gov/dailymed/downloadpdffile.cfm"
FAA_PAGES = (
    "https://www.faa.gov/regulations_policies/handbooks_manuals/aviation",
    "https://www.faa.gov/pilots/safety/pilotsafetybrochures",
)
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


@dataclass(frozen=True)
class Candidate:
    source: str
    category: str
    source_id: str
    title: str
    url: str
    relative_path: str
    license_note: str
    split: str


def stable_seed(seed: int, value: str) -> int:
    digest = hashlib.sha256(f"{seed}:{value}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def safe_name(value: str, limit: int = 100) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return (name[:limit] or "document").rstrip("._")


def split_for_index(index: int, total: int) -> str:
    # Each source/category is split independently so heldout is not
    # accidentally made up of only one easy document family.
    if index < round(total * 0.60):
        return "discovery"
    if index < round(total * 0.80):
        return "calibration"
    return "heldout"


def choose_group(items: list[dict], count: int, seed: int, group: str) -> list[tuple[dict, str]]:
    ordered = sorted(items, key=lambda x: str(x.get("source_id") or x.get("path") or x.get("url")))
    random.Random(stable_seed(seed, group)).shuffle(ordered)
    chosen = ordered[: min(count, len(ordered))]
    return [(item, split_for_index(i, len(chosen))) for i, item in enumerate(chosen)]


def get_json(client: httpx.Client, url: str, **params) -> dict | list:
    response = client.get(url, params=params or None)
    response.raise_for_status()
    return response.json()


def discover_arxiv(client: httpx.Client, count: int, seed: int) -> list[Candidate]:
    if count <= 0:
        return []
    params = {
        "search_query": "cat:cs.CL OR cat:cs.IR OR cat:cs.AI",
        "start": 0,
        "max_results": max(100, count * 2),
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    response = client.get(f"{ARXIV_API}?{urlencode(params)}")
    response.raise_for_status()
    root = ET.fromstring(response.content)
    raw = []
    for entry in root.findall("atom:entry", ATOM_NS):
        id_node = entry.find("atom:id", ATOM_NS)
        title_node = entry.find("atom:title", ATOM_NS)
        if id_node is None or title_node is None:
            continue
        arxiv_id = re.sub(r"v\d+$", "", id_node.text.rsplit("/", 1)[-1])
        title = " ".join(title_node.text.split())
        raw.append({"source_id": arxiv_id, "title": title})
    result = []
    for item, split in choose_group(raw, count, seed, "arxiv"):
        filename = f"arxiv_{safe_name(item['source_id'])}_{safe_name(item['title'], 72)}.pdf"
        result.append(Candidate(
            source="arxiv", category="scientific", source_id=item["source_id"],
            title=item["title"], url=f"https://arxiv.org/pdf/{quote(item['source_id'])}.pdf",
            relative_path=f"documents/scientific/{filename}",
            license_note="Article-specific arXiv license; local evaluation only; do not redistribute blindly.",
            split=split,
        ))
    return result


def discover_dailymed(client: httpx.Client, count: int, seed: int) -> list[Candidate]:
    if count <= 0:
        return []
    # Pull more than needed so seeded selection is varied rather than just
    # mirroring the most recent publication page.
    data = get_json(client, DAILYMED_API, page=1, pagesize=max(100, count * 4))
    raw = []
    for item in data.get("data", []):
        setid = item.get("setid")
        if setid:
            raw.append({"source_id": setid, "title": item.get("title", setid)})
    result = []
    for item, split in choose_group(raw, count, seed, "dailymed"):
        filename = f"dailymed_{safe_name(item['source_id'])}_{safe_name(item['title'], 64)}.pdf"
        result.append(Candidate(
            source="dailymed", category="medical", source_id=item["source_id"],
            title=item["title"], url=f"{DAILYMED_PDF}?setId={quote(item['source_id'])}",
            relative_path=f"documents/medical/{filename}",
            license_note="Official NLM access; label content may have source-specific rights. Local evaluation only.",
            split=split,
        ))
    return result


# The FAA pilot-safety-brochures page mixes English originals with
# Spanish/Portuguese/Turkish translations under the same listing, no
# separate index — ~40% of that page's PDFs are non-English (confirmed by
# fetching it directly: "Span_Fatigue.pdf", "Turkish_SpatialD.pdf",
# "Hypoxia_Portuguese_BR.pdf", "span_hypoxia.pdf", ...). This corpus is
# English-only; filtering by href/filename keyword here is what actually
# caught these — a stopword-ratio check on downloaded content (see
# eval/mixed_corpus/probe_extraction.py) found six that slipped through
# before this filter existed, and still missed "Turkish_SpatialD.pdf"
# outright since Turkish shares no marker words with the Spanish/
# Portuguese hint list used there. Filtering at discovery time is strictly
# more reliable than catching it after the fact.
_NON_ENGLISH_HREF_RE = re.compile(
    r"portuguese|spanish|turkish|\bspan_|_span\b|_es[_.]|_pt[_.]|_tr[_.]|_br[_.]"
    r"|french|_fr[_.]|german|_de[_.]|russian|chinese|korean|japanese|vietnamese|arabic",
    re.I,
)


def discover_faa(client: httpx.Client, count: int, seed: int) -> list[Candidate]:
    if count <= 0:
        return []
    found: dict[str, dict] = {}
    for page_url in FAA_PAGES:
        response = client.get(page_url)
        response.raise_for_status()
        for href in re.findall(r"href=[\"']([^\"']+\.pdf(?:\?[^\"']*)?)[\"']", response.text, re.I):
            if _NON_ENGLISH_HREF_RE.search(href):
                continue
            url = urljoin(page_url, href.replace("&amp;", "&"))
            source_id = url.split("?", 1)[0].rsplit("/", 1)[-1]
            found[url] = {"source_id": source_id, "title": source_id, "url": url}
    result = []
    for item, split in choose_group(list(found.values()), count, seed, "faa"):
        filename = f"faa_{safe_name(item['source_id'])}"
        if not filename.lower().endswith(".pdf"):
            filename += ".pdf"
        result.append(Candidate(
            source="faa", category="manuals", source_id=item["source_id"],
            title=item["title"], url=item["url"], relative_path=f"documents/manuals/{filename}",
            license_note="Official U.S. FAA source; verify any third-party embedded material before redistribution.",
            split=split,
        ))
    return result


def discover_olmocr(client: httpx.Client, counts: dict[str, int], seed: int) -> list[Candidate]:
    result = []
    for category, count in counts.items():
        if count <= 0:
            continue
        data = get_json(client, f"{HF_TREE}/{category}", recursive="true", expand="false", limit=1000)
        if not isinstance(data, list):
            raise RuntimeError(f"Unexpected Hugging Face response for {category}: {data}")
        raw = [
            {"source_id": item["path"], "path": item["path"], "title": Path(item["path"]).name}
            for item in data
            if item.get("type") == "file" and item.get("path", "").lower().endswith(".pdf")
        ]
        for item, split in choose_group(raw, count, seed, f"olmocr:{category}"):
            filename = f"olmocr_{category}_{safe_name(Path(item['path']).name)}"
            result.append(Candidate(
                source="olmocr", category=f"hard_{category}", source_id=item["source_id"],
                title=item["title"], url=f"{HF_RESOLVE}/{quote(item['path'], safe='/')}",
                relative_path=f"documents/hard/{category}/{filename}",
                license_note="olmOCR-bench dataset (ODC-BY); preserve attribution and source metadata.",
                split=split,
            ))
    return result


def discover(args, client: httpx.Client) -> list[Candidate]:
    candidates = []
    candidates.extend(discover_arxiv(client, args.arxiv, args.seed))
    candidates.extend(discover_dailymed(client, args.medical, args.seed))
    candidates.extend(discover_faa(client, args.manuals, args.seed))
    hard_counts = {
        "old_scans": args.old_scans,
        "old_scans_math": args.old_scans_math,
        "headers_footers": args.headers_footers,
        "multi_column": args.multi_column,
        "tables": args.tables,
        "long_tiny_text": args.tiny_text,
    }
    candidates.extend(discover_olmocr(client, hard_counts, args.seed))
    return candidates


def load_or_create_selection(args, client: httpx.Client, output: Path) -> list[Candidate]:
    path = output / "selection.json"
    if path.exists() and not args.refresh_selection:
        rows = json.loads(path.read_text(encoding="utf-8"))
        print(f"Loaded frozen selection: {len(rows)} candidates from {path}")
        return [Candidate(**row) for row in rows]
    candidates = discover(args, client)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(x) for x in candidates], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Frozen selection: {len(candidates)} candidates in {path}")
    return candidates


def inspect_pdf(path: Path, max_pages: int) -> tuple[int | None, str | None]:
    if fitz is None:
        return None, None
    try:
        with fitz.open(path) as document:
            pages = len(document)
        if pages > max_pages:
            return pages, f"page limit exceeded ({pages} > {max_pages})"
        return pages, None
    except Exception as exc:
        return None, f"invalid PDF: {exc}"


def download_one(client: httpx.Client, candidate: Candidate, output: Path, max_bytes: int, max_pages: int) -> dict:
    destination = output / candidate.relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 4:
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        pages, error = inspect_pdf(destination, max_pages)
        if destination.read_bytes()[:5] == b"%PDF-" and not error:
            return {**asdict(candidate), "status": "existing", "bytes": destination.stat().st_size,
                    "pages": pages, "sha256": digest}

    part = destination.with_suffix(destination.suffix + ".part")
    digest = hashlib.sha256()
    size = 0
    try:
        with client.stream("GET", candidate.url) as response:
            response.raise_for_status()
            content_length = int(response.headers.get("content-length", "0") or 0)
            if content_length > max_bytes:
                raise ValueError(f"content-length {content_length} exceeds {max_bytes}")
            with part.open("wb") as handle:
                for chunk in response.iter_bytes(1024 * 1024):
                    size += len(chunk)
                    if size > max_bytes:
                        raise ValueError(f"download exceeds {max_bytes} bytes")
                    handle.write(chunk)
                    digest.update(chunk)
        with part.open("rb") as handle:
            if handle.read(5) != b"%PDF-":
                raise ValueError("response is not a PDF")
        pages, error = inspect_pdf(part, max_pages)
        if error:
            raise ValueError(error)
        part.replace(destination)
        return {**asdict(candidate), "status": "downloaded", "bytes": size,
                "pages": pages, "sha256": digest.hexdigest()}
    except Exception as exc:
        part.unlink(missing_ok=True)
        return {**asdict(candidate), "status": "failed", "error": str(exc)[:500]}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--arxiv", type=int, default=80)
    parser.add_argument("--medical", type=int, default=45)
    parser.add_argument("--manuals", type=int, default=25)
    parser.add_argument("--old-scans", type=int, default=30)
    parser.add_argument("--old-scans-math", type=int, default=10)
    parser.add_argument("--headers-footers", type=int, default=15)
    parser.add_argument("--multi-column", type=int, default=20)
    parser.add_argument("--tables", type=int, default=20)
    parser.add_argument("--tiny-text", type=int, default=10)
    parser.add_argument("--max-file-mb", type=int, default=50)
    parser.add_argument("--max-pages", type=int, default=500)
    parser.add_argument("--delay", type=float, default=0.35)
    parser.add_argument("--refresh-selection", action="store_true",
                        help="Discard reproducibility and discover a new source selection")
    parser.add_argument("--selection-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if fitz is None and not args.selection_only:
        print(
            "PyMuPDF is required for PDF/page-limit validation. "
            "Activate the project venv before downloading: source venv/bin/activate",
            file=sys.stderr,
        )
        return 2
    timeout = httpx.Timeout(90.0, connect=30.0)
    headers = {"User-Agent": USER_AGENT}
    with httpx.Client(headers=headers, timeout=timeout, follow_redirects=True) as client:
        candidates = load_or_create_selection(args, client, args.output)
        if args.selection_only:
            return 0
        rows = []
        for index, candidate in enumerate(candidates, 1):
            row = download_one(client, candidate, args.output, args.max_file_mb * 1024 * 1024, args.max_pages)
            rows.append(row)
            marker = "OK" if row["status"] != "failed" else "FAIL"
            detail = f"{row.get('bytes', 0) / 1024 / 1024:.1f} MB" if marker == "OK" else row.get("error", "")
            print(f"[{index:03d}/{len(candidates):03d}] {marker:4s} {candidate.category:22s} {detail}  {candidate.title[:55]}", flush=True)
            time.sleep(args.delay)

    successful = [row for row in rows if row["status"] != "failed"]
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(successful, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.output / "download_report.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    counts: dict[str, int] = {}
    splits: dict[str, int] = {}
    for row in successful:
        counts[row["category"]] = counts.get(row["category"], 0) + 1
        splits[row["split"]] = splits.get(row["split"], 0) + 1
    total_bytes = sum(row.get("bytes", 0) for row in successful)
    print(f"\nAccepted {len(successful)}/{len(rows)} PDFs ({total_bytes / 1024**3:.2f} GiB)")
    print(f"Categories: {json.dumps(counts, sort_keys=True)}")
    print(f"Splits: {json.dumps(splits, sort_keys=True)}")
    print(f"Manifest: {manifest_path}")
    return 0 if successful else 1


if __name__ == "__main__":
    sys.exit(main())
