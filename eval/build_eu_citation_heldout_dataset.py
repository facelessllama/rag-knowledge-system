"""
Builds eval/eu_citation_heldout_dataset.json — a genuinely untouched
held-out set for the natural short-citation lookup path (ingestion/
chunker.py::extract_citation_number, rag/retriever.py::
extract_citation_numbers / chunks_by_citation_number / identity_match
promotion).

Why this exists: eu_id_free_dataset.json's natural_lookup_by_number cases
(see build_eu_id_free_dataset.py) are the *same* 20 questions whose
failures originally motivated the citation-number index fix (see
eval/README.md, "ID-free spot-check @ 57,000"). Re-running the fix against
the questions that found the bug proves the fix works on those questions —
it doesn't demonstrate the fix generalizes. This script draws a disjoint
set of documents (excluding every celex_id already used anywhere in
eu_golden_dataset.json / eu_heldout_dataset.json / eu_cross_question_dataset.json
/ eu_id_free_dataset.json) and phrases the citation five different natural
ways, so the result actually tests generalization across both new
documents and new query phrasing rather than one fixed template.

Every candidate's (citation_number, citation_year) pair is required to be
unique across the full 57,000-document manifest — otherwise "What does
Regulation No X/Y concern?" wouldn't have a single correct answer, the
same ambiguity class as the hard_negative_date fix in
build_eu_id_free_dataset.py.

Usage:
    python eval/build_eu_citation_heldout_dataset.py
"""
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from ingestion.chunker import extract_citation_number  # noqa: E402
from build_eu_golden_dataset import MANIFEST_PATH, ADVERSARIAL_EU  # noqa: E402

_FILENAME_RE = re.compile(r"^(\S+)\s*-\s*(.+)$")

N_PER_FORMAT = 10

USED_DATASETS = [
    "eu_golden_dataset.json",
    "eu_heldout_dataset.json",
    "eu_cross_question_dataset.json",
    "eu_id_free_dataset.json",
]


def already_used_celex_ids() -> set[str]:
    used = set()
    for name in USED_DATASETS:
        path = EVAL_DIR / name
        if not path.exists():
            continue
        for case in json.loads(path.read_text()):
            fname = case.get("expected_doc_filename")
            if not fname:
                continue
            m = _FILENAME_RE.match(Path(fname).stem)
            if m:
                used.add(m.group(1))
    return used


# Five distinct natural phrasings a real user might type. All satisfy
# rag/retriever.py::_CITATION_NUMBER_QUERY_RE ("No" + number + "/" + year)
# so each exercises the same index/promotion mechanism through different
# surface form — the point of "several natural citation formats" (see
# module docstring) rather than re-testing one fixed template on new docs.
# The last format deliberately uses the 2-digit year to exercise
# extract_citation_numbers()'s year-expansion path directly.
def _formats(num: str, year: str):
    yy = year[-2:]
    return [
        ("citation_fmt_bracket_eec", f"What does Regulation (EEC) No {num}/{year} concern?"),
        ("citation_fmt_council_no_dot", f"What is Council Regulation No. {num}/{year} about?"),
        ("citation_fmt_commission_summarize", f"Summarize Commission Regulation (EC) No {num}/{year}."),
        ("citation_fmt_bare_no", f"What does No {num}/{year} deal with?"),
        ("citation_fmt_two_digit_year", f"What is the subject of Regulation No {num}/{yy} of the European Communities?"),
    ]


def main():
    manifest = json.loads(MANIFEST_PATH.read_text())
    used_celex = already_used_celex_ids()
    print(f"already-used celex_ids across existing EU datasets: {len(used_celex)}")

    # (num, year) -> filenames, over the WHOLE corpus, to guarantee each
    # picked citation uniquely identifies one document.
    citation_index: dict[tuple[str, str], list[str]] = {}
    doc_celex: dict[str, str] = {}
    for filename in manifest:
        m = _FILENAME_RE.match(Path(filename).stem)
        if not m:
            continue
        celex_id = m.group(1)
        doc_celex[filename] = celex_id
        meta = extract_citation_number(filename)
        if not meta:
            continue
        key = (meta["citation_number"], meta["citation_year"])
        citation_index.setdefault(key, []).append(filename)

    candidates = []
    for filename, celex_id in doc_celex.items():
        if celex_id in used_celex:
            continue
        meta = extract_citation_number(filename)
        if not meta:
            continue
        key = (meta["citation_number"], meta["citation_year"])
        if len(citation_index[key]) != 1:
            continue  # ambiguous short citation, same class of bug as hard_negative_date
        year = meta["citation_year"]
        if not (1930 <= int(year) <= 2029):
            continue  # 2-digit-year format only round-trips in this range (see retriever.py expansion)
        candidates.append((filename, meta["citation_number"], year))

    print(f"eligible untouched, unambiguous, year-round-trippable candidates: {len(candidates)}")

    dataset = []
    fmt_names = [f[0] for f in _formats("0", "2000")]
    per_format_pool = {name: [] for name in fmt_names}
    for i, (filename, num, year) in enumerate(candidates):
        fmt_name, question = _formats(num, year)[i % len(fmt_names)]
        if len(per_format_pool[fmt_name]) >= N_PER_FORMAT:
            continue
        per_format_pool[fmt_name].append(filename)
        dataset.append({
            "id": f"cho_{doc_celex[filename]}_{fmt_name}",
            "question": question,
            "expect_answer": True,
            "expected_doc_filename": filename,
            "expected_substring": None,  # free-form — Recall@K only, like natural_lookup_by_number
            "type": fmt_name,
        })
        if all(len(v) >= N_PER_FORMAT for v in per_format_pool.values()):
            break

    out_path = EVAL_DIR / "eu_citation_heldout_dataset.json"
    out_path.write_text(json.dumps(dataset, indent=2, ensure_ascii=False))
    for name in fmt_names:
        print(f"{name}: {len(per_format_pool[name])}")
    print(f"TOTAL: {len(dataset)} cases -> {out_path}")


if __name__ == "__main__":
    main()
