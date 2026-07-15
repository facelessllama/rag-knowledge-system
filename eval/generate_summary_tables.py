"""
Regenerates every markdown table quoted in VALIDATION.md / eval/README.md
directly from eval/scale_results/*.json — the raw per-case data those
documents summarize. Exists so a future change to the underlying results
can never again silently drift from the hand-typed tables describing them
(see "Recall@5 counting bug" in VALIDATION.md for what happened the one
time this wasn't true: the docs eventually needed a full correction pass
because the tables were typed once and never re-derived).

This script only reads already-saved results — it does not talk to a live
instance. Re-run eval/run_eval.py (or the relevant build_eu_*.py + a live
rerun) first if the underlying data itself needs to change; then re-run
this script to regenerate the tables from the new files.

Usage:
    python eval/generate_summary_tables.py
"""
import json
import re
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
SCALE_DIR = EVAL_DIR / "scale_results"
TOP_K = 5

_FILENAME_RE = re.compile(r"^(\S+)\s*-\s*(.+)$")
_ISSUER_RE = re.compile(r"\b(Council|Commission)\s+(Regulation|Directive|Decision)\s*\((EEC|EC|EU)\)")


def load(name: str) -> list[dict]:
    return json.loads((SCALE_DIR / name).read_text())


def recall_stats(cases: list[dict], top_k: int = TOP_K) -> dict:
    recall_cases = [r for r in cases if r.get("recall_hit") is not None or "rank" in r]
    n = len(recall_cases)
    strict = sum(1 for r in recall_cases if r.get("rank") is not None and r["rank"] <= top_k)
    loose = sum(1 for r in recall_cases if r.get("rank") is not None)
    mrr = sum(r.get("reciprocal_rank", 0) or 0 for r in recall_cases) / n if n else 0.0
    return {"n": n, "strict": strict, "loose": loose, "mrr": mrr}


def abstention_stats(cases: list[dict]) -> tuple[int, int]:
    correct = sum(1 for r in cases if r.get("abstention_correct"))
    return correct, len(cases)


def pct(num: int, den: int) -> str:
    return f"{100 * num / den:.1f}%" if den else "n/a"


def print_ladder_table():
    print("## Scale ladder (strict Recall@5, rank <= 5)\n")
    print("| corpus size | golden strict R@5 | heldout strict R@5 | golden MRR | heldout MRR | golden abstention | heldout abstention |")
    print("|---:|---:|---:|---:|---:|---:|---:|")
    for size in (1000, 5000, 15000, 30000, 57000):
        g = load(f"golden_at_{size}.json")
        h = load(f"heldout_at_{size}.json")
        gr, hr = recall_stats(g), recall_stats(h)
        ga, gn = abstention_stats(g)
        ha, hn = abstention_stats(h)
        print(f"| {size:,} | {gr['strict']}/{gr['n']} ({pct(gr['strict'], gr['n'])}) "
              f"| {hr['strict']}/{hr['n']} ({pct(hr['strict'], hr['n'])}) "
              f"| {gr['mrr']:.3f} | {hr['mrr']:.3f} "
              f"| {pct(ga, gn)} | {pct(ha, hn)} |")
    print()


def print_id_free_table():
    print("## ID-free spot-check @ 57,000 (strict Recall@5)\n")
    for label, fname in [("before citation-number index fix", "id_free_at_57000.json"),
                          ("after citation-number index fix", "id_free_at_57000_post_citation_index.json")]:
        print(f"**{label}** (`{fname}`):\n")
        print("| type | n | strict Recall@5 | MRR | substring correct |")
        print("|---|---:|---:|---:|---:|")
        cases = load(fname)
        by_type: dict[str, list[dict]] = {}
        for r in cases:
            by_type.setdefault(r["type"], []).append(r)
        for t, rs in by_type.items():
            stats = recall_stats(rs)
            checkable = [r for r in rs if r.get("substring_correct") is not None]
            correct = sum(1 for r in checkable if r["substring_correct"])
            substr = f"{correct}/{len(checkable)}" if checkable else "n/a"
            if stats["n"]:
                print(f"| {t} | {stats['n']} | {stats['strict']}/{stats['n']} ({pct(stats['strict'], stats['n'])}) "
                      f"| {stats['mrr']:.3f} | {substr} |")
        overall = recall_stats(cases)
        print(f"| **overall** | {overall['n']} | **{overall['strict']}/{overall['n']} "
              f"({pct(overall['strict'], overall['n'])})** | **{overall['mrr']:.3f}** | |")
        print()


def print_citation_heldout_table():
    print("## Citation-number generalization test @ 57,000 (strict Recall@5)\n")
    dataset = {c["id"]: c for c in json.loads((EVAL_DIR / "eu_citation_heldout_dataset.json").read_text())}
    results = load("citation_heldout_at_57000.json")

    print("| format | n | strict Recall@5 | MRR |")
    print("|---|---:|---:|---:|")
    by_type: dict[str, list[dict]] = {}
    for r in results:
        by_type.setdefault(r["type"], []).append(r)
    for t, rs in by_type.items():
        stats = recall_stats(rs)
        print(f"| {t} | {stats['n']} | {stats['strict']}/{stats['n']} ({pct(stats['strict'], stats['n'])}) "
              f"| {stats['mrr']:.3f} |")
    overall = recall_stats(results)
    print(f"| **overall** | {overall['n']} | **{overall['strict']}/{overall['n']} "
          f"({pct(overall['strict'], overall['n'])})** | **{overall['mrr']:.3f}** |")
    print(f"| in-sources<=6 (not Recall@5) | {overall['n']} | {overall['loose']}/{overall['n']} "
          f"({pct(overall['loose'], overall['n'])}) | |")
    print()

    def claims(question: str):
        inst = "Council" if "Council" in question else ("Commission" if "Commission" in question else None)
        bracket = None
        for b in ("EEC", "EC", "EU"):
            if f"({b})" in question:
                bracket = b
                break
        return inst, bracket

    # citation_fmt_two_digit_year makes no institution/bracket claim either,
    # so a naive "no claim -> matched bucket" rule would silently lump it in
    # with citation_fmt_bare_no — but two_digit_year independently
    # underperforms (5/10, see table above) for an unrelated reason (the
    # query's literal "No N/YY" text has less overlap with the document's
    # own always-4-digit-year header than a 4-digit-year query does, even
    # though the year is resolved correctly at the lookup level — see
    # tests/test_retriever.py::test_extract_citation_numbers_expands_two_digit_year).
    # Folding it in would inflate the "matched" bucket's score with a
    # confound this split isn't measuring, so it's excluded here the same
    # way bare_no's "no claim, always worked" cases don't test this split.
    EXCLUDED_FROM_ISSUER_SPLIT = {"citation_fmt_bare_no", "citation_fmt_two_digit_year"}

    matched, mismatched = [], []
    for r in results:
        if r["type"] in EXCLUDED_FROM_ISSUER_SPLIT:
            continue
        case = dataset.get(r["id"])
        if not case:
            continue
        m = _FILENAME_RE.match(Path(case["expected_doc_filename"]).stem)
        actual = _ISSUER_RE.search(m.group(2)) if m else None
        if not actual:
            continue
        actual_inst, actual_bracket = actual.group(1), actual.group(3)
        q_inst, q_bracket = claims(case["question"])
        mismatch = (q_inst and q_inst != actual_inst) or (q_bracket and q_bracket != actual_bracket)
        (mismatched if mismatch else matched).append(r.get("rank"))

    def strict_n(ranks):
        return sum(1 for r in ranks if r is not None and r <= TOP_K)

    print("Issuer/bracket-word accuracy split (citation_fmt_bare_no and\n"
          "citation_fmt_two_digit_year excluded — neither makes an\n"
          "institution/bracket claim, and two_digit_year has its own\n"
          "separate confound; see comment in generate_summary_tables.py):\n")
    print("| | n | strict Recall@5 |")
    print("|---|---:|---:|")
    print(f"| query matches actual issuer/bracket | {len(matched)} | {strict_n(matched)}/{len(matched)} "
          f"({pct(strict_n(matched), len(matched))}) |")
    print(f"| query mismatches actual issuer/bracket | {len(mismatched)} | {strict_n(mismatched)}/{len(mismatched)} "
          f"({pct(strict_n(mismatched), len(mismatched))}) |")
    print()


if __name__ == "__main__":
    print_ladder_table()
    print_id_free_table()
    print_citation_heldout_table()
