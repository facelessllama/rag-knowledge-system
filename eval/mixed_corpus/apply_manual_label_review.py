#!/usr/bin/env python3
"""Applies a one-time MANUAL review of the 17 suspiciously-long
(>300 char) "verified" fact_figure_caption/fact_table_caption cases
found while reading the post-fix A/B's all-three-wrong sample (see
project memory / EXPERIMENT_HISTORY.md). Per the user's explicit
instruction, this is NOT a builder-heuristic change and does not touch
retrieval or any saved model answer — it only relabels label_status on
golden_dataset_v3.json based on a human reading each case's expected_
substring against real source-PDF context, moving each into:

- "verified_reviewed" — read by hand, genuinely a real (if long/dense)
  caption, no leaked table/chart data or misattribution found.
- "ambiguous" — read by hand, genuinely corrupted (leaked table/chart
  content with no period to trigger the v3 split) or, in one case,
  outright misattributed to a plain in-text reference rather than the
  real caption.

Every decision's reasoning is recorded in REVIEW_NOTES below and written
into the case's own `manual_review_note` field, not just applied
silently.

Usage:
    python eval/mixed_corpus/apply_manual_label_review.py \\
        --dataset eval/mixed_corpus/golden_dataset_v3.json
"""
import argparse
import json
from pathlib import Path

CORPUS_DIR = Path(__file__).resolve().parent

# id -> (new_label_status, reason)
REVIEW_NOTES = {
    "sci_caption_2607.13655_Figure14": (
        "ambiguous",
        "Chart axis tick marks + legend items repeated 4x ('105 106 steps "
        "0.4 0.6 0.8 1.0 distance action move_left move_right...') with no "
        "period anywhere — the v3 no-period blind spot, confirmed by hand.",
    ),
    "sci_caption_2607.12383_Figure3": (
        "verified_reviewed",
        "Genuine two-sentence caption: description + a real explanatory "
        "continuation about the 6%/7,949-post finding. No leaked table data.",
    ),
    "sci_caption_2607.12823_Figure19": (
        "verified_reviewed",
        "Genuine flowing prose describing the figure's narrative example "
        "(the 'hospital case, Emma'). No leaked table/chart content.",
    ),
    "sci_caption_2607.11433_Figure3": (
        "ambiguous",
        "Text describes an APPENDIX's general contents ('This appendix "
        "provides the annotation procedure...'), not what Figure 3 itself "
        "shows — looks like a misattributed span, not table-cell leakage.",
    ),
    "sci_caption_2607.14782_Figure12": (
        "ambiguous",
        "Report boilerplate/running-header furniture leaked in mid-span "
        "('2ND EDITION REPORT ℓ 50 Source: Global Index on Responsible "
        "AI (GIRAI) Global Survey, 2026.') between two genuine caption "
        "sentences.",
    ),
    "sci_caption_2607.13501_Figure6": (
        "verified_reviewed",
        "Genuine flowing technical prose describing a specific example "
        "case shown in the figure. No leaked table/chart content.",
    ),
    "sci_caption_2607.09403_Table19": (
        "ambiguous",
        "Raw table header + full data rows glued directly onto a title-"
        "like heading with no period at all — the v3 no-period blind spot.",
    ),
    "sci_caption_2607.13104_Table5": (
        "ambiguous",
        "A bare list of taxonomy category/subcategory names with no "
        "connecting prose at all — leaked table content, no period.",
    ),
    "sci_caption_2607.14138_Figure1": (
        "verified_reviewed",
        "Genuine (dense, multi-panel) prose describing the figure's "
        "panels and UI elements. No leaked table/chart content.",
    ),
    "sci_caption_2607.10880_Figure10": (
        "verified_reviewed",
        "Genuine technical prose describing a formal-methods figure's "
        "notation and line references. Structured but coherent, real "
        "sentence connectives throughout.",
    ),
    "sci_caption_2607.14157_Table6": (
        "verified_reviewed",
        "Genuine technical prose explaining the table's statistical "
        "semantics. No leaked raw table data.",
    ),
    "sci_caption_2607.11981_Table3": (
        "ambiguous",
        "Raw table headers + numeric sweep rows ('1 task x 1 encoder x 1 "
        "head 1 .315 .311...') with no period — the v3 no-period blind spot.",
    ),
    "sci_caption_2607.13123_Table1": (
        "ambiguous",
        "Raw table headers + citation-style data rows with no period — "
        "the v3 no-period blind spot.",
    ),
    "sci_caption_2607.09641_Table1": (
        "ambiguous",
        "The originally-discovered real bug that motivated flagging this "
        "whole class: raw table cell values with no period ('...XGBoost "
        "(Raw Tabular) 28833 1 1389 0 0.000 0.000 0.000...').",
    ),
    "sci_caption_2607.15238_Table3": (
        "ambiguous",
        "Raw grid-search table data AND bleeds into an entirely unrelated "
        "following section ('4.3 Distance-Based Classification...') — "
        "double contamination, no period to stop either.",
    ),
    "sci_caption_2607.15968_Table7": (
        "ambiguous",
        "Confirmed against the real source PDF: this is a plain in-text "
        "reference ('The results obtained differ from these of Table 7. "
        "Now the low pWork of the...') wrongly resolved as a caption "
        "opener because 'Now' is not in the discourse-connective rejection "
        "list — a genuine misattribution, not just trailing contamination. "
        "Worth adding 'now' to _STRUCTURAL_DISCOURSE_CONNECTIVES /"
        "_DISCOURSE_CONNECTIVES in a future pass; not done here.",
    ),
    "sci_caption_2607.12659_Figure10": (
        "verified_reviewed",
        "Genuine technical prose explaining notation (T, T', dVLM, dAE). "
        "No leaked table/chart content.",
    ),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=str(CORPUS_DIR / "golden_dataset_v3.json"))
    args = ap.parse_args()

    path = Path(args.dataset)
    cases = json.loads(path.read_text(encoding="utf-8"))
    by_id = {c["id"]: c for c in cases}

    applied = []
    for cid, (new_status, reason) in REVIEW_NOTES.items():
        case = by_id.get(cid)
        if case is None:
            print(f"WARNING: {cid} not found in {path}, skipping")
            continue
        old_status = case.get("label_status")
        case["label_status"] = new_status
        case["manual_review_note"] = reason
        applied.append({"id": cid, "old_label_status": old_status, "new_label_status": new_status})

    path.write_text(json.dumps(cases, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    from collections import Counter
    counts = Counter(c.get("label_status") for c in cases if c["type"] in ("fact_figure_caption", "fact_table_caption"))
    print(f"Applied {len(applied)} manual relabels to {path}")
    print(f"New label_status breakdown: {dict(counts)}")
    for a in applied:
        print(f"  {a['id']}: {a['old_label_status']} -> {a['new_label_status']}")


if __name__ == "__main__":
    main()
