#!/usr/bin/env python3
"""Generates extraction_verdict.md from extraction_probe.json + manifest.json.

Regenerate this after any change to the corpus or a probe_extraction.py
re-run, rather than hand-editing the verdict — it's derived data, and by the
same discipline as the rest of eval/ (facts extracted from source, not
hand-transcribed), it should never drift from what the probe actually found.

Usage:
    python eval/mixed_corpus/generate_verdict.py
"""
import json
from pathlib import Path

CORPUS_DIR = Path(__file__).resolve().parent
probe = json.loads((CORPUS_DIR / "extraction_probe.json").read_text(encoding="utf-8"))
manifest = {m["relative_path"]: m for m in json.loads((CORPUS_DIR / "manifest.json").read_text(encoding="utf-8"))}

reject = [r for r in probe if r["predicted_status"] == "would_reject_no_text"]
error = [r for r in probe if r["predicted_status"] == "would_error"]
garbage_hard = [r for r in probe if r["quality_flag"] == "likely_garbage" and r["category"].startswith("hard_")]
garbage_other = [r for r in probe if r["quality_flag"] == "likely_garbage" and not r["category"].startswith("hard_")]
too_short = [r for r in probe if r["quality_flag"] == "too_short_to_judge"]
clean = [r for r in probe if r["quality_flag"] == "likely_ok"]

excluded_count = len(reject) + len(error) + len(garbage_hard)
usable_for_questions = len(probe) - excluded_count

lines = []
lines.append("# Extraction verdict — which documents will actually make it in\n")
lines.append(f"Generated from `extraction_probe.json` (762 documents, same "
              f"`PDFParser`/`SmartChunker` code path as the live `/upload` API — "
              f"see README.md's \"Extraction/chunking probe\" section for methodology "
              f"and the medical false-positive caveat). Regenerate via "
              f"`generate_verdict.py`, don't hand-edit.\n")

lines.append("## Bottom line\n")
lines.append(f"- **{len(probe) - len(reject) - len(error)} / {len(probe)}** will be accepted by a real "
              f"`/upload` call (produce ≥1 chunk).")
lines.append(f"- **{len(reject) + len(error)} / {len(probe)}** will be rejected outright "
              f"(HTTP 422 \"Could not extract text from document\" or a parse error) — "
              f"these never enter the index, there is no chunk to almost-retrieve, don't "
              f"build questions expecting them to be findable.")
lines.append(f"- Of the accepted ones, **{len(garbage_hard)}** more (all in `hard/*`) read as OCR noise, "
              f"not real text — technically indexed, but building a fact-based question against "
              f"them would test Tesseract's inability to read handwriting/dense formulas, not "
              f"the RAG system. Exclude from golden questions; keep as documented OCR-limitation cases.")
lines.append(f"- **{usable_for_questions} / {len(probe)} ({100*usable_for_questions/len(probe):.1f}%)** "
              f"is the number to build fact-based golden questions against.")
lines.append(f"- {len(garbage_other)} more (`medical`, all short structured DailyMed labels) hit the "
              f"same OCR-noise heuristic but were spot-checked and are **real, legible English "
              f"text** — a genre false positive (low connective-word density is normal for "
              f"ingredient/warning lists). Not excluded from the usable count above; listed "
              f"below only so a manual skim can double-check the rest.")
lines.append(f"- {len(too_short)} documents have too little extracted text (<15 word-tokens) for "
              f"the stopword heuristic to judge either way — manual look recommended.\n")

def table(rows, cols):
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    lines.append("")

lines.append(f"## Will NOT be ingested ({len(reject) + len(error)})\n")
lines.append("Real `/upload` raises `ValueError(\"Could not extract text from document\")` → "
              "HTTP 422 → rolled back, never stored. These simply won't exist in the index; "
              "asking a question that expects one of them to be findable is testing nothing.\n")
rows = [{"file": Path(r["relative_path"]).name, "category": r["category"],
         "chars": r.get("total_chars", "-"), "reason": r.get("error", "0 chunks (all pages < 100 chars)")[:60]}
        for r in reject + error]
table(sorted(rows, key=lambda r: r["category"]), ["category", "file", "chars", "reason"])

lines.append(f"## Ingests, but reads as OCR noise — exclude from golden questions ({len(garbage_hard)})\n")
lines.append("Clears the chunker's 100-char gate, so it *will* sit in the index, but "
              "English-stopword ratio < 0.12 says it's garbled recognition output, not "
              "readable text — verified by hand against raw OCR text on several of these "
              "(pure noise, e.g. `old_scans/43.pdf`: `\"| a ariel\\nes ae i ee AC gn...\"`). "
              "All from the olmOCR-bench `hard/*` slices, i.e. exactly the adversarial-scan "
              "content this corpus deliberately includes to find this failure mode.\n")
rows = [{"file": Path(r["relative_path"]).name, "category": r["category"],
         "chars": r["total_chars"], "stopword_ratio": r["stopword_ratio"], "tokens": r["token_count"]}
        for r in garbage_hard]
table(sorted(rows, key=lambda r: (r["category"], r["stopword_ratio"])),
      ["category", "file", "chars", "stopword_ratio", "tokens"])

lines.append(f"## Flagged by the same heuristic, but a confirmed false positive — keep ({len(garbage_other)})\n")
lines.append("All `medical` (DailyMed drug labels): short, structured, born-digital English "
              "(ingredient lists, dosage tables, warnings) with naturally few connective "
              "words. Spot-checked `COLGATE_KIDS...pdf` (stopword_ratio 0.101) by hand — "
              "perfectly legible, correctly extracted \"Drug Facts\" label text. Trust this "
              "list less than the `hard/*` one above; a quick skim of the rest (13 documents, "
              "cheap) is worth doing before building questions from them, but don't drop them "
              "on the heuristic's say-so alone.\n")
rows = [{"file": Path(r["relative_path"]).name, "chars": r["total_chars"],
         "stopword_ratio": r["stopword_ratio"], "tokens": r["token_count"]}
        for r in garbage_other]
table(sorted(rows, key=lambda r: r["stopword_ratio"]), ["file", "chars", "stopword_ratio", "tokens"])

if too_short:
    lines.append(f"## Too little text for the heuristic to judge ({len(too_short)})\n")
    rows = [{"file": Path(r["relative_path"]).name, "category": r["category"],
             "chars": r["total_chars"], "tokens": r["token_count"]} for r in too_short]
    table(rows, ["category", "file", "chars", "tokens"])

out_path = CORPUS_DIR / "extraction_verdict.md"
out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"Wrote {out_path} ({len(probe)} documents: {len(reject)+len(error)} rejected, "
      f"{len(garbage_hard)} noise, {len(garbage_other)} false-positive-flagged, "
      f"{usable_for_questions} usable)")
