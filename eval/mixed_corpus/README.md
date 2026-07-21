# Mixed PDF corpus

This corpus is the cross-domain and document-quality evaluation set. It is
separate from the legal golden/heldout sets and the EURLEX57K scale test.

The default build freezes roughly 250 public-source candidates across:

- scientific papers (arXiv);
- medical product labels (DailyMed/NLM);
- technical manuals (FAA);
- difficult pages from olmOCR-bench: old scans, mathematical scans, tables,
  multiple columns, repeated headers/footers, and very small text.

## Frozen corpus v2

Built on 2026-07-20 with seed `20260720` (`--arxiv 599 --refresh-selection`,
all other slices at v1's counts):

| Slice | PDFs |
|---|---:|
| Scientific | 597 |
| Medical | 45 |
| FAA manuals | 16 |
| Old scans | 30 |
| Old mathematical scans | 10 |
| Headers / footers | 15 |
| Multiple columns | 20 |
| Tables | 19 |
| Long tiny text | 10 |
| **Total** | **762** |

The frozen split is 457 discovery / 153 calibration / 152 heldout (~60/20/20),
stratified inside every source category. Five scientific candidates were
rejected (one 547-page overage, one 404) and three FAA manuals for exceeding
the 50 MiB upload limit; all remain recorded in the gitignored
`download_report.json`, not in `manifest.json`.

**Seven non-English documents were found and removed post-download**: the FAA
brochures page mixes English originals with Spanish/Portuguese/Turkish
translations under the same listing (`Span_Fatigue.pdf`,
`Turkish_SpatialD.pdf`, `Hypoxia_Portuguese_BR.pdf`, ...) with no separate
index — about 40% of that page's PDFs are non-English. `discover_faa()` in
`scripts/build_mixed_pdf_corpus.py` now filters these out by href keyword
before download, so a future `--refresh-selection` won't reintroduce them.
The 7 already-downloaded offenders (6 manuals + 1 `hard_tables` document,
caught by `probe_extraction.py`'s language heuristic, see below) were
removed from `documents/` and `manifest.json`; the removed entries are kept
for provenance in `excluded_non_english.json`.

### Extraction/chunking probe (`probe_extraction.py`)

Passing the chunker's minimum-length gate (100 chars) does not mean OCR read
a page correctly — a badly garbled scan routinely clears 100 characters of
pure noise and gets indexed anyway. `probe_extraction.py` runs the exact
same `ingestion.pdf_parser.PDFParser` + `ingestion.chunker.SmartChunker`
classes the live API uses, against every manifest document, without
touching Qdrant/Postgres/embeddings, and predicts three outcomes:

- **would_reject_no_text** (12/762, 1.6%) — 0 chunks, the API's own
  `if not chunks: raise ValueError(...)` would reject these on real upload.
  Concentrated in `hard_old_scans` (10/30) and one each in
  `hard_old_scans_math` / `hard_headers_footers`.
- **likely_garbage** (34/762, 4.5%) — clears the chunker gate but its
  English-stopword ratio (<0.12) reads as OCR noise, not real text.
  Concentrated in the olmOCR-bench `hard/*` categories (as expected — these
  are deliberately adversarial scans/handwriting/dense-formula pages plain
  Tesseract, with no math/handwriting model, isn't equipped to read).
  **This is a heuristic triage signal, not ground truth** — spot-checked
  against `medical` (13/45 flagged) and confirmed those are a **false-positive
  genre effect**: DailyMed drug labels are short, structured, born-digital
  English text (ingredient lists, warnings) that's perfectly legible but
  naturally low in connective stopwords. Do not exclude `medical`
  `likely_garbage` hits from golden questions without a manual look first;
  trust the flag much more on `hard/*`, where it was verified against raw
  OCR output on several documents (pure noise, e.g. `old_scans/43.pdf`:
  `"| a ariel\nes ae i ee AC gn..."`).
- **would_ingest / likely_ok** — everything else; safe to build fact-based
  golden questions from.

`predicted_status` and `quality_flag` are patched onto each `manifest.json`
entry so dataset-building code can filter without re-running the probe.
Per-document detail (chars, OCR pages, stopword ratio, token count) is in
the gitignored `extraction_probe.json`.

**Evaluation implication**: documents predicted `would_reject_no_text` won't
even be in the live index after a real ingestion run — they're not
"unanswerable questions," there's no chunk to almost-retrieve. Report
ingestion success/failure as its own metric, broken out by category, never
folded into retrieval/answer-accuracy numbers computed only over what
actually made it into the index.

## Build

```bash
source venv/bin/activate
python scripts/build_mixed_pdf_corpus.py
```

The first run writes `selection.json`; later runs reuse it and resume missing
downloads. Pass `--refresh-selection` only when intentionally defining a new
corpus version. `documents/` and `download_report.json` are gitignored.

`manifest.json` contains only accepted PDFs, with source URL, category, frozen
discovery/calibration/heldout split, page count, byte size, and SHA-256.

## Evaluation rule

Inspect failures and design normalization only with `discovery`. Use
`calibration` for thresholds and configuration. Do not inspect or repeatedly
run `heldout` until the approach is frozen. The current generic pipeline is
the baseline; corpus-specific legal identity promotion must not be counted as
a generic improvement.

Do not redistribute the downloaded corpus as a bundle. Source licenses vary;
publish the selection/manifest, downloader, methodology, and measured results.

## Golden dataset and baseline accuracy (discovery split, 447 cases)

`build_golden_dataset.py` generates `golden_dataset.json` — every fact
extracted programmatically from the actual PDF text/metadata (never hand-
transcribed) and round-trip-verified against the source document at build
time before being kept, same discipline as `eval/build_golden_dataset.py`.
Question types: `fact_author` / `fact_figure_caption` (scientific, the
latter deliberately biased toward captions on late pages of 20+-page
documents), `fact_active_ingredient` / `fact_rx_approval` /
`fact_rx_indication` (medical — DailyMed ships two incompatible label
formats, OTC and Rx, handled separately), `fact_table_caption` (manuals),
`topic_search` (recall-only, query built from title words, never the
literal title), `comparison_scoped` / `comparison_unscoped` (see below),
`hard_negative` (verified absent from every title in the full manifest,
not just discovery).

Run via `eval/run_eval.py --dataset eval/mixed_corpus/golden_dataset.json
--semantic-judge` against the isolated instance (port 8002 — see
"Isolation" above; the 762-document baseline ingest already populated it).
`--semantic-judge` asks the local LLM to check a substring-match miss
before failing it outright — added because a strict verbatim check scored
`fact_rx_indication` at 0% despite every one of its 4 answers being
factually correct, just reworded ("LUCEMYRA is indicated for the
mitigation of..." vs. "...is indicated for mitigation of...").

**Results below are from the first pass (447 cases, port-8002
`mixed_corpus_test`) and are preliminary**: the corpus had drifted mid-run
(a later re-upload of two payload-limit-failed documents shifted Qdrant's
incremental sparse-vector IDF stats for the whole collection) and the
caption ground truth was truncated to `[:80]` chars at the time. Both
issues are fixed in the "Frozen v2 baseline" section below, which
supersedes the exact numbers here — this table is kept for the
methodology and the comparison/attribution findings, which still hold.

**Results, most important finding first:**

| Type | Document Recall@5 | Evidence-page Recall@5 | Correctness (substring→semantic) |
|---|---:|---:|---:|
| comparison_scoped | 100% (20/20) | — | — |
| comparison_unscoped | 5% (1/19) | — | — |
| fact_figure_caption | 97% (165/170) | **31% (53/170)** | 3%→11% (5→19/170) |
| fact_author | 97% (158/163) | — | 88%→89% (144→145/163) |
| fact_rx_indication | 100% (4/4) | — | **0%→100% (0→4/4)** |
| fact_rx_approval | 100% (10/10) | — | 80% (unchanged) |
| topic_search | 90% (44/49) | — | — |
| hard_negative | — | — | correctly refused 3/3 |

1. **Comparison is two different features, not one number.** The frontend's
   actual "Compare Documents" button passes `document_ids` on the request
   (`api/schemas.py`'s `QueryRequest.document_ids`), which fires
   `rag/retriever.py::_augment_compare_queries`'s one guaranteed sub-query
   per named document — scored here as `comparison_scoped`, 100%. A plain
   natural-language question that merely *names* two documents never
   reaches that code path at all and found both documents only 5% of the
   time (`comparison_unscoped`). Reporting only the unscoped number under
   a plain "comparison accuracy" label — which an earlier pass of this
   test did — would have silently never tested the shipped feature.
2. **Document-level recall hid a real chunk-level limitation.** 97%
   document recall on `fact_figure_caption` looked like "retrieval works
   great on long documents" until Evidence-page Recall@5 — did any
   returned chunk actually come from the page the queried Figure/Table
   lives on, not just the right document — came in at 31%. The document
   is found; the specific page holding a caption 40+ pages into a
   100-page paper usually isn't among the 5 chunks handed to the LLM. The
   model correctly refuses rather than guessing in most of the remaining
   cases (56/170 answered at all) — a real, load-bearing limitation this
   pass was specifically designed to surface, not a bug.
3. **The semantic judge is what separates a scoring artifact from a real
   miss.** `fact_rx_indication` went from 0% to 100% — every answer was
   right from the start. `fact_figure_caption` only moved from 3% to 11%
   even with the same judge — confirming the caption-recall gap above is
   real, not a strict-substring artifact.

One `comparison_unscoped` case errored (empty exception message,
`cmp_unscoped_medical_2dcd4816-...`) — not yet root-caused, doesn't change
the headline numbers, follow up before trusting `comparison_unscoped`'s
exact denominator.

Full per-case detail: gitignored `last_run_scores_golden_dataset.json`
(same discipline as `eval/README.md` — regenerate via `run_eval.py` rather
than treating the JSON as a permanent record). Note the two `golden_dataset
.json` files in this repo (this one and `eval/`'s) share a filename but not
a directory — `run_eval.py --output` was added specifically so one corpus's
run can no longer silently overwrite the other's saved scores (it did,
once, mid-pass; recovered by re-running against the untouched original
instance and confirming the numbers matched `VALIDATION.md` exactly).

## Frozen v2 baseline (454 cases) — supersedes the numbers above

The golden builder was rewritten (`build_golden_dataset.py`): captions are
now captured as the full untruncated string (no more `[:80]`), each case
carries `evidence_char_start`/`evidence_char_end` offsets in the same
coordinate space as the real chunker's `char_start`/`char_end`, and an
**Evidence-chunk Recall** metric (`eval/run_eval.py::evidence_chunk_
overlaps`) checks whether a *specific returned chunk* — not just the
right page — covers the caption span. Regenerated dataset: 454 cases
(`fact_figure_caption` 177, was 170).

A second, frozen collection, `mixed_corpus_v2` (port 8003, Postgres
`ragdb_mixed_corpus_v2`), was built to remove the corpus-drift problem —
750/762 (98.4%) indexed cleanly, zero NUL-byte or payload-limit failures
this time (both fixes from above proven clean on a genuine fresh run).
`mixed_corpus_test` (port 8002) is left running untouched; do not compare
its saved scores against v2 runs — see the drift note above for why that
comparison is invalid.

`fact_figure_caption` (177 cases), against `mixed_corpus_v2`:

| | preliminary (drifted corpus, truncated ground truth) | v2 baseline | v2 + structural lookup |
|---|---|---|---|
| Doc Recall | 97% (165/170) | 100% (177/177) | 100% (177/177) |
| Evidence-page Recall | 31% (53/170) | 29% (51/177) | **36% (63/177)** |
| Evidence-chunk Recall | not measured | 20% (35/177) | **28% (49/177)** |
| Correctness (substring→semantic) | 3%→11% | 1%→11% | 0%→14% |

**The headline finding survives clean measurement**: Evidence-page Recall
reproduced almost exactly (31%→29%) on a corpus with zero drift — this was
never a comparison artifact, it's a real system limit. Evidence-chunk
Recall (20%, then 28% after the fix below) sits meaningfully below
Evidence-page Recall — some fraction of "page found" is still the wrong
chunk on that page.

**Structural Figure/Table-N lookup** (`rag/retriever.py::extract_
structural_references`, committed `8c089c7`): when a query explicitly
names a figure/table, text-scans the top 3 stage-1 candidate documents
(`vector_db/qdrant_client.py::chunks_for_document`) for the best-matching
caption and injects at most one chunk per document as an identity match —
deliberately narrow, unlike an earlier two-stage pool-widening attempt
that was tried and reverted (made Evidence-page Recall *worse*, 31%→24%,
by flooding the reranker with same-document decoy mentions of the same
figure number). Raised Evidence-chunk Recall 19.9%→28.2% overall
(+8.3pp, no regressions elsewhere), short of a 35-40% stretch goal.
Widening the scan window from top-3 to top-5 was also tried and reverted —
flat to slightly worse (27.1%), because a wider window gives more chances
for a same-numbered Figure/Table in an unrelated document to collide with
the real one. Root cause of the remaining ceiling is more likely
cross-document caption-number collisions than "correct document didn't
make the candidate list" — full taxonomy of the remaining misses is the
next step (see `eval/mixed_corpus/README.md`'s issue tracker / project
memory for the exact plan).

## Conditional generator benchmark: DeepSeek vs. local Qwen on confirmed-evidence captions

This is **not** a general "which LLM is better" test — DeepSeek is an
**evaluated cloud backend**, not an available mode of this product. The
live API's default and only wired-in generator is local Qwen 2.5 7B via
Ollama (`api/main.py`); nothing here changes that. It exists to answer one
narrower question: once retrieval has already found the right evidence,
does the generator itself still lose points, and does a larger/cloud model
close that gap?

Method (`eval/mixed_corpus/capture_generation_contexts.py` +
`eval/compare_generators.py`, both reusable tooling, not one-off scripts):
for the 49 `fact_figure_caption` cases where the structural-lookup run
above had a confirmed `evidence_chunk_hit=True`, retrieval was run once
per case against the frozen `mixed_corpus_v2` collection (the production
sequence: expand → retrieve_expanded → rerank → promote_identity_matches),
the resulting prompt saved verbatim, then sent unchanged to three models:
local `qwen2.5:7b`, `deepseek-v4-flash`, `deepseek-v4-pro`. A fresh
`evidence_chunk_overlaps()` check against the saved context excluded any
case where retrieval nondeterminism meant the captured context no longer
actually contained the evidence (1 excluded). `qwen2.5:7b` judged
correctness for all three models' answers — including its own — never
DeepSeek judging itself.

> On 47 technical caption questions with a verified evidence chunk in
> identical cached contexts, local Qwen 2.5 7B achieved 40.4%
> correctness, DeepSeek V4 Flash 76.6%, and DeepSeek V4 Pro 72.3%. These
> are conditional generation results, not end-to-end RAG accuracy. The
> sample does not establish that Flash is superior to Pro.

Full breakdown, pairwise contingency tables, and the `saved_by_change`
subset: `eval/mixed_corpus/generator_ab_report.md`.

Caveats, all load-bearing:
- **DeepSeek is not wired into `api/main.py`'s default pipeline.** Local
  `qwen2.5:7b` remains the product default; nothing here is user-selectable
  today.
- **A future cloud mode would send document chunks to an external
  provider.** That must be stated explicitly to any user who enables it,
  not left implicit.
- **DeepSeek does not fix missing evidence.** It only helps once retrieval
  has already found the right chunk — 28% of `fact_figure_caption`
  questions, per the structural-lookup result above. 28% × 77% ≈ 22%
  achievable end-to-end with the cloud generator vs. 28% × 40% ≈ 11% with
  local Qwen — retrieval, not generation, remains the primary limiter of
  overall coverage.
- **The Flash vs. Pro gap did not replicate.** An initial scratch-script
  pass showed Flash beating Pro 39/48 vs 34/48 with a lopsided 6:1
  discordant-pair ratio; the official reproduction through
  `compare_generators.py` (n=47) narrowed this to a near-coin-flip 6:4
  split. Don't generalize "Flash is the stronger model" from this sample.
- **A known golden-dataset label-noise issue affects the absolute
  percentages here.** A caption immediately followed by table-cell data
  can pull that data into `expected_substring` (see the golden-builder's
  "known accepted residual gap" above) — this depresses all three models'
  scores roughly equally, so it doesn't change the direction of the gap,
  but the exact magnitude isn't trustworthy until a golden-dataset v3
  separates caption text from adjacent table content.
