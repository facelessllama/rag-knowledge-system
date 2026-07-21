# Mixed PDF corpus

This corpus is the cross-domain and document-quality evaluation set. It is
separate from the legal golden/heldout sets and the EURLEX57K scale test.

For the chronological record of baselines, rejected experiments, accepted
fixes, exact commits, and local raw-artifact hashes, see
[`EXPERIMENT_HISTORY.md`](EXPERIMENT_HISTORY.md). This README explains the
method and findings; the history file is the compact audit trail.

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
the real one. That result, plus one hand-investigated case showing a
same-numbered-caption collision, made cross-document collisions look like
the likely dominant cause of the remaining ceiling — **the full taxonomy
below shows that guess was wrong**; see it for what actually dominates.

## Full taxonomy of the remaining `fact_figure_caption` misses

Per the user's explicit instruction, this is a diagnostic pass to find out
which failure mode actually dominates the remaining 128 Evidence-chunk
Recall misses (454-case v2+structural run) before picking a next retrieval
mechanism — not another blind mechanism attempt like the two tried and
reverted above, each of which generalized from a single deeply-investigated
case (see project memory's own caution about repeating that mistake).

Method (`eval/mixed_corpus/build_caption_label_index.py` +
`eval/mixed_corpus/analyze_caption_misses.py`, both reusable tooling):
1. A corpus-wide caption-label index — for every one of the 750 ingested
   documents, which (kind, num) Figure/Table labels have an actual
   caption-tier match somewhere in its indexed chunks, reusing the exact
   same detection function (`rag.retriever.structural_match_tier`)
   `retrieve_expanded` itself calls. A deterministic Qdrant payload scroll,
   not an embedding search — none of the cross-process nondeterminism
   documented below applies to building this.
2. For each of the 128 misses, the real production retrieval sequence
   (`query_expander.expand` → `retriever.retrieve_expanded` → `reranker.
   rerank` → the three `promote_*` calls) was re-run once, in-process,
   with `VectorStore.chunks_for_document`, `rag.retriever.best_structural_
   chunk`, and `VectorStore.hybrid_search` wrapped (not modified) to record
   which documents the structural lookup actually scanned, what it found,
   and whether the winning chunk was already present in the raw hybrid
   pool. Same nondeterminism safeguard as `capture_generation_contexts.py`:
   a case that no longer reproduces as a miss on this fresh run is
   excluded, not silently counted (1/128 excluded on this basis).

**Result (127 analyzed, 1 excluded)**:

| Category | n | Next mechanism |
|---|---:|---|
| Correct doc below candidate window | 0 | document disambiguation |
| Multiple documents share the same Figure N | 0 | collision-aware scoring |
| Caption absent from indexed chunks | 3 | parser/chunker |
| A prose mention / wrong occurrence picked instead of the caption | 12 | caption-aware scoring |
| Structural chunk found but got displaced before generation | **112** | *(see below — a specific, already-diagnosed bug, not a vague category)* |

**The dominant finding, confirmed on all 112/112 cases in the last
bucket**: `retrieve_expanded`'s identity-lookup blocks (case-number/
CELEX-ID/citation-number/structural-reference) all share one guard —
"`if key not in all_chunks:` inject with `score=1.0`,
`identity_match=True`". If the SAME chunk was already found (however
weakly) by plain hybrid search, this silently no-ops: the chunk keeps
its original weak score and never gets `identity_match=True`, so it
loses to the same document's own better-scoring chunks in the final
selection — exactly as if the structural lookup had never run at all.
Case/CELEX/citation-number matches are unaffected by this in practice
because they have a **second, independent** path further down in
`retrieve_expanded` that marks `identity_match=True` from the chunk's own
stored payload metadata (`case_number`/`celex_id`/`citation_number`
fields, set at ingestion) regardless of this guard. A caption has no such
stored metadata — it isn't extracted at ingestion time at all — so the
structural-reference mechanism has no fallback, and this guard-skip is
its only failure path in this dataset. Confirmed directly for every one
of the 112 cases (not inferred): each case's winning `best_structural_
chunk` result was independently found to already be present, at a weak
score, in the raw multi-query hybrid-search pool recorded via the
`hybrid_search` wrapper.

**Important caveat on the two zero categories**: `in_structural_window`
was `True` for all 127 analyzed cases (`doc_rank_in_chunks` was 1, 2, or 3
every time) — the structural top-3 window was never the bottleneck *in
this dataset*. That is very likely an artifact of how `build_golden_
dataset.py` phrases `fact_figure_caption` questions — `f'What does
{kind} {num} show in the paper "{title}"?'` always includes the exact
paper title, which all but guarantees the correct document ranks at or
near #1 on stage-1 hybrid/BM25 score alone. A real user asking about a
figure without quoting the paper's exact title would very plausibly hit
the below-candidate-window and collision failure modes this pass measured
at zero — this taxonomy answers "what's wrong once the right document is
already found," not "how often is the right document found at all" for
realistic phrasing. Worth a follow-up pass with title-free question
phrasing before concluding those two categories don't matter in practice.

The 3 `caption_absent_from_indexed_chunks` and 12 `prose_or_wrong_
occurrence_picked` cases are real but minor next to the 112 — full
per-case detail (`scanned_doc_ids`, `collision_count`, every boolean
above) is in the gitignored `eval/mixed_corpus/caption_miss_taxonomy.json`.

## Step 4: guard-skip bug fixed (`rag/retriever.py::retrieve_expanded`)

Given the taxonomy above, this wasn't a "choose a new retrieval
mechanism" decision — it was a one-line bugfix to an invariant the old
code silently violated: *whether the structural lookup promotes a match
must not depend on whether that same chunk was already found (weakly) by
hybrid search first.* The old code only injected a structural match
`if key not in all_chunks` — i.e. only when hybrid search hadn't already
found it. Now it always looks up the canonical chunk by key and sets
`score=1.0`/`source="structural_reference"`/`identity_match=True` on it,
whether newly injected or already present. Two new regression tests in
`tests/test_retriever.py` cover the exact broken path (all competing
chunks deliberately in the same document, so the per-document diversity
guarantee can't mask a regression) and its equivalence with the
already-working "chunk absent from hybrid" case.

**Measured on the SAME frozen `mixed_corpus_v2` collection, full 454
cases, `--semantic-judge`** (`eval/mixed_corpus/last_run_scores_
golden_dataset_v2_structural_fix.json`):

| `fact_figure_caption` (177 cases) | before fix | after fix |
|---|---:|---:|
| Doc Recall | 100% (177/177) | 100% (177/177) |
| Evidence-page Recall | 36% (63/177) | **92.7% (164/177)** |
| Evidence-chunk Recall | 28.2% (49/177) | **91.5% (162/177)** |
| Answered-when-should | 56% (100/177) | 95.5% (169/177) |
| Correctness (substring→semantic) | 0%→14% | 1.7%→39% |
| Avg. sources per case | 2.0 | 2.3 |

A retrieval-only pre-check (no generation — direct `retrieve_expanded` +
rerank + promote calls) across all 177 cases predicted 91.5% Evidence-
chunk Recall before the official run was even started; the official run
landed at the same number, confirming the fix's effect is exactly the
structural-injection mechanism, not a generation-side artifact.

**No regressions elsewhere**: `comparison_scoped` 100%→100%,
`hard_negative` 3/3→3/3, `fact_author` 163/163→163/163 recall
(151/163→151/163 correctness), `fact_rx_approval` 8/10→8/10,
`topic_search` 46/49→46/49. Overall `Recall@5` 94.9%→95.3%
(v2 baseline → after fix), MRR 0.924→0.926. Full test suite: 356 passed
(354 + 2 new).

The remaining ~8% gap is the 3 `caption_absent_from_indexed_chunks` +
12 `prose_or_wrong_occurrence_picked` cases from the taxonomy above — a
parser/chunker issue and a caption-disambiguation issue respectively,
neither touched by this fix, both small enough not to justify their own
pass right now.

## Title-free robustness check, and a second fix it found

The 91.5% above was measured on questions that always include the
paper's own title (`f'What does {kind} {num} show in the paper "{title}"?'`)
— a real product user asking about a figure would rarely quote a paper's
exact title. `eval/mixed_corpus/title_free_retrieval_check.py`
(retrieval-only, no generation) reruns the same 177 `fact_figure_caption`
cases under three question variants against the same frozen
`mixed_corpus_v2` collection:

| Variant | Question | Scope | Doc Recall | Evidence-page | Evidence-chunk | `structural_reference` present |
|---|---|---|---:|---:|---:|---:|
| `current_golden` | full title + Figure N | none | 100% | 93.2% | 91.5% | 98.3% |
| `title_free_global` | Figure N only | none | 4.0% | 2.8% | 2.3% | 1.1% |
| `title_free_scoped` (before fix) | Figure N only | correct `document_id` | 100% | 78.5% | 69.5% | **0.0%** |
| `title_free_scoped` (after fix) | Figure N only | correct `document_id` | 100% | 96.0% | **95.5%** | 98.3% |

**`title_free_global` collapsing to near-zero is expected, not a bug** —
"What does Figure 7 show?" over a 762-document corpus with no other
context is honestly ambiguous. It's a useful adversarial stress test (and
arguably the system's correct behavior there is to ask which paper, not
guess), not a fair product-accuracy number, and isn't tracked as a
regression target.

**`title_free_scoped` was the real product gap**: the realistic workflow
— a user has a document open (so the API already has its `document_id`)
and asks about one of its figures without repeating the title —
`structural_present` at exactly 0.0% before the fix confirmed
`retrieve_expanded` was skipping the structural lookup entirely whenever
`document_ids` was set (`if query_structural_refs and not doc_scope`),
an assumption that a scoped query "already knows its documents" so the
per-document caption scan was redundant. It wasn't: knowing which
document matters says nothing about whether hybrid search found the
right chunk within it.

**Fixed** (`rag/retriever.py::retrieve_expanded`, commit `5f0f73e`): when
`doc_scope` is set, scan every explicitly selected document instead of
skipping — these are already user-confirmed relevant, not a stage-1
guess, so `STRUCTURAL_TOP_DOCS`'s ambiguity-bounding cap doesn't apply
(bounded instead by `MAX_DOCUMENT_IDS`, 50). Interestingly,
`title_free_scoped` after the fix (95.5%) is now slightly *higher* than
`current_golden` (91.5%) — an explicit `document_id` scope removes any
chance of cross-document interference entirely, a cleaner signal than an
unscoped search even with the title included.

**No regressions**: `comparison_scoped`/`comparison_unscoped`/
`hard_negative` all unchanged in the official `run_eval.py` pass (the
golden dataset's own `fact_figure_caption` cases are unscoped, so don't
exercise this exact path — the retrieval-only check above is what
actually verifies it). 3 new tests in `tests/test_retriever.py`: applies
within `doc_scope`, scans ALL scoped documents (not capped at 3), and
never touches an out-of-scope document even when it would otherwise be
the strongest stage-1 candidate — protecting the compare flow. Full
suite: 358 passed.

**Deliberately not pursued further** (per the user's own call): chasing
the remaining ~5-8% residual gap (the same 3+12 taxonomy cases, plus
whatever `title_free_scoped` still misses) risks overfitting to this one
corpus for diminishing real-world benefit. Golden-dataset v3 (separating
caption text from adjacent table-cell content) and a post-fix DeepSeek
A/B replication are the higher-value next steps — see the project
memory checkpoint.

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
- **DeepSeek does not fix missing evidence.** This A/B used contexts captured
  at commit `8c089c7`, when the first structural lookup reached only 28.2%
  Evidence-chunk Recall. At that historical checkpoint, 28% × 77% ≈ 22%
  was the rough cloud-generator ceiling and retrieval was still the primary
  limiter. That conclusion no longer describes the current retriever: the
  later canonical-promotion and scoped-lookup fixes raised Evidence-chunk
  Recall to 91.5% unscoped-with-title and 95.5% title-free-with-document-
  scope. The A/B still establishes a conditional generation gap on identical
  evidence, but a post-fix replication is required before estimating current
  end-to-end cloud accuracy or declaring the current dominant bottleneck.
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

## Golden-dataset v3, and what re-scoring the saved answers found

v3 (`eval/mixed_corpus/build_golden_dataset.py`, see `extract_caption`'s
docstring) splits `caption_text` from `table_cells` for every
`fact_figure_caption`/`fact_table_caption` case — the label-noise issue
named above. v2 (`golden_dataset.json`) is untouched; v3 is a separate
file (`golden_dataset_v3.json`) with a companion `.metadata.json` (builder
version, corpus manifest hash, git commit, and every case whose
`expected_substring` changed from v2 and why — 31 of 181 caption cases).
Each case also gets `label_status`: `verified` (149 — no ambiguity),
`extracted` (10 — a confident table_cells split), `ambiguous` (22 — kept
in the dataset, flagged for review rather than either merged or split
with false confidence). All 10 `extracted` cases were checked by hand
against the real source PDFs: 9/10 correct splits, 1 known false positive
(arxiv_2607.11978's Table 3 — a genuine sentence enumerating several
capitalized abbreviations triggers the same signal a real table header
would) documented in `_classify_trailing_segment`'s own docstring rather
than chased further — a text-only heuristic can't fully solve this
without real page-layout info, same conclusion v2 already reached.

**Re-scoring the ALREADY-SAVED DeepSeek A/B answers against v3**
(`eval/mixed_corpus/rescore_against_v3.py` — no generator called again;
the only model call is the same local semantic-judge re-verification
`run_eval.py` already does on a substring miss) answers the question the
label-noise caveat above left open: is the generation gap partly a
scoring artifact, or is it real?

| Model | v2 correctness | v3 correctness | wrong→correct | correct→wrong | still all-wrong |
|---|---:|---:|---:|---:|---:|
| qwen2.5:7b | 41.7% | 50.0% | 4 (3 label-fix) | 0 | 24 (19 on verified labels) |
| deepseek-v4-flash | 77.1% | 81.2% | 4 (4 label-fix) | 2 (0 label-fix) | 7 (6 on verified labels) |
| deepseek-v4-pro | 72.9% | 75.0% | 1 (1 label-fix) | 0 | 12 (9 on verified labels) |

**The generation gap is real, not primarily a scorer artifact.**
Absolute correctness moved only modestly (+8.3pp / +4.1pp / +2.1pp) —
per the user's own pre-declared criterion, a *large* share of "all wrong"
disappearing would have pointed at a scorer defect; instead the large
majority of still-wrong cases are on **unchanged (`verified`)** labels
(19/24, 6/7, 9/12), where v3's expected_substring is byte-identical to
v2's. The label fix that motivated v3 was real (`wrong_to_correct` is
mostly attributable to it), but it was never the dominant explanation for
the qwen/Flash/Pro gap. The 2 `deepseek-v4-flash` `correct→wrong` flips
were checked by hand: both on `verified` (unchanged) labels, i.e. pure
semantic-judge nondeterminism between the original and re-scoring pass,
not a real regression or a v3 defect.

**Next**: capture fresh contexts from the current (post both retrieval
fixes) retriever and repeat this A/B properly — this rescoring reused the
OLD pre-fix contexts/answers (valid for isolating the label-noise
question, since nothing about the retrieved context changed), but the
confirmed-evidence *population* itself is now completely different
(91.5%/95.5% vs. the 28.2% these 48 cases were drawn from), so a full
re-draw is needed before trusting new absolute end-to-end numbers.

## Post-fix DeepSeek A/B: the full new evidence-hit population, scored against v3

New capture (not a rescore) on the CURRENT retriever (both retrieval
fixes applied) — `eval/mixed_corpus/capture_generation_contexts.py`
against `golden_dataset_v3.json`, every `fact_figure_caption` case
attempted, keeping every one with a confirmed `evidence_chunk_hit`:
**162/177 (91.5%)** — 113 of these are cases the retrieval fixes newly
rescued (`meta.saved_by_change`, vs. the pre-fix baseline), not just the
same 48 from before. Sent unchanged to `qwen2.5:7b`, `deepseek-v4-flash`,
`deepseek-v4-pro` via `eval/compare_generators.py`. Artifacts (new names,
so the historical 48-case A/B stays intact and comparable):
`generator_ab_postfix_v3_contexts.json`, `_results.json`,
`_report.md`. Analysis: `eval/mixed_corpus/analyze_postfix_ab.py` →
`analyze_postfix_ab_report.json`.

Three metrics, computed directly per-case (never by multiplying rounded
percentages), each reported on **`verified`-only** (the headline — no
label ambiguity), **`verified`+`extracted`** (second cut), and
**`ambiguous`** (reported separately, never folded into either headline):

| | n eligible | n evidence-hit | Conditional correctness | End-to-end supported correctness | Refused despite evidence |
|---|---:|---:|---:|---:|---:|
| **qwen2.5:7b** — verified | 148 | 135 | 45.2% (61/135) | 41.2% (61/148) | 1.5% (2/135) |
| qwen2.5:7b — verified+extracted | 158 | 145 | 46.9% (68/145) | 43.0% (68/158) | 2.1% (3/145) |
| qwen2.5:7b — ambiguous | — | 17 | 52.9% (9/17) | — | 5.9% (1/17) |
| **deepseek-v4-flash** — verified | 148 | 135 | 71.1% (96/135) | 64.9% (96/148) | 0.7% (1/135) |
| deepseek-v4-flash — verified+extracted | 158 | 145 | 73.1% (106/145) | 67.1% (106/158) | 0.7% (1/145) |
| deepseek-v4-flash — ambiguous | — | 17 | 82.4% (14/17) | — | 0.0% (0/17) |
| **deepseek-v4-pro** — verified | 148 | 135 | 66.7% (90/135) | 60.8% (90/148) | 0.0% (0/135) |
| deepseek-v4-pro — verified+extracted | 158 | 145 | 69.0% (100/145) | 63.3% (100/158) | 0.0% (0/145) |
| deepseek-v4-pro — ambiguous | — | 17 | 70.6% (12/17) | — | 0.0% (0/17) |

Conditional correctness on the full new population (45.2/71.1/66.7%) is
close to the historical 48-case number (40.4/76.6/72.3%) — the gap did
not collapse or blow up when the population grew ~3.4x and the labels
got cleaner, a reasonable stability check on both the A/B methodology and
the original 48-case sample not having been a fluke.

**Model agreement, 162 evidence-hit cases**: all three correct 63, all
three wrong 28, disagreement 71.

| | wrong→correct | correct→wrong |
|---|---:|---:|
| qwen2.5:7b → deepseek-v4-flash | 50 | 7 |
| qwen2.5:7b → deepseek-v4-pro | 46 | 11 |

**Manual review** (per plan: all disagreements skimmed, the qwen-
correct/DeepSeek-wrong direction read in full — 18 cases across both
pairs — plus a 10-case all-wrong sample):

1. **A real, specific Pro failure mode, not scorer noise**: in several
   qwen-correct/Pro-wrong cases, Pro's answer visibly loses track of
   which document it's describing — e.g. `sci_caption_2607.14777_Table8`:
   Pro's answer opens "The provided excerpts include snippets from three
   different papers: 'SEED...', 'LOGOS...', and 'All Explanations are
   Wrong...'" before addressing the actual question; `sci_caption_2607.
   11436_Figure5`: "Based on the provided excerpts, only one document
   matches the paper in question..."; `sci_caption_2607.09403_Table19`:
   answer is truncated mid-sentence ("...Table 19 is titled ... It").
   These read as a genuine generation-quality issue specific to Pro
   (context-attribution confusion / truncation), not a scoring artifact.
2. **Some flash/pro "wrong" verdicts look like plausible scorer noise**
   on manual read (e.g. `sci_caption_2607.10963_Figure2` — Flash's answer
   substantively matches the expected 2D interval-tree description) —
   consistent with this project's already-documented semantic-judge
   nondeterminism, not a new finding.
3. **A newly-found, more severe golden-v3 blind spot**, discovered by
   reading the all-three-wrong sample: `_classify_trailing_segment`'s
   split only ever runs when the sentence-splitter finds a SECOND
   sentence piece after the caption's first one. When leaked table/chart
   content (axis tick labels, category-name lists, numeric data rows)
   has **no period anywhere in it at all** — routine for this kind of
   content — the naive splitter never produces a second piece, the whole
   contaminated blob is swallowed as the FIRST "sentence," and the split
   mechanism never runs at all. Confirmed directly:
   `sci_caption_2607.09641_Table1`'s `expected_substring` (424 chars,
   `label_status="verified"`) is "Performance Evaluation on the
   E-Commerce Fraud Dataset Model TN FP FN TP Precision Recall F1-Score
   XGBoost (Raw Tabular) 28833 1 1389 0 0.000 0.000 0.000 [...]" — raw
   table cell values with no model ever going to reproduce them verbatim,
   guaranteeing an all-three-wrong verdict regardless of true answer
   quality. A quick corpus scan: **17/148 (11%) of `verified`
   `fact_figure_caption` cases have an expected_substring over 300
   characters** — not proof each one is corrupted this way, but a
   plausible-enough concentration that the `verified` tier's absolute
   correctness numbers above should be read as a **floor**, not a
   precise measurement, until a golden v4 pass scans for table/chart-like
   density anywhere in the extracted span, not just in a second sentence-
   splitter piece. Not fixed in this pass — same "diminishing returns"
   reasoning as the rest of this residual-gap family, but flagged
   explicitly rather than left for someone to discover by surprise.

**Bottom line**: DeepSeek Flash/Pro's ~20-25pp conditional-correctness
edge over local Qwen replicates on the full, ~3.4x larger, cleaner-label
post-fix population — this is not a fluke of the original 48-case
sample. Pro shows a specific, real context-confusion failure mode absent
from Flash. The `verified`-tier absolute numbers are a floor, not a
ceiling, given the newly-found no-period leakage blind spot above.

## Separate check: title-free, document-scoped generation

Deliberately a SEPARATE artifact/report from the A/B above, never
merged: dropping the paper title changes the prompt itself, so mixing
the two populations would conflate a retrieval-scope difference with a
wording difference in one number. `eval/mixed_corpus/capture_title_free_
scoped_contexts.py` (reuses `title_free_retrieval_check.py`'s question
wording, `document_ids=[expected_doc_id]` scope) captured 168/177
contexts; `generator_ab_postfix_v3_titlefreescoped_*` is the generation
pass, `analyze_postfix_ab_titlefreescoped_report.json` the analysis
(same `analyze_postfix_ab.py`, pointed at the new results file).

| | Conditional correctness (verified) | End-to-end supported | Refused despite evidence |
|---|---:|---:|---:|
| qwen2.5:7b | 47.5% (67/141) | 45.3% (67/148) | 0.7% |
| deepseek-v4-flash | 52.5% (74/141) | 50.0% (74/148) | 0.0% |
| deepseek-v4-pro | 48.9% (69/141) | 46.6% (69/148) | 0.0% |

**The DeepSeek edge shrinks dramatically without the title**: the ~21-26pp
gap on the title-bearing unscoped A/B above narrows to +5.0pp (Flash) and
+1.4pp (Pro) here — on the *scored numbers*, Qwen and DeepSeek are close
to indistinguishable once the question doesn't name the paper.

**This is reported as an open, unresolved finding, not a confirmed
conclusion** — a manual spot-check of `qwen-correct/DeepSeek-wrong`
cases here (unlike the equivalent check on the title-bearing A/B, which
found a real Pro-specific failure mode) mostly turned up DeepSeek
answers that read as substantively correct paraphrases on manual
inspection (e.g. `sci_caption_2607.11436_Figure5`,
`sci_caption_2607.10212_Table6`) — plausibly consistent with this
project's already-documented semantic-judge nondeterminism rather than a
real quality drop. The already-known golden-v3 blind spot (leaked
table/chart content with no period, e.g. `sci_caption_2607.09641_Table1`
again) also reappears in the all-three-wrong sample here, depressing all
three models' scores roughly equally. **Two competing explanations are
both plausible and not yet distinguished**: (a) the paper title genuinely
lets a larger cloud model draw on background knowledge/context a smaller
local model can't, so the edge is real and title-dependent; or (b) the
narrower context/shorter, more literal answers this scenario tends to
produce interact with scorer strictness differently for verbose
cloud-model phrasing than for Qwen's phrasing. Deciding between these
needs either a larger hand-audited sample specific to this scenario or a
recalibrated judge — not done here.
