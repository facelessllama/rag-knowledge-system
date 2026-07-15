# eval/

Regression harness for retrieval and answer quality. Not part of `pytest`
(needs a running instance with `test_corpus/` + `test_corpus_heldout/`
ingested) — run manually, see commands below.

## Files

- `build_golden_dataset.py` → `golden_dataset.json` (137 cases) — the
  calibration set. `RELEVANCE_THRESHOLD` (see `api/main.py`) was tuned
  against this set.
- `build_heldout_dataset.py` → `heldout_dataset.json` (83 cases) — documents
  and adversarial questions never touched during calibration or debugging.
  Exists because `golden_dataset.json` was used both to pick the threshold
  *and* to report the metrics claiming it works — circular. Run against
  this set to check a fix actually generalizes, not just fits what it was
  tuned on.
- `run_eval.py --dataset <path>` — runs either dataset against a live
  instance, reports Recall@K, MRR, abstention accuracy, answer correctness.
  Writes `last_run_scores_<name>.json` (gitignored).
- `test_deepseek_accuracy.py` — standalone A/B test, local (Ollama) vs
  DeepSeek (cloud) generation on identical retrieved context. Not wired
  into `api/main.py`'s default pipeline. See "DeepSeek A/B" below.

Both datasets have three question types generated per PDF filename
(`<TYPE>_<NUM>_<YEAR>_<parties>.pdf`):
- `case_summary` — party names only ("What is the case X vs Y about?")
- `case_lookup_by_number` — includes the case number ("What is case CRM
  10691 of 2020 (X vs Y) about?")
- `fact_date` / `fact_money` — programmatically extracted from
  `test_corpus/text_legal_docs/` ("come into force on <date>", single-amount
  "£<N>" clauses), plus 15 hand-written adversarial (should-refuse) questions.

## Commands

```bash
python eval/build_golden_dataset.py
python eval/build_heldout_dataset.py
python eval/run_eval.py --dataset eval/golden_dataset.json
python eval/run_eval.py --dataset eval/heldout_dataset.json
python eval/test_deepseek_accuracy.py --failing-only   # needs DEEPSEEK_API_KEY in .env
```

## Current state (as of document-opening-chunk promotion + refusal-retry)

| | golden (calibration) | heldout |
|---|---|---|
| Recall@5 | 100.0% | 100.0% |
| MRR | 1.000 | 1.000 |
| Answered when should | 100.0% (122/122) | 100.0% (68/68) |
| Refused when should | 100% (15/15) | 100% (15/15) |
| Answer correctness (substring) | 98.8% (81/82) | 100% (29/29) |

Reproduced across two consecutive runs of golden_dataset.json — abstention
accuracy 100% both times. The one remaining answer-correctness miss
(`date_EN_0088`) is a cosmetic substring mismatch ("January 1st, 2013" vs.
the expected "January 1, 2013" format), unrelated to retrieval/abstention.

(The one golden abstention miss, `date_EN_0084`, is a newly-isolated, distinct
issue — see "Known issue: multi-doc near-duplicate-title flakiness" below.
Not the same root cause as the A.B.A. 493 case fixed below, and not fixed
here.)

(Answer correctness moves a point run-to-run — `date_EN_0084`/`_0088` are
generation-level substring misses on the LLM's exact wording, at temperature
0.1 but not 0, unrelated to retrieval/threshold. The one heldout abstention
miss is `ho_case_lookup_by_number_A.B.A._493_2020_RAJKUMAR_YADAV_vs_THE_ST` —
see "New, unrelated observation" below; retrieval for it is verified perfect.)

`reranker.rerank()` now scores `chunk_context_text()` (title-prefixed) instead
of bare `chunk.text` — see "Next steps" history below, item 1. This closed the
`case_summary` gap outright: 12/20 (60%) → 20/20 (100%) on heldout. The
cross-encoder previously judged boilerplate-only text ("Heard learned counsel
for the petitioner...") with no way to tell documents apart; it now sees the
party names from the title directly, which is exactly what `case_summary`
questions ask about. `case_lookup_by_number` also went 18/19 → 19/19 —
`MACA_80` (previously written off as a data-completeness edge case, since its
extracted PDF text has no case number) now resolves via the title's party
names instead. One remaining failure — `date_EN_0088` (Animals Scientific
Procedures Act 1986 Fees Order 2012) — is a pre-existing, unrelated
substring-match miss on the golden set, not something this change touched.

This raised a real question — was `RELEVANCE_THRESHOLD=1.5` (calibrated
*before* this change) still valid against the new score distribution? Checked
directly: pooling both datasets (220 cases), should-refuse scores now top out
at 1.00 and should-answer scores bottom out at 4.94 — a 3.9-point gap, and
critically, this gap is uniform across every question type, not just
`case_summary`. The title-context fix lifted *all* should-answer scores
together, not only the ones that were previously borderline. That means the
per-question-type/adaptive threshold originally planned as item 2 below turned
out to be unnecessary — a single global value has ample margin now. Moved
`RELEVANCE_THRESHOLD` to `3.0` (the midpoint of the new gap, vs. the old 1.5
which was hugging the should-refuse edge) for more headroom; re-ran
`run_eval.py` on both datasets to confirm no regression (see "Current state"
table above — unaffected).

## Structured case metadata at ingestion

Item 3 from "Next steps" below: `ingestion/chunker.py::extract_case_metadata()`
now parses each case PDF's filename (`<TYPE>_<NUM>_<YEAR>_<PARTY1>_vs_
<PARTY2>.pdf`) into structured `case_number`/`case_year`/`parties` fields at
ingestion time, stored on every chunk's Qdrant payload (indexed — see
`_PAYLOAD_INDEXES` in `vector_db/qdrant_client.py`) instead of being
re-derived ad hoc by regexing each candidate chunk's own text at query time.
Two concrete upgrades this enabled in `rag/retriever.py`:

1. **Guaranteed retrievability by case number.** Previously, a case-number
   match could only be marked on a chunk that hybrid search's top-`pool`
   candidates happened to already include — if the target document scored
   too low semantically to make that pool, the exact-match promotion
   (`promote_identity_matches`) had nothing to promote. `chunks_by_case_number()`
   now does a direct Qdrant payload-filtered lookup for any (number, year)
   found in the query and injects those chunks into the candidate pool
   regardless of semantic ranking — this stops being a bet that gets worse
   as the corpus grows.
2. **Party-name matching for `case_summary`.** New `extract_party_match()`
   promotes a chunk when *all* of its document's party names appear in the
   query — as strong an identity signal as an exact case-number match, and
   available for the first time since parties weren't structured data
   before. Requires all parties (not just one) specifically because many
   filenames in this corpus share a common party (e.g. "... vs STATE OF
   ODISHA") — matching on just one would over-promote unrelated cases.

Existing corpus (400 docs, ~9800 chunks) backfilled via
`scripts/backfill_case_metadata.py` — a payload patch from each point's
already-stored `filename`, no re-parsing/re-embedding needed (4747 chunks
matched the case-file convention and got tagged; 5074 skipped, expected —
those are the plain-text UK statutory instruments with no case number).

Measured via `run_eval.py`: Recall@5 and refuse-when-should are unchanged at
100%. One new heldout abstention miss appeared
(`ho_case_lookup_by_number_A.B.A._493_2020_...`) — investigated directly
(queried the live API, inspected `debug.top_chunks`): retrieval is perfect,
all 8 reranked chunks are from the correct document at scores 8.0-8.3
(`source: "hybrid"` throughout — this case's target was already found by
ordinary hybrid search, this change's new code paths didn't even fire for
it). The LLM's generated answer starts with a refusal despite clearly
relevant context, and reproduces after a full API process restart (not
random-seed noise). This is a generation-stage issue, not a retrieval
regression from this change — flagged below as a new, separate follow-up.

## Context reading-order reassembly (the A.B.A. 493 refusal, root-caused)

Investigated the deterministic refusal flagged above
(`ho_case_lookup_by_number_A.B.A._493_2020_RAJKUMAR_YADAV_vs_THE_ST`) by
reconstructing the exact prompt sent to the LLM and bisecting which chunks/
ordering triggered it. Root cause confirmed: `rag/prompt_builder.py`
presented chunks to the LLM in **rerank-score order**, not document reading
order. For this document, the chunk that establishes case identity (page 1,
chunk 0 — "IN THE HIGH COURT OF JHARKHAND AT RANCHI A.B.A. No. 493 of 2020
... Rajkumar Yadav ...") scored *lowest* of the top-5 reranked chunks
(names/formatting-heavy text reads as less semantically "relevant" to a
cross-encoder than the procedural prose around it) and landed 5th in the
context, preceded by four procedural excerpts that repeatedly cite a
*different* number — "Pratappur P.S. Case No.133 of 2019" (the underlying
police FIR, not the court application). qwen2.5:7b, given the off-topic-
looking number first and the actual case caption buried later, concluded the
context didn't cover "case A.B.A. 493 of 2020" and refused — reproducibly,
even after a full server restart (verified by replaying the exact 8-chunk
context directly against the generator, isolated from retrieval).

Bisection (`/tmp/.../probe_refusal.py`, not checked in) confirmed it's pure
ordering, not content: the identical 8 chunks reordered with the caption
chunk moved first answered correctly on every trial; dropping either of the
two "wrong number" procedural chunks (rather than reordering) also fixed it;
the original rerank-score order failed 3/3.

**Fix:** `rag/prompt_builder.py::_order_for_reading()` — chunks are now
re-sorted into `(document, page_num, chunk_index)` order before being
formatted into the prompt, so each document's excerpts read the way the
document itself reads, caption first. Documents themselves stay in their
original relevance order (multi-doc mode still shows the most relevant
document's excerpts before a less relevant one's) — only the chunks *within*
a document get reordered. Verified: the A.B.A. 493 query now answers
correctly 4/4 on the live API; `run_eval.py` on both datasets shows no
regression elsewhere (Recall@5 and refuse-when-should unchanged at 100%).
`tests/test_prompt_builder.py` covers both the reordering itself and that
document-level relevance order is preserved.

## Multi-doc near-duplicate-title flakiness (root-caused and fixed)

Verifying the A.B.A. 493 fix surfaced a *different* failure: `date_EN_0084`
(`The Localism Act 2011 (Commencement No. 6 ...)`) abstained intermittently
(~35-90% of trials, varying by run) even though Recall@5 always found the
correct document. First hypothesis — genuine `temperature=0.1` sampling
noise, not fixable without a prompt rewrite — turned out to be **wrong**,
caught by trying to reproduce it in an isolated script: the same query
through a clean standalone pipeline call answered correctly 20/20, while
the live API kept failing on the "same" query. That discrepancy was the
tell that something *upstream* of generation differed.

Instrumented `LLMGenerator.generate()` to log the exact `messages` payload
sent to Ollama on every call (temporarily — not part of the committed
diff) and compared a failing live-API call against a succeeding standalone
one. They were **not** sending the same prompt: the live call's context for
`EN_0084` started mid-sentence — `"e cited as the Localism Act 2011
(Commencement No. 6 ...) and shall come into force on..."` — missing the
document's opening chunk (`chunk_index=0`, page 1), which actually reads
`"...This Order may be cited as the Localism Act 2011 (Commencement No.
6...) and shall come into force on the day after..."`. The reranker was
cutting the opening/citation chunk from the final top-5 in favor of other
chunks that scored higher on pure semantic similarity to the date question
— exactly the same mechanism as the A.B.A. 493 case (a short, formulaic
chunk that establishes document identity loses to content-specific chunks
on cross-encoder score), except here the chunk doesn't just get
*reordered* out of position, it gets **cut from the set entirely** — reading-
order reassembly can't fix a chunk that was never in the final 5 to begin
with. Confirmed with 22 live calls, all byte-identical prompts (hashed):
13 refused, 9 answered — real inference-level nondeterminism, but on a
prompt that was missing needed context, not on a complete one (the A.B.A.
493 fix's reconstructed complete prompt was 100% reproducible either way).

**Fix, two parts:**

1. **Root cause** — `rag/retriever.py::promote_document_opening_chunks()`:
   for every document already represented in the reranked `top_chunks`, if
   its `chunk_index == 0` chunk exists anywhere in the broader candidate
   pool (`chunks`, retrieve_expanded's output) but didn't survive
   reranking, add it back. Cheap (a handful of chunks at most per query)
   and general — not specific to this one document pair. Wired into both
   `/query` and `/query/stream` in `api/main.py`, right after
   `promote_identity_matches`.
2. **Defense in depth** — `rag/generator.py::LLMGenerator
   .generate_with_refusal_retry()` / `.generate_stream_with_refusal_retry()`:
   callers only reach generation after their own relevance-threshold gate
   already passed, so a refusal at that point is resampled (same prompt,
   fresh call) up to `refusal_retries` times before being accepted — cheap
   insurance against whatever inference-level nondeterminism remains on a
   *complete* prompt (rare — the A.B.A. 493 case showed 0% failure on a
   complete prompt across many trials — but not provably zero). The
   streaming variant buffers only the opening ~24 characters of the
   response to detect a refusal opener before committing to forward
   tokens to the client, so a normal answer streams with no added latency.
   `eval/run_eval.py` and `eval/test_deepseek_accuracy.py` now share the
   same `is_refusal()` check instead of three separately-maintained copies
   of the same prefix tuple.

**Verified:** 20/20 trials of the exact failing query against the live API
— 0 refusals (was 13/22 ≈ 59% single-shot, ~25% with retry-only before the
retrieval fix). `run_eval.py` on both datasets: 100%/100% Recall@5 and
abstention accuracy, reproduced across two separate golden_dataset.json
runs. `pytest` 98/98.

## DeepSeek A/B — the bottleneck is retrieval, not generation

Ran `test_deepseek_accuracy.py --failing-only` (16 known-failing cases,
identical retrieved context fed to both models): **qwen2.5:7b (local) and
DeepSeek (cloud) agreed on every single case — 9/16 correct, same 7
failures, zero disagreements.**

Inspecting the failures: in all 7, the expected document never appears
among the top retrieved chunks at all (completely unrelated cases surface
instead). In all 9 successes, the expected document is the top retrieval
hit. Success/failure correlates perfectly with whether retrieval found the
right document — not with which LLM generated the answer.

**Conclusion: no LLM upgrade (cloud or otherwise) will fix this class of
failure.** The fix has to be in retrieval/reranking, not generation. Don't
spend budget on a bigger/better generation model for this problem.

## Next steps (not done yet)

1. ~~Pass `chunk_context_text()` (see `ingestion/chunker.py`) to
   `reranker.rerank()`, not just to embedding~~ — done (`rag/reranker.py`).
   Closed the `case_summary` gap; see "Current state" above.
2. ~~Per-question-type (or otherwise calibrated/adaptive)
   `RELEVANCE_THRESHOLD` instead of one global value~~ — investigated, turned
   out unnecessary: item 1's fix lifted should-answer scores uniformly across
   types, so a single recalibrated value (`3.0`, see "Current state" above)
   has a comfortable 3.9-point margin on every type. Revisit only if a future
   corpus/question-type shows a similarly borderline gap that a global
   threshold can't cover.
3. ~~Structured case-number metadata extracted at ingestion time (not just
   matched ad-hoc against query text) would make `case_summary`-style
   lookups by party name more robust too, if paired with a metadata index~~
   — done (`ingestion/chunker.py::extract_case_metadata`,
   `vector_db/qdrant_client.py::chunks_by_case_number`,
   `rag/retriever.py::extract_party_match`). See "Structured case metadata
   at ingestion" above.
4. ~~Investigate the deterministic generation refusal on
   `ho_case_lookup_by_number_A.B.A._493_2020_...`~~ — root-caused and fixed:
   chunks were presented in rerank-score order, scattering a document's
   identity-establishing caption chunk away from its supporting excerpts.
   See "Context reading-order reassembly" above
   (`rag/prompt_builder.py::_order_for_reading`).
5. ~~Investigate the multi-doc near-duplicate-title flakiness on
   `date_EN_0084`~~ — root-caused and fixed: initial hypothesis (pure
   `temperature=0.1` sampling noise) was wrong — the reranker was actually
   cutting the document's opening/citation chunk from the final top-5
   entirely, not just reordering it (item 4's fix couldn't help here since
   there was nothing to reorder). See "Multi-doc near-duplicate-title
   flakiness" above (`rag/retriever.py::promote_document_opening_chunks`,
   `rag/generator.py::LLMGenerator.generate_with_refusal_retry` /
   `.generate_stream_with_refusal_retry`). 20/20 trials clean, was ~59%
   single-shot failure.
6. No further known issues — both datasets are at 100%/100%/100%
   (Recall@5 / abstention accuracy / answer correctness, golden's one
   cosmetic substring-format miss aside), reproduced across repeated runs.
   Next natural step, if pursuing it, is the corpus-scale question raised
   earlier (expanding well beyond the current ~400 documents) rather than
   further accuracy chasing on this corpus.

## Scale test: EU legislation corpus (COMPLETE — 400 → 57,000 documents)

Separate track from everything above: does accuracy degrade as the corpus
scales (1k → 5k → 15k → 30k → 57k), how consistent is a *repeated*
identical question, and does asking different questions about the *same*
document reveal partial degradation an aggregate metric would hide? Uses
a real, unrelated 57,000-document corpus (EURLEX57K — EU legislation,
`31958D1127(01) - EEC Council Rules of the Transport Committee.pdf`-style
filenames, CELEX ID + description) instead of synthetic/small-scale data,
so findings aren't an artifact of the existing 400-doc corpus's specific
shape. This becomes the reference "golden test" for scale-driven work, the
same role golden_dataset.json/heldout_dataset.json play above.

**Isolation**: a second API process on port 8001, separate Qdrant
collection (`eu_scale_test`) and separate Postgres database
(`ragdb_eu_scale_test`) — the main instance (port 8000, this file's
existing 400-doc corpus) is untouched throughout; confirmed via spot-check
after every shared-code change.

**New infrastructure** (generic, not corpus-specific — reusable for any
future scale test):
- `scripts/bulk_ingest_pdfs.py` — resumable bulk ingestion with a fixed,
  seeded shuffle order persisted to `eval/eu_manifest.json`, so
  `--up-to 5000` always means "the same first 5000 files" regardless of
  which run it's called from — each checkpoint is a strict superset of the
  previous one.
- `eval/build_eu_golden_dataset.py` — golden/heldout/cross-question dataset
  builder for this corpus (facts extracted from PDF text via regex, not
  hand-transcribed, same discipline as `build_golden_dataset.py`).
- `eval/build_eu_id_free_dataset.py` — ID-free natural-lookup/semantic-fact/
  hard-negative dataset (see "ID-free spot-check" below).
- `eval/build_eu_citation_heldout_dataset.py` — genuinely-disjoint citation-
  number generalization set, 5 natural phrasing formats, every `celex_id`
  used by any other EU dataset excluded (see "Citation-number
  generalization test" below).
- `eval/consistency_test.py` — repeats a representative question sample
  N times against a live instance, recording both answer correctness and
  *which document was actually retrieved* each time (separates retrieval
  flakiness from generation flakiness).
- `eval/run_eval.py` fixed: crashed with `ZeroDivisionError` on any
  dataset with zero adversarial (`expect_answer: False`) cases — the
  cross-question dataset has none by design. Now reports `n/a` instead.
  Later also fixed to count Recall@K strictly (`rank ≤ top_k`, not
  "anywhere in the returned sources array") and to accept a list of
  equally-valid answers for `expected_substring` — see "Recall@5 counting
  bug" in `VALIDATION.md`. `consistency_test.py` shares the same
  multi-answer substring check (`run_eval.substring_ok`).

**Two bugs found on this corpus and fixed** (mirroring the case-number/
party-name work above, same architecture, corpus-specific extraction
logic):
1. **CELEX ID exact match** (`ingestion/chunker.py::extract_celex_id`,
   `vector_db/qdrant_client.py::chunks_by_celex_id`, wired into
   `rag/retriever.py`) — more load-bearing here than case numbers are for
   the court-case corpus: a CELEX ID *never* appears in its own document's
   body text at all (EU legislation cites itself in a different format,
   e.g. filename `31997R0955` vs. body text `1997/955/EC`), so hybrid
   search had no literal string to find without a structured index.
2. **`cross_reference_fact` question wording** — the original phrasing
   ("what does X reference?") made the model consistently self-report X's
   *own* number in 19/19 inspected failures. Rewritten to extract the verb
   immediately preceding the citation ("amending"/"repealing"/"Having
   regard to"/...) and build a question that can't be answered with the
   document's own identity ("What earlier regulation is X amending?" /
   "does X cite (excluding its own number)?").

Impact of both fixes together, pilot (400 docs): golden Recall@5 88.7% →
**100%**, abstention accuracy 89.4% → **100%**, answer correctness 67.2% →
**100%**; heldout 86.7%/88.1%/58.2% → **100%/99.4%/90.3%**. Backfilled the
already-ingested 400 docs' `celex_id` via `scripts/backfill_case_metadata.py`
(extended to handle both identity conventions; scrolls existing points, no
re-parse/re-embed needed).

Initial heldout `cross_reference_fact` misses (5/9) were written off at the
time as a *dataset* ambiguity rather than a retrieval/generation defect —
many EU documents cite multiple earlier regulations ("Having regard to
X... and Y..."), and the regex-extracted "expected" one wasn't always the
one the model surfaced, even though both were legitimately present in the
text. That explanation was correct but the response to it wasn't — the
ground truth stayed single-answer instead of being fixed. A later review
(see "Recall@5 counting bug" below) corrected `extract_facts()` to collect
*every* citation number a document's body actually supports, not just the
first regex match, and `run_eval.py`/`consistency_test.py` to accept a
match against any of them. `cross_reference_fact` ground truth across
golden/heldout/cross_question (17 cases total) now carries a list instead
of a single string; 5 of those 17 genuinely have more than one valid
citation in the document body.

**Scale ladder results — full run** (golden / heldout, `eval/scale_results/`):

**Correction (post-hoc, see "Recall@5 counting bug" in `VALIDATION.md`):**
the table below originally reported a flat 100% Recall@5 at every corpus
size. That number was computed by checking whether the expected document
appeared *anywhere* in the API's returned `sources` array, which
`promote_identity_matches()` can widen to six entries even when `top_k=5`
— so a document at rank 6 was being counted as a Recall@5 hit throughout.
The table below is the strict `rank ≤ 5` version, recomputed from the same
saved per-case data and then **re-verified with a fresh live run against
the 57,000-document corpus** (matched the recomputation within normal
run-to-run variance — 109/150 and 104/150 live vs. 110/150 and 104/150
recomputed). The "in returned sources, up to 6-wide" column is what the
original 100% figure actually measured — real, but a different and looser
claim than Recall@5, and `run_eval.py` now reports both under distinct
labels so they can't be conflated again.

| corpus size | strict Recall@5 (golden/heldout) | in-sources≤6 (golden/heldout) | Abstention acc. | MRR (golden / heldout) |
|---|---|---|---|---|
| 1,000 | 96.0% / 95.3% (144,143/150) | 100% / 100% | 100% / 99.4% | 0.706 / 0.722 |
| 5,000 | 86.7% / 84.0% (130,126/150) | 100% / 100% | 100% / 99.4% | 0.599 / 0.608 |
| 15,000 | 80.7% / 78.0% (121,117/150) | 100% / 100% | 100% / 100% | 0.539 / 0.535 |
| 30,000 | 75.3% / 74.0% (113,111/150) | 100% / 100% | 100% / 100% | 0.496 / 0.504 |
| **57,000** | **72.7% / 69.3% (109,104/150)** | **100% / 100%** | **99.4% / 99.4%** | **0.448 / 0.453** |

Answer correctness (substring match on the checkable subset, unaffected by
the Recall@5 bug — it measures fact extraction, not document rank) stayed
strong throughout and is reported separately per checkpoint in
`eval/scale_results/*.json`; on the fresh 57,000 live rerun it was
98.2% (54/55) golden and 95.2% (59/62) heldout — the heldout figure moved
a couple points from the original 93.5% (58/62) because the
`cross_reference_fact` ground-truth fix above (accepting any citation the
document body genuinely supports, not just the first regex match) now
credits answers it previously marked wrong.

**Consistency test @ 57,000** (15 questions × 20 repeats = 300 calls,
`eval/consistency_results_at_57000.json`, same sample methodology as the
1,000 checkpoint): **300/300 answers still correct** — the "ask the same
question 50 times, get 45 wrong" failure mode does not occur even at full
scale. But retrieval stability cracked for the first time: **14/15
questions retrieval-stable** (was 15/15 at the 1,000 checkpoint) — one
`celex_lookup` question returned a different top document across repeats,
though the final answer stayed correct every time (generation papered
over the retrieval flip). Read this as a leading indicator: retrieval
instability shows up before it costs a wrong answer.

**Cross-question-per-document test @ 57,000** (`eval/scale_results/
cross_question_at_57000.json`): Recall@5 100%, answer correctness 97.2%
(35/36 checkable) — essentially unchanged from the 1,000 checkpoint's
27/30-documents-fully-correct result. No sign of a document being "half
indexed" even at full scale; the one miss is the same residual
`cross_reference_fact` ambiguity class described above (a document citing
multiple earlier regulations; the model picked a legitimate one that
wasn't the regex-extracted "expected" one).

### Methodology caveat — read this before the verdict below

**Every positive-answer question in `eu_golden_dataset.json` /
`eu_heldout_dataset.json` / `eu_cross_question_dataset.json` contains the
target document's exact CELEX ID in the question text itself** — see
`eval/build_eu_golden_dataset.py:120-155` (`f"What is {celex_id} about?"`,
`f"On what date was {celex_id} adopted?"`, etc.). There are no ID-free,
natural-phrasing positive questions anywhere in this scale-ladder dataset.

That matters because retrieval doesn't have to find these documents by
embedding/BM25 similarity alone. `rag/retriever.py::promote_identity_matches()`
(wired in at `api/main.py:513`) does a hard, structured-index override: any
chunk whose `celex_id` metadata matches an ID found in the query text is
force-inserted into the final result set with `rerank_score = RELEVANCE_THRESHOLD
+ 1.0` — a score that beats the abstention gate — **regardless of what the
cross-encoder reranker actually thought of it**. The append also isn't
re-sliced back down to `top_k`, so "Recall@5" for these cases sometimes
measures recall within a window quietly larger than 5.

Concretely, this means:
- **Recall@5 ≈ 100% throughout the ladder is largely mechanically
  guaranteed by this override**, not emergent evidence that dense+sparse
  hybrid search stays accurate as the corpus grows to 57,000. It mainly
  demonstrates that the CELEX-ID structured lookup (an O(1)-ish Qdrant
  payload filter) keeps working at scale — true and useful, but a
  narrower claim than "semantic search has max accuracy at 57,000
  documents."
- **The system-level abstention gate is similarly protected** for
  ID-bearing queries — a chunk that clears the floor score can't trigger
  the canned "couldn't find relevant information" refusal. It is *not*
  fully protected, though: the LLM can still generate a refusal-shaped
  answer even when handed the right chunk (this is exactly what happened
  in the one golden abstention miss below), so abstention accuracy is a
  mix of mechanically-guaranteed retrieval and genuine generation
  reliability, not purely the latter.
- **Answer correctness is not mechanically protected** — it measures
  whether the LLM extracts the right fact once *given* a context that's
  guaranteed (for these ID-bearing queries) to contain the correct
  document. That's a real, meaningful generation-quality signal, but it's
  decoupled from whether retrieval alone — without the ID crutch — would
  have surfaced that document for a realistic user who doesn't quote the
  CELEX ID.
- **MRR is the one metric this override does not protect.** The floor
  score guarantees inclusion above threshold, not top rank — the
  reranker still independently decides ordering among candidates. The
  monotonic MRR decline reported below is genuine evidence of growing
  rank confusion under near-duplicate density, not an artifact of the
  identity-match shortcut.

**What this ladder had actually validated up to this point**: the
structured-ID lookup path, generation faithfulness given correct context,
and ranking-confidence decay (via MRR) at scale. **What it had not yet
validated**: whether hybrid dense+sparse *semantic* search alone — the
path an ID-less, naturally-phrased question would have to rely on —
holds up at 57,000 near-duplicate documents. See the spot-check directly
below, which closes this gap. Treat the golden/heldout Recall@5 and
abstention numbers in "Final verdict" with the above in mind — MRR was
the load-bearing metric of that part of the report, not Recall@5.

### ID-free spot-check @ 57,000 — closing the caveat above

Built `eval/build_eu_id_free_dataset.py` → `eval/eu_id_free_dataset.json`
(120 cases) specifically to answer what the caveat above left open. Every
question is checked at build time (`assert_id_free()`) to contain no
CELEX-ID-shaped token and no `"<N> of <YYYY>"` case-number-shaped token,
so `promote_identity_matches()` never fires — retrieval has to stand on
embedding/BM25/reranking alone, run once against the full 57,000-doc
corpus (already ingested; no re-ingestion needed). Four types:

- **`natural_lookup_by_number`** (20) — refers to a document by its
  natural citation form, e.g. *"What does Regulation (EC) No 632/80
  concern?"* — a real citation style, but not the raw CELEX-ID token.
- **`natural_lookup_by_subject`** (20) — refers to a document by a
  paraphrase of its subject and rough date, no number at all.
- **`semantic_date_fact`** / **`semantic_cross_reference`** (20 each) —
  the same underlying facts as the golden dataset's `date_fact`/
  `cross_reference_fact` types, referred to by subject instead of ID.
- **`hard_negative_date`** (30) — deliberately drawn from this corpus's
  largest near-duplicate clusters (up to 509 near-identical members —
  see "Data used" below), with adoption date as the only differentiator
  supplied. Directly tests disambiguation under the exact pressure that
  drove the ladder's declining MRR, with no ID shortcut available.

**Results** (`eval/scale_results/id_free_at_57000.json`, strict `rank ≤ 5`
— see "Recall@5 counting bug" in `VALIDATION.md`; this table was
originally published with the same window-counting bug as the main
ladder, but every type here happened to score identically under both
definitions at this stage, so the numbers below are unchanged from first
publication):

| type | n | Recall@5 | MRR |
|---|---|---|---|
| hard_negative_date | 30 | **30/30 (100%)** | 1.000 |
| natural_lookup_by_subject | 20 | 20/20 (100%) | 1.000 |
| semantic_date_fact | 20 | 20/20 (100%) | 1.000 |
| semantic_cross_reference | 20 | 18/20 (90%) | 0.825 |
| natural_lookup_by_number | 20 | **15/20 (75%)** | **0.535** |
| **overall** | 110 | **103/110 (93.6%)** | **0.884** |
| abstention accuracy | | 117/120 (97.5%) | |
| answer correctness | | 32/40 (80.0%) | |

**Two findings, and they point in opposite directions.**

1. **Near-duplicate disambiguation itself is not the weak point.**
   `hard_negative_date` — 30 questions deliberately drawn from clusters of
   up to 509 near-identical documents, disambiguated by adoption date
   alone, no ID — went **30/30 with MRR 1.000**. This directly answers
   the worry the whole ladder started from: given a genuine natural
   differentiator, the system reliably picks the *one* correct document
   out of hundreds of near-duplicates, no ID crutch needed. The MRR decay
   seen across the main golden/heldout ladder is real, but it's a
   *ranking-confidence* effect, not a *disambiguation-capability* one —
   the system can tell these documents apart when the question gives it
   something to tell them apart *by*.
2. **The actual weak point is the bare natural citation number**
   (`natural_lookup_by_number`, 75% Recall@5, MRR 0.535 — worse than
   every other category, including the near-duplicate stress test).
   Inspecting the 5 misses shows a consistent, previously-unknown
   mechanism: the full CELEX ID (`31986R0480`) has a dedicated structured
   index (`chunks_by_celex_id`), but the natural short-form citation
   ("Regulation (EC) No 480/86") does not — so the system falls back to
   embedding/BM25 similarity on a query that's almost pure number, and
   that number collides across *other* documents:
   - `idf_natlookup_31993R0478` ("No 478/93"): answered confidently about
     a *different* document that merely cites "478/93" in passing, not
     the document itself.
   - `idf_natlookup_31986R0480` ("No 480/86"): retrieved **No 480/87** —
     an off-by-one-year collision on the same number.
   - `idf_natlookup_31986R0501` ("No 501/86"): explicitly told "the
     documents discuss various Regulations (EC) No 501, but do not
     specify which" — an honest report of exactly this collision.
   - `idf_natlookup_31980R0632`: correctly abstained rather than guess —
     the one case where the failure mode was "safe" (refused instead of
     hallucinating), still counted as a miss since retrieval genuinely
     failed.

**Net read**: the CELEX-ID structured-index shortcut wasn't hiding a
disambiguation problem (finding 1 rules that out) — it was hiding a real
but narrower gap: **the natural "Type No N/YY" citation format, one of
the most common ways this kind of document actually gets referenced in
prose, has no equivalent structured index and is measurably the weakest
retrieval path tested at this scale.** Everything else ID-free — subject
paraphrase, date-based lookup, cross-reference, and near-duplicate
disambiguation by date — held at or near 100%.

### Citation-number index — implemented and verified

Closed the gap above directly, mirroring `chunks_by_celex_id`/
`chunks_by_case_number` exactly:

- `ingestion/chunker.py::extract_citation_number()` — parses (number,
  4-digit year) from the same "No `<N>` `<YY[YY]>` of `<full date>`"
  preamble EURLEX57K filenames carry, using the *full* year from the
  trailing date clause (not the possibly-2-digit "No" year) so it's
  unambiguous by construction — this is exactly what resolves the
  480/86-vs-480/87 collision found above.
- `rag/retriever.py::extract_citation_numbers()` — parses the natural
  `"No N/YY"` form (with the slash, so it can't collide with
  `_CASE_NUMBER_RE`'s `"N of YYYY"` shape) out of query text, 2-digit
  years expanded the same way a reader conventionally resolves one.
  Wired into `retrieve_expanded()` and `promote_identity_matches()`
  exactly like `celex_id`/`case_number` — force-included above threshold,
  independent of reranker opinion.
- `vector_db/qdrant_client.py::chunks_by_citation_number()` +
  `citation_number`/`citation_year` payload indexes.
- `scripts/backfill_case_metadata.py` extended to backfill the new fields
  onto already-ingested points without re-parsing/re-embedding — run
  against `eu_scale_test` (57,000 docs): **189,663 chunks backfilled, 0
  skipped**. (Also run against the main `knowledge_base` collection for
  completeness — 0 matches, expected, that corpus has no EU-legislation
  filenames.)

**Re-ran `eu_id_free_dataset.json` @ 57,000 after the fix**
(`eval/scale_results/id_free_at_57000_post_citation_index.json`):

**Correction**: this run *was* affected by the window-counting bug — the
originally-published 108/110 (98.2%) "Recall@5" included 2 cases that only
landed at rank 6. The strict `rank ≤ 5` numbers below were recomputed and
then reconfirmed with a fresh live run (see "Recall@5 counting bug" in
`VALIDATION.md`).

| type | Recall@5 before fix | strict Recall@5 after fix | MRR before | MRR after |
|---|---|---|---|---|
| natural_lookup_by_number | 15/20 (75%) | **18/20 (90%)** | 0.535 | 0.631 |
| **overall** | 103/110 (93.6%) | **106/110 (96.4%)** | 0.884 | 0.901 |
| in-sources≤6 (not Recall@5) | | 108/110 (98.2%) | | |
| abstention accuracy | 117/120 (97.5%) | **120/120 (100%)** | | |

All 5 previously-missed `natural_lookup_by_number` cases spot-checked
individually post-fix, including the exact year-collision case
(`idf_natlookup_31986R0480`, "No 480/86") — now correctly resolves to the
480/86 document specifically (its answer cites "25 February 1986",
matching the expected document exactly, not the previously-confused
480/87). Recall@5 moved from 75% to a strict 90% for that type (2 of the
20 now land at rank 6, just outside the window); MRR (0.631, not 1.000)
still isn't perfect — same as `celex_id`, the index guarantees inclusion
above threshold, not top rank, so the reranker still independently decides
final position among candidates. That's expected and correctly not
"gamed" by this fix.

The remaining overall misses are `semantic_cross_reference` cases — now
partially addressed by the ground-truth fix described above (documents
that genuinely cite multiple earlier regulations now credit any of them,
not just the first regex match): substring-correctness on this type moved
12/20 → 13/20. The other misses are retrieval, not ground truth: the
citation the model surfaced wasn't among the ones `extract_facts()` found
in the document's first ~1,200 characters, which is a real, narrower
extraction-window limit, not a scoring artifact. **Every ID-free retrieval
path *drawn from questions the fix was tuned on* is now at or above 90%
strict Recall@5** — but see the dedicated generalization test below, which
was built specifically because re-testing the same questions that found a
bug doesn't prove the fix generalizes, and found a real gap that this
re-test alone would have missed entirely.

### Citation-number generalization test — a real, still-open gap

`eval/build_eu_citation_heldout_dataset.py` → `eval/eu_citation_heldout_dataset.json`
(50 cases). Built specifically to close the gap the section above admits:
the "100% Recall@5" citation-index result was re-tested on the *same 20
questions* whose failures motivated the fix in the first place. This set
draws 50 entirely different documents (every `celex_id` used anywhere in
`eu_golden_dataset.json` / `eu_heldout_dataset.json` /
`eu_cross_question_dataset.json` / `eu_id_free_dataset.json` is excluded,
and each candidate's short citation number is required to be unique across
the full 57,000-document corpus so the question has exactly one correct
answer) and cites each one five different natural ways:

| format | example | strict Recall@5 | MRR |
|---|---|---|---|
| `citation_fmt_bare_no` | "What does No 1021/2008 deal with?" | 10/10 | 0.950 |
| `citation_fmt_bracket_eec` | "What does Regulation (EEC) No 1021/2008 concern?" | 9/10 | 0.667 |
| `citation_fmt_commission_summarize` | "Summarize Commission Regulation (EC) No 1021/2008." | 6/10 | 0.420 |
| `citation_fmt_two_digit_year` | "...Regulation No 1021/08 of the European Communities?" | 5/10 | 0.373 |
| `citation_fmt_council_no_dot` | "What is Council Regulation No. 1021/2008 about?" | 3/10 | 0.337 |
| **overall** | | **33/50 (66.0%)** | **0.549** |
| in-sources≤6 (not Recall@5) | | 50/50 (100%) | |
| abstention accuracy | | 49/50 (98.0%) | |

**The index itself has no gap: all 50 documents were force-included by
`chunks_by_citation_number()` without exception** — the same guaranteed-
inclusion mechanism verified above holds up perfectly on entirely new
documents. What varies by phrasing is *rank*, and inspecting why turned up
a clean, specific cause: whenever the query names an issuing body
(Council/Commission) or bracket type (EEC/EC/EU) that doesn't match the
actual document, rank craters. Splitting all 40 cases that make such a
claim (`citation_fmt_bare_no` makes none, so it's excluded from this
split) by whether the claim is accurate:

| | n | strict Recall@5 |
|---|---|---|
| issuer/bracket in the query matches the document | 16 | **16/16 (100%)** |
| issuer/bracket in the query does *not* match the document | 23 | **12/23 (52%)** |

11 of the 23 mismatched cases land at exactly rank 6, not just "somewhere
outside top 5" — and the code explains why precisely.
`promote_identity_matches()` (`rag/retriever.py:94-131`) only *appends* an
identity-matched chunk missing from the reranked top-5; it deliberately
does not re-sort the list by score afterward (see the comment at line
110-115 — re-sorting risked cutting a just-boosted match back out). So
whenever the cross-encoder reranker scores the mismatched-phrasing chunk
below the top 5 — plausible on its own, since the query text ("Council
Regulation") no longer literally matches the chunk's title-prefixed
context ("Commission Regulation...") — the identity index still guarantees
it gets appended, but always lands in the newly-created 6th slot rather
than wherever its floor score would actually rank it. `citation_fmt_council_no_dot`
is the starkest example: it always says "Council Regulation," and every
one of its 10 target documents is actually a *Commission* regulation (an
accident of the corpus, not deliberate) — 3/10 strict Recall@5 as a direct
result.
`citation_fmt_two_digit_year` underperforms independent of this effect
(it makes no institution/bracket claim) — likely because the reranker's
cross-encoder compares raw query text against the chunk's title-prefixed
context, and a 2-digit year ("No 1021/08") has less literal overlap with
a document header that always spells the year in full ("...of 17 October
2008..."), even though `extract_citation_numbers()`'s year-expansion
correctly resolves the *lookup* itself (confirmed separately by
`tests/test_retriever.py::test_extract_citation_numbers_expands_two_digit_year`
and the index still force-including the document in 10/10 cases here).

**Net read**: the citation-index fix's original "100% Recall@5, gap
closed" claim was true for the question style it was tested on and false
as a general claim — a real, previously-undocumented weakness survives:
**the structured index protects recall unconditionally, but rank stays
exposed to how accurately the query names the issuing institution.** This
is not fixed as of this writing. Two options, not yet chosen between: (a)
drop the institution word from the index lookup key entirely — the number
and year are already sufficient to force-include the right document, so
the reranker's institution-mismatch penalty could be routed around by not
handing it a mismatched claim in the promoted chunk's scoring context, or
(b) treat `identity_match` promotion more aggressively for citation-number
hits specifically, since case (a) above already shows recall is never the
problem for this path, only rank.

### Final verdict

**Recall@5 degrades steadily across two orders of magnitude (400 → 57,000)
for queries that name their target document's exact CELEX ID** (see
caveat above; this dataset has no other kind) — strict `rank ≤ 5`: 96.0%/
95.3% (1,000) → 86.7%/84.0% (5,000) → 80.7%/78.0% (15,000) → 75.3%/74.0%
(30,000) → 72.7%/69.3% (57,000), golden/heldout. This is a correction of
an earlier published claim that Recall@5 was a flat 100% throughout — see
"Recall@5 counting bug" in `VALIDATION.md` for what was wrong and how it
was fixed and re-verified live. The corrected, honest headline for this
query pattern: this system's failure mode under scale, for ID-bearing
lookups, genuinely does include losing top-5 rank as the corpus grows —
not just "losing confidence" while staying in the window. What *does*
hold at 100% throughout is a strictly looser guarantee: the structured
CELEX-ID lookup always force-includes the correct document somewhere in
the returned set (a window that can be 6 wide, not 5) — real, and worth
stating precisely as "Recall@6 = 100%," not blurred into "Recall@5."

**MRR degraded monotonically and never plateaued within the tested range**
— 0.815/0.833 (400) → 0.448/0.453 (57,000, reconfirmed via fresh live
rerun), roughly a 45% relative drop, golden and heldout moving in lockstep
at every single checkpoint (never diverged by more than ~0.02). This part
of the original report was accurate throughout — MRR was never protected
by the window-counting bug, since a floor score guarantees inclusion, not
rank. Whether it keeps degrading, flattens, or eventually starts costing
Recall@5 further beyond 57,000 is genuinely unknown — that would need a
checkpoint beyond the tested range to answer.

**The first non-ranking failures appeared exactly in the 30,000–57,000
band, synchronized across three independently-measured signals** that had
been perfect (or near-perfect) at every prior checkpoint:
- Golden abstention accuracy: 159/160 (first miss ever on this dataset;
  `eu_31993D1021(02)_date` — model refused a date-lookup it should have
  answered, retrieval score well above `RELEVANCE_THRESHOLD` so the
  right chunk *was* found, the model just didn't extract the fact from it).
- Golden answer correctness: 54/55 (first miss ever on this dataset).
- Consistency retrieval stability: 14/15 (first instability ever observed
  on this test).

Each signal broke by exactly one case, not a cluster — this reads as the
onset of a trend, not a cliff. The mechanism is consistent with the MRR
story throughout: this corpus has an unusually high density of
near-duplicate boilerplate (the same regulation type — e.g. "fixing export
refunds on cereals" — reissued weekly/monthly for decades), and as more
near-identical variants accumulate, the reranker's confidence gets spread
thinner across them — which is now confirmed to cost actual top-5 rank at
this scale, not just MRR, and occasionally cost the deciding chunk or the
retrieved document outright.

**Concurrency and scale, tested together for the first time**:
`scripts/concurrency_test.py --requests 10` against the 400-document
instance (port 8000) shows 2.68x speedup on admitted concurrent requests
vs. sequential baseline — comfortably over the script's own 1.5x
"confirmed" threshold. The same test against the 57,000-document instance
(port 8001) shows only **1.18x** — under the script's own "little to no
gain" warning line. Sequential single-query latency also grew roughly
3x (2.3s → 6.9s avg) at this scale. The likely mechanism: reranking pools
several times more candidates per query at 57,000 documents (170-190
chunks reranked per query observed in this run, vs. a much smaller pool
at 400 documents), and that step is intentionally serialized through a
single-worker GPU executor (`rag/executors.py` — one physical GPU) — so
the GPU-bound stage increasingly dominates wall time at scale and eats
into the benefit concurrent requests would otherwise get from overlapping
I/O-bound work. Not a bug, and not previously measured: a real, now-
documented capacity ceiling for anyone planning to combine a large corpus
with multiple simultaneous users.

**A citation-index fix that looked complete wasn't, and a broader test
caught it**: see "Citation-number generalization test" above. The
structured lookup never misses on recall (50/50 documents force-included
on entirely new, never-tested documents), but rank collapses from 100% to
52% strict Recall@5 specifically when the query names the wrong issuing
institution (Council vs. Commission) or bracket type (EEC/EC/EU) — a
previously undocumented, still-open gap that a narrower re-test (the same
20 questions that found the original bug) would never have surfaced.

**What this does *not* establish** (explicit boundaries, not swept under
the rug):
- All of this is one corpus, one language, one question style
  (fact-extraction over legal/legislative text). It doesn't transfer
  automatically to, say, conversational support tickets or technical
  manuals — the near-duplicate density that drives the MRR trend is a
  property of *this* corpus, not a universal scale law (see "Data used"
  below).
- 57,000 was the ceiling of this test, not a proven system limit. The
  ladder stops there because that's the size of the available dataset,
  not because degradation stopped being interesting.
- The citation-phrasing rank-sensitivity gap found above is reported, not
  fixed — two candidate mitigations are noted in that section, neither
  implemented yet.

### Engineering tradeoffs worth considering (items 1-4, 6: not implemented — options)

0. ~~Add a structured index for the natural "Type No N/YY" citation
   format~~ — **implemented, verified against the questions that found the
   gap, and then re-tested against a disjoint generalization set that
   found a further, narrower gap** — see "Citation-number index" and
   "Citation-number generalization test" above. Recall is fully solved
   (index never misses, including on 50 brand-new documents); rank still
   degrades when the query names the wrong issuing institution. Item 6
   below is the direct follow-up.
1. **Widen the rerank/retrieval window for high-duplicate-density
   corpora.** `TOP_K_RESULTS=5` is what's been under test throughout;
   since the failure mechanism is "correct doc gets crowded to rank 2-6,"
   and strict Recall@5 is now confirmed to actually decay (72.7%/69.3% at
   57,000, not the originally-reported flat 100%), a corpus expected to
   have this kind of near-duplicate density could use a larger top-k as
   cheap insurance against the rank eventually crossing out of the window
   — costs a bit of latency/prompt budget, not a code change. This option
   is more clearly warranted now than it was under the original (wrong)
   100%-throughout reading.
2. **Attack the root cause instead of buying margin: near-duplicate
   detection/clustering at ingestion.** Collapse boilerplate reissues into
   a canonical document + variant pointers, so the reranker isn't being
   asked to distinguish near-identical text in the first place. Bigger
   lift than (1), but fixes the actual cause rather than compensating
   downstream — worth it only if the target corpus is genuinely expected
   to have this structure.
3. **Revisit `promote_document_opening_chunks`
   (`rag/retriever.py`) at this density.** That fix was tuned against the
   original ~400-doc court-case corpus's near-duplicate-title problem: it
   hasn't been re-tuned or re-validated specifically against the much
   higher duplicate density seen here — worth checking before assuming it
   scales unchanged.
4. **Don't over-build for a scale the real corpus won't reach.** If the
   production corpus is expected to stay in the hundreds-to-low-thousands
   range, most of the above is less urgent — Recall@5 was still 86.7%/
   84.0% at 5,000 documents and only starts dropping meaningfully past
   15,000. Match the mitigation to the actual expected corpus size, not to
   the worst case tested here — but note this is a real, measured curve
   now, not a flat "everything's fine below 57,000" line.
5. ~~If concurrency + scale matters for the real deployment, test that
   combination explicitly~~ — **done**, see "Concurrency and scale, tested
   together for the first time" in "Final verdict" above: 2.68x speedup at
   400 documents drops to 1.18x at 57,000 — a real, now-measured capacity
   ceiling, not an assumption.
6. **Close the citation-phrasing rank-sensitivity gap.** Root-caused, not
   just observed: `promote_identity_matches()` (`rag/retriever.py:94-131`)
   appends a match missing from the reranked top-5 without re-sorting by
   score afterward (the comment there explains this was deliberate, to
   avoid a just-boosted match getting sorted back out by a still-higher
   scorer) — so a mismatched-institution chunk that clears the relevance
   floor still always lands in the newly-created 6th slot instead of
   wherever its floor score would actually place it among the existing
   top 5. The untried fix: insert the promoted chunk at its score-sorted
   position within the existing top_k list (dropping the lowest-scored
   existing entry if the list would grow past top_k), instead of a flat
   append — directly addresses the "always exactly rank 6" pattern seen in
   11 of the 23 mismatched generalization-test cases above, without
   touching the append-vs-resort tradeoff for cases already working today.

### Data used, and why its shape matters

The ladder ran against **EURLEX57K** — a real, public corpus of EU
legislation (filenames follow CELEX ID convention, e.g.
`31958D1127(01) - EEC Council Rules of the Transport Committee.pdf`),
ingested into a fully isolated second instance (port 8001, `eu_scale_test`
Qdrant collection, `ragdb_eu_scale_test` Postgres DB — see "Isolation"
above), never touching the production 400-document corpus or its
calibrated `RELEVANCE_THRESHOLD`.

It was deliberately chosen to be **unrelated** to the existing court-case
corpus, so that a clean result wouldn't just be an artifact of that
corpus's specific shape. Its defining structural property — and the load-
bearing fact behind every finding above — is an **unusually high density
of near-duplicate documents**: EU legislation reissues the same regulation
type on a rolling basis (weekly/monthly, sometimes for decades), so the
corpus contains many documents that differ mainly in a date and a case
number while sharing near-identical boilerplate text. This is what makes
the corpus a *harder*, not merely a *bigger*, test than raw document count
suggests — and why the results above should be read as "how this system
behaves under heavy near-duplicate pressure at scale," not as a general
"any 57,000-document corpus" claim. A corpus of 57,000 genuinely distinct
documents might show a flatter MRR curve than this one did.

---

## Plain-language summary (for a non-technical reader)

**What was tested.** The RAG system (search + answer generation over PDF
documents) was deliberately loaded with a base of 57,000 documents — real
EU legislation, completely unrelated to the company's actual working
corpus (which stays around ~400 documents). This wasn't a "test for luck"
— a large, unrelated dataset was picked on purpose, to check the system's
behavior as volume grows, not a lucky result on one small case.

**Why this specific data, and what makes it tricky.** This dataset has an
unusually high number of near-identical documents: the same type of
regulation gets reissued weekly or monthly for decades, differing mostly
in date and number. This is a deliberately "unfriendly," stress-test
scenario — if the system holds up here, that's a stronger result than a
test on a set of clearly distinct, unrelated documents.

**An important catch, found after the fact — and then checked directly.**
Every test question in the original dataset names the target document's
exact ID (e.g. "What is 32011D0126 about?"). The system has a shortcut
specifically for that case: if a question names a document's exact ID,
that document is force-included in the results, bypassing the normal "how
relevant is this" scoring. So the original headline "the right document
was always found" result mostly proved that shortcut keeps working as the
database grows — not that the system would find the right document as
reliably for a natural question that *doesn't* name the ID, which is how
most real users would actually ask.

A 120-question follow-up test with no ID in any question was built and
run to check this directly. Good news first: even asking about documents
buried in clusters of 500+ near-identical near-duplicates, using only a
natural date reference (no ID), the system found the right one **every
single time**. One real gap it uncovered: questions that referred to a
document only by its short natural citation number (e.g. "Regulation No
480/86", as a person familiar with EU law might actually say it, rather
than its formal ID) succeeded only 75% of the time — including one case
where it confused a 1986 regulation with an unrelated 1987 one sharing the
same short number. A dedicated lookup for this short-citation form was
added (mirroring the one that already existed for the formal ID), and
re-testing the same questions that found the bug confirmed it now
succeeds on that exact question type. That looked like the end of the
story — it wasn't, see below.

**A second, harder catch, found by a documentation review — and then
checked directly, the same way the first one was.** A later review of the
test scripts themselves found that "the right document was in the top 5"
was being measured wrong: the system can return up to six sources for a
five-result request (the exact-ID shortcut above sometimes adds an extra
one), and a document landing in that 6th slot was being counted as a
top-5 hit. Recomputed with the strict, correct definition, "the right
document was always in the top 5" turns out to be false at any real
scale — it starts at 96% (1,000 documents) and drops to 73% by 57,000.
The looser, real thing that stayed true throughout is closer to "the
right document was always somewhere in the first six results" — still a
solid result, just a different and smaller claim than originally
published, and every number below was re-run live against the actual
57,000-document system to make sure the correction itself was right, not
just re-typed.

Re-testing the citation-number fix on the same questions that found the
original bug also turned out to be too easy an exam. A new, disjoint set
of 50 questions — different documents, five different natural ways to
phrase a citation — found that the lookup itself never fails (the right
document is always somewhere in the results), but *where* it ranks
depends heavily on phrasing: asking about "Commission Regulation No X/Y"
when it's actually a Commission regulation (or not naming the issuing
body at all) put it in the top 5 100% of the time; asking about "Council
Regulation No X/Y" when it's actually a Commission regulation — an easy,
realistic mistake — dropped that to 52%, usually landing the document
exactly one place outside the visible results. This is now a documented,
open gap, not a fixed one.

**What was found, corrected version.**
- The system never — not at 400 documents, not at 57,000 — completely
  "lost" the right document (for questions naming the exact document ID):
  the correct file was always among the first six results. It was not,
  however, always in the top five — that number declines steadily as the
  document base grows (see above), which the original write-up got wrong.
- The more documents there were, the more often the correct document
  slipped from 1st place to somewhere lower, and — more often than
  originally reported — slipped out of the visible top-5 results
  entirely, landing 6th instead.
- At the largest volumes (in the 30,000-57,000 range) a handful of
  isolated misses (one case per check) showed up for the first time: one
  unwarranted refusal to answer, one wrong answer, one case where asking
  the same question twice returned two different source documents (though
  the final answer was correct both times). Below that range, none of
  these misses occurred at all.
- Concurrent-request handling, tested for the first time against the full
  57,000-document base: the system's usual ~2.7x speedup from handling
  requests concurrently drops to roughly 1.2x at this scale — a real,
  now-measured limit for anyone planning to combine a large document base
  with several simultaneous users, not the "probably fine" it was before.
- In plain terms: the system doesn't break sharply or suddenly start
  failing en masse, but it becomes measurably less reliable — not just
  less confident — as both the document count and the realism of the test
  itself increased. Each round of "let's actually check that" (the ID
  shortcut, the metric itself, the citation-fix generalization) found a
  real gap the previous round had missed or overstated.

**Honestly, what was NOT tested, and what's still an open gap.**
- The citation-phrasing rank sensitivity found above (wrong issuing body
  named in the question) is reported, not fixed. Two candidate fixes are
  identified in the technical detail above; neither is implemented yet.
- Whether degradation keeps getting worse past 57,000 documents, or stops
  there, is unknown — testing wasn't taken further (that was the limit of
  the available dataset).
- The result is specific to this kind of data (lots of near-duplicates).
  On a corpus of fully distinct documents, degradation at the same volume
  could be smaller, or might not show up at all.

**Tradeoffs worth discussing, if this becomes relevant.**
- If the real document base stays in the hundreds-to-low-thousands range,
  most of the warning signs found here are less urgent — Recall@5 was
  still 86.7%/84.0% at 5,000 documents, well past a realistic small
  deployment — but note this is a real, gradually declining curve now,
  not the flat "everything's perfect below 57,000" line originally
  reported.
- If the base grows to tens of thousands of *similar* documents, there
  are two options: (a) cheaply widen the search window (7-10 candidates
  instead of 5) as insurance, or (b) detect and group near-duplicates at
  upload time — more development work, but it removes the actual cause
  instead of just buying a safety margin.
- If many users working simultaneously on a large database matters, that
  combination has now actually been tested (see above) and does show a
  real, measurable slowdown — worth planning around directly, not
  assuming away.
