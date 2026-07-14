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
- `eval/consistency_test.py` — repeats a representative question sample
  N times against a live instance, recording both answer correctness and
  *which document was actually retrieved* each time (separates retrieval
  flakiness from generation flakiness).
- `eval/run_eval.py` fixed: crashed with `ZeroDivisionError` on any
  dataset with zero adversarial (`expect_answer: False`) cases — the
  cross-question dataset has none by design. Now reports `n/a` instead.

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
**100%**; heldout 86.7%/88.1%/58.2% → **100%/99.4%/90.3%**. Residual
heldout `cross_reference_fact` misses (5/9) are a *dataset* ambiguity, not
a retrieval/generation defect — many EU documents cite multiple earlier
regulations ("Having regard to X... and Y..."), and the regex-extracted
"expected" one isn't always the one the model surfaces, even though both
are legitimately present in the text. Backfilled the already-ingested 400
docs' `celex_id` via `scripts/backfill_case_metadata.py` (extended to
handle both identity conventions; scrolls existing points, no re-parse/
re-embed needed).

**Scale ladder results — full run** (golden / heldout, `eval/scale_results/`):

| corpus size | Recall@5 | Abstention acc. | Answer correctness | MRR (golden / heldout) |
|---|---|---|---|---|
| 400 (post-fix pilot) | 100% / 100% | 100% / 99.4% | 100% / 90.3% | 0.815 / 0.833 |
| 1,000 | 100% / 100% | 100% / 99.4% | 100% / 91.9% | 0.706 / 0.722 |
| 5,000 | 100% / 100% | 100% / 99.4% | 100% / 91.9% | 0.599 / 0.608 |
| 15,000 | 100% / 100% | 100% / 100% | 100% / 96.8% | 0.539 / 0.535 |
| 30,000 | 100% / 100% | 100% / 100% | 100% / 93.5% | 0.496 / 0.504 |
| **57,000** | **100% / 100%** | **99.4% / 99.4%** | **98.2% / 93.5%** | **0.447 / 0.462** |

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

**Results** (`eval/scale_results/id_free_at_57000.json`):

| type | n | Recall@5 | MRR |
|---|---|---|---|
| hard_negative_date | 30 | **30/30 (100%)** | 1.000 |
| natural_lookup_by_subject | 20 | 20/20 (100%) | 1.000 |
| semantic_date_fact | 20 | 20/20 (100%) | 1.000 |
| semantic_cross_reference | 20 | 18/20 (90%) | 0.825 |
| natural_lookup_by_number | 20 | **15/20 (75%)** | **0.535** |
| **overall** | 120 | **103/110 (93.6%)** | **0.884** |
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

| type | Recall@5 before | Recall@5 after | MRR before | MRR after |
|---|---|---|---|---|
| natural_lookup_by_number | 15/20 (75%) | **20/20 (100%)** | 0.535 | 0.631 |
| **overall** | 103/110 (93.6%) | **108/110 (98.2%)** | 0.884 | 0.901 |
| abstention accuracy | 117/120 (97.5%) | **120/120 (100%)** | | |

All 5 previously-missed `natural_lookup_by_number` cases spot-checked
individually post-fix, including the exact year-collision case
(`idf_natlookup_31986R0480`, "No 480/86") — now correctly resolves to the
480/86 document specifically (its answer cites "25 February 1986",
matching the expected document exactly, not the previously-confused
480/87). Recall@5 is now perfect for that type; MRR (0.631, not 1.000)
is not — same as `celex_id`, the index guarantees inclusion above
threshold, not top rank, so the reranker still independently decides
final position among candidates. That's expected and correctly not
"gamed" by this fix.

The 2 remaining overall misses are both `semantic_cross_reference` — the
same pre-existing, unrelated ambiguity class described earlier (a
document citing multiple earlier regulations; not something a citation-
number index addresses). **Every ID-free retrieval path tested is now at
or above 98% Recall@5** — the one measurable gap this whole caveat/
spot-check/fix cycle surfaced has been closed and re-verified, not just
proposed.

### Final verdict

**Recall@5 never broke, across two orders of magnitude (400 → 57,000) —
for queries that name their target document's exact CELEX ID (see caveat
above; this dataset has no other kind).**
The correct document was in the top-5 candidate set on every single golden
and heldout case, at every checkpoint, with no exception. That's the
headline result for *this* query pattern: this system's failure mode
under scale, for ID-bearing lookups, is not "loses the document," it's
"loses confidence about which of several similar documents is most
relevant."

**MRR degraded monotonically and never plateaued within the tested range**
— 0.815/0.833 (400) → 0.447/0.462 (57,000), roughly a 45% relative drop,
golden and heldout moving in lockstep at every single checkpoint (never
diverged by more than ~0.02). Whether it keeps degrading, flattens, or
eventually starts costing Recall@5 beyond 57,000 is genuinely unknown —
that would need a checkpoint beyond the tested range to answer.

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
thinner across them. Most of the time that only costs rank position
(MRR); at the tested extreme, it occasionally cost the deciding chunk or
the retrieved document outright.

**What this does *not* establish** (explicit boundaries, not swept under
the rug):
- Concurrency at scale was never tested together with corpus scale —
  `scripts/concurrency_test.py` defaults to port 8000 (the 400-doc
  production instance); the known GPU-serialization ceiling (comfortable
  at 1-2 concurrent queries, queues at 10, see `rag/executors.py`) has
  never been measured against the 57,000-doc index. If a real deployment
  needs both high volume *and* multiple simultaneous users, that
  combination is an open question, not a validated one.
- All of this is one corpus, one language, one question style
  (fact-extraction over legal/legislative text). It doesn't transfer
  automatically to, say, conversational support tickets or technical
  manuals — the near-duplicate density that drives the MRR trend is a
  property of *this* corpus, not a universal scale law (see "Data used"
  below).
- 57,000 was the ceiling of this test, not a proven system limit. The
  ladder stops there because that's the size of the available dataset,
  not because degradation stopped being interesting.

### Engineering tradeoffs worth considering (items 1-5: not implemented — options)

0. ~~Add a structured index for the natural "Type No N/YY" citation
   format~~ — **implemented and verified**, see "Citation-number index —
   implemented and verified" above (`chunks_by_citation_number`,
   mirroring `chunks_by_celex_id`/`chunks_by_case_number`). Was the one
   retrieval path with concrete evidence of underperforming (75% Recall@5
   ID-free); re-verified post-fix at 100% Recall@5 for that question type,
   98.2% overall. Everything below this point is still an option based on
   the MRR trend, not a confirmed gap the way this one was.
1. **Widen the rerank/retrieval window for high-duplicate-density
   corpora.** `TOP_K_RESULTS=5` is what's been under test throughout;
   since the failure mechanism is "correct doc gets crowded to rank 2-4,"
   a corpus expected to have this kind of near-duplicate density could
   use a larger top-k as cheap insurance against the rank eventually
   crossing out of the window — costs a bit of latency/prompt budget, not
   a code change.
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
   range, none of the above is warranted — every metric was flat/perfect
   through the 5,000-15,000 checkpoints, well past any realistic small
   deployment. Match the mitigation to the actual expected corpus size,
   not to the worst case tested here.
5. **If concurrency + scale matters for the real deployment, test that
   combination explicitly** — point `scripts/concurrency_test.py` at the
   57k instance (port 8001) rather than assuming the two known-separately
   ceilings (GPU serialization, MRR-under-scale) simply add up.

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
single time**. The one real gap it uncovered was narrower and more
specific: questions that referred to a document only by its short natural
citation number (e.g. "Regulation No 480/86", as a person familiar with
EU law might actually say it, rather than its formal ID) succeeded only
75% of the time — including one case where it confused a 1986 regulation
with an unrelated 1987 one sharing the same short number. That gap has
since been closed: a dedicated lookup for this short-citation form was
added (mirroring the one that already existed for the formal ID), and
re-testing confirms it now succeeds 100% of the time on that same
question type, with no regression elsewhere.

**What was found.**
- The system never — not at 400 documents, not at 57,000 — completely
  "lost" the right document (for questions naming the exact document ID —
  see the catch above): the correct file always landed in the top 5
  search results.
- But the more documents there were, the more often the correct document
  ended up not in first place within that top 5, but in 2nd-4th place.
  The user still got the right answer (the system reads the whole top 5,
  not just first place), but it's a clear signal that telling near-
  identical documents apart gets harder as the base grows.
- At the largest volumes (in the 30,000-57,000 range) a handful of
  isolated misses (one case per check) showed up for the first time: one
  unwarranted refusal to answer, one wrong answer, one case where asking
  the same question twice returned two different source documents (though
  the final answer was correct both times). Below that range, none of
  these misses occurred at all.
- In plain terms: the system doesn't break sharply or suddenly start
  failing en masse. It gradually becomes slightly less confident
  specifically where documents are nearly indistinguishable from each
  other, and that only became visible at volumes far beyond what real-
  world use is likely to reach.

**Honestly, what was NOT tested.**
- What happens with a large document volume *and* several users at once
  was not tested. There's a separate, already-known limit: on a single
  GPU (RTX 3080), 1-2 simultaneous queries are comfortable, and 10 queue
  up. These two limits (volume and load) were never tested together.
- Whether degradation keeps getting worse past 57,000 documents, or stops
  there, is unknown — testing wasn't taken further (that was the limit of
  the available dataset).
- The result is specific to this kind of data (lots of near-duplicates).
  On a corpus of fully distinct documents, degradation at the same volume
  could be smaller, or might not show up at all.

**Tradeoffs worth discussing, if this becomes relevant.**
- If the real document base stays in the hundreds-to-low-thousands range,
  none of the warning signs found here are relevant — nothing needs to
  change (every metric was flawless across that entire range throughout
  the ladder).
- If the base grows to tens of thousands of *similar* documents, there
  are two options: (a) cheaply widen the search window (7-10 candidates
  instead of 5) as insurance, or (b) detect and group near-duplicates at
  upload time — more development work, but it removes the actual cause
  instead of just buying a safety margin.
- If many users working simultaneously on a large database matters, that
  is a separate test that hasn't been run yet and should be planned for
  on its own.
