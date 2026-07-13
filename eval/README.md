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
