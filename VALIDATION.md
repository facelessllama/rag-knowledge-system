# What actually backs this project up (plain-language)

Most "I built a RAG app" portfolio pieces show a screenshot of one good
answer. That proves the happy path exists — it doesn't prove the system
holds up once you stop being gentle with it. This document is about the
second thing: what was actually checked, how, and why that's meaningful
evidence rather than a demo.

The short version: the system was checked the way a skeptical client
would check it, not the way a developer checks their own work. That
included deliberately trying to break it, finding real bugs and fixing
their actual cause, and — twice now — going back to question whether the
*test itself* was honest, finding out it wasn't quite, and fixing that
too. The second of those two times is why this document has almost
nothing in common with its own first draft: a review of the eval tooling
found that the headline `Recall@5 = 100%` claim below was based on a
miscounted metric, and every number in this document was recomputed and
independently re-verified live against the same 57,000-document corpus
before being republished. What that review found is described in full,
including two real, previously-undiscovered gaps it surfaced along the
way — not smoothed over.

## What was actually checked

**1. Does it get the basics right, reliably — not just once?**
On the system's real working set (~400 legal documents), it was tested
against two separate question sets: one used while tuning it, and a
second, untouched one built specifically to catch "it only works because
we accidentally tuned it to this exact test." Both came back with
Recall@5 = 100% and answer correctness at 98.8%/100%, reproduced across
repeated runs. This is the one part of this document where "100%" is not
an overstatement — this corpus is small enough (~400 documents) that the
window-counting bug described below never had room to matter, confirmed
directly by recomputing every saved case's actual rank.

**2. Does it fall apart as the amount of data grows?**
The working set is small. To find out what happens at real scale, the
system was loaded with 57,000 unrelated real documents (EU legislation)
— picked specifically because that dataset is full of near-identical
documents (the same type of filing reissued monthly for decades), which
is about the hardest condition for a search system to stay accurate
under. Volume was ramped up in stages, testing at each step, for
questions that name their target document's exact ID.

**The honest result, not the first-draft one**: the correct document's
*rank* degrades steadily as the corpus grows — from 96%/95% of questions
landing in the top 5 at 1,000 documents down to 73%/69% at 57,000 — a
real, gradual decline, not a cliff. A structured lookup index (described
below) does still force the correct document into a slightly wider
6-document window on every single question at every checkpoint, so the
document is never fully "lost" — but presenting that as "Recall@5 = 100%"
was wrong, and calling it "Recall@6 = 100%" instead is the honest version
of the same fact. See "The metric bug" below for how this was found and
fixed.

**3. Was the test itself telling the truth?**
Partway through, it became clear that every test question happened to
include the exact filing number of the document it was asking about —
which meant the system could be quietly "cheating" via a shortcut built
for exactly that case, rather than genuinely searching. Instead of
leaving that unexamined, a second test was built from scratch that
specifically avoided giving the system that shortcut, using only the
kind of vague, natural phrasing a real person would actually type. On the
57,000-document corpus, that test found:
- Natural lookup by subject/paraphrase, date-based facts, and
  disambiguating one specific document out of a 509-member cluster of
  near-duplicates using only its adoption date: **100% Recall@5, every
  time.** The near-duplicate-disambiguation worry the whole scale test
  started from turned out not to be the weak point.
- The one genuine weak spot: short-form citation references (like "the
  regulation numbered 480 of '86") were sometimes confused with unrelated
  documents that merely mention that number in passing — including one
  case that confused two different documents from different years
  sharing the same short number. That was root-caused and fixed with a
  dedicated lookup index; re-testing the same 20 questions that found the
  bug went from 75% to 100% Recall@5.

Re-testing the questions that *found* a bug and confirming the fix is not
the same as confirming the fix generalizes — so a **third**, disjoint set
of 50 questions was built afterward, covering documents and citation
phrasings the fix had never seen. That test is what surfaced the next
finding.

**4. Does the fix generalize, or did it just pass its own exam?**
The new 50-question set varied both the documents (none used in any
earlier test) and the phrasing (five different natural ways to cite a
regulation by number). Result: the target document was force-included by
the lookup index in all 50 cases without exception — the index itself
never misses. But **whether it also landed in the top 5** depended
heavily on phrasing: among the 30 questions that actually name an issuing
body or bracket type, getting it right put the document in the top 5 in
6 out of 6 cases with a checkable filename (100%). When the question
named the *wrong* issuing body — a plausible real mistake, since
regulations get referred to loosely in speech — rank fell to 12 out of 23
(52%), with the miss usually landing at exactly rank 6, just outside the
window a user would naturally read. This is a real,
previously-undocumented gap: the structured index guarantees the document
is *found*, but a wrong institution word in the query still measurably
hurts where it *ranks*. Not fixed yet — see `eval/README.md` for the
detail and the options for closing it.

**5. Does it change its mind if you ask the same thing twice?**
The same questions were repeated dozens of times each, checking not just
whether the final answer stayed correct, but whether the system was even
pulling up the same source document each time. At 57,000 documents: 300/300
final answers correct across repeats, but retrieval picked a different top
document on one question out of fifteen — the first instability of this
kind seen at any checkpoint. Read as a leading indicator, not a failure:
generation papered over the retrieval flip that time, but it's a signal
worth watching, not one to hide.

**6. Does load and scale combine safely?**
Never tested together before this review. Now measured directly:
concurrent-request speedup drops from 2.68x on the small 400-document
corpus to 1.18x at 57,000 documents — the system's own concurrency test
script flags anything under 1.5x as "little to no gain," and 57,000
documents crosses that line. The likely mechanism: reranking has to sort
through several times more candidates per query at this scale, and that
step runs on a single dedicated GPU worker by design (only one physical
GPU) — so the GPU-bound stage increasingly dominates wall time and eats
into the benefit that overlapping requests would otherwise get from
async I/O. Not a bug — a real, now-documented capacity limit for anyone
planning to run this system with both a large corpus and multiple
simultaneous users.

## The metric bug, and how it was found

A later review of the eval tooling (`eval/run_eval.py`) noticed that
`Recall@5` was computed by checking whether the expected document
appeared *anywhere* in the API's returned `sources` array — but that
array can hold six documents even when the request asked for five,
because a separate mechanism (an exact-ID structured-lookup override)
force-adds the correct document above the requested cutoff when the
query names it directly. A document landing at rank 6 was therefore
being counted as a "Recall@5" hit throughout the scale-ladder results.

This was not a cosmetic difference: recomputing the same saved results
with the strict `rank ≤ 5` definition dropped the headline from a flat
100% at every corpus size to a real decline from 96% down to 73% as the
corpus grew to 57,000 documents. The fix was made in three parts, all
independently verified rather than assumed:
1. `run_eval.py` now only counts a hit when `rank ≤ top_k`, and reports
   the old, looser "somewhere in the returned window" number alongside it
   under an explicit label so the two can never be confused again.
2. Every existing scale-ladder result was recomputed from the raw
   per-case data already saved on disk (nothing needed re-running to
   catch the mistake — the rank of every case had been recorded all
   along, just aggregated with the wrong cutoff).
3. Every affected live dataset (golden, held-out, cross-question,
   ID-free, plus the new 50-question generalization set) was then
   **re-run live against the real 57,000-document corpus** with the
   corrected script, not just recomputed from old numbers — closing the
   gap between "the math was fixed" and "the fix was actually verified
   against the live system." All of it matched the offline recomputation
   within normal run-to-run LLM variance (typically ±1 case out of 150).

Two smaller, related issues were found and fixed in the same pass:
- One of the "hard, disambiguate-by-date-only" test cases turned out to
  have an ambiguous ground truth itself — three documents in its
  near-duplicate cluster shared the exact same adoption date, so the
  question had more than one defensible correct answer. The dataset
  builder now requires a chosen date to be unique across the *entire*
  cluster, not just among the specific documents it had already picked.
- Questions asking what an earlier regulation a document cites/amends
  only accepted the *first* citation the extraction script happened to
  find, even though EU legal preambles routinely cite several earlier
  regulations legitimately. The ground truth now accepts any citation the
  document body actually supports — this was a real ground-truth defect,
  not a retrieval or generation defect, and conflating the two would have
  kept blaming the system for a scoring mistake.

## The cross-domain corpus test, and what it found

Everything above uses two corpora the system has effectively been tuned
around: ~400 legal documents and 57,000 EU regulations — both legal text,
both fed through the same filename conventions the ingestion code already
special-cases. To find out what happens on content the system has never
seen and was never shaped by, it was pointed at 762 real, public-source
PDFs it had no prior exposure to: 597 current arXiv papers, 45 FDA drug
labels, 16 FAA safety manuals, and 104 deliberately adversarial pages from
the olmOCR-bench dataset (old scans, dense mathematical notation,
multi-column layouts, repeated headers/footers, very small text) —
selected specifically to stress the parts a clean legal-PDF corpus never
exercises. The corpus was frozen with a seeded discovery/calibration/
heldout split before any error analysis began, so what follows is what a
completely generic, untuned pass through the ingestion pipeline actually
does, not a result already shaped by looking at the failures first.

A pre-flight check (parsing every document locally, the same parser/
chunker classes the live API uses, without touching the database) had
already predicted which documents the API's own "could not extract text"
rejection would catch: 12 of 762, concentrated in the old-scan and
handwritten-math slices, where plain OCR — no math or handwriting model —
has no real chance. Running the actual documents through a live,
isolated instance (separate database, separate vector collection, so this
never touched the corpora above) confirmed exactly that prediction — 12
of 12 — and turned up two things the offline check couldn't have found,
because they only happen once real storage is involved:

**A real, previously-undiscovered bug**: 4.6% of pages (571 of 12,334,
measured directly off the corpus's PDFs) across the 597-document arXiv
sample carried a literal NUL byte in their extracted text — invisible in
any PDF viewer, a font-encoding artifact some PDF producers (arXiv's own
pipeline among them) emit, which PyMuPDF passes through verbatim. A NUL
on any single page aborts that whole document's ingestion (one Postgres
error rolls back the document, not just the affected chunk), so at the
document level this was 36.0% of the sample (215 of 597, counting both
affected page text and affected PDF metadata fields) failing outright
with "string literal cannot contain NUL (0x00) characters" — that had
nothing to do with the adversarial content the corpus was built to test.
This is a real bug that would affect any user uploading an affected PDF
through the ordinary upload path — not an artifact of this test's unusual
content. Fixed at the point of extraction (both the page text and PDF
metadata paths in `ingestion/pdf_parser.py`), with a second, independent
guard added at the chunking layer for any other text that might reach it
by a different route, six new regression tests, and confirmed by
re-running the exact previously-failing documents through the live API
before resuming the full corpus — all 215 now ingest cleanly.

**A second, smaller finding, since fixed**: two of the 762 documents (the
largest ones in the corpus — a 245-page arXiv paper and the 350-page FAA
`prh_change1` handbook) failed a different way — a single Qdrant upsert
request exceeded Qdrant's own 32 MB payload limit
(`vector_db/qdrant_client.py::upsert_chunks` sent every one of a
document's chunks in one call, regardless of how many). This is a
distinct bug from the one above (a batching limit, not a text-encoding
one): the corpus is well within the application's own 50 MB upload limit,
so a real, permitted upload could fail at the vector-store step purely
from having enough chunks. Fixed by batching upserts on actual serialized
point size (24 MB threshold, headroom under Qdrant's 32 MB limit) instead
of one call per document; confirmed by re-uploading both exact
previously-failing documents through the live API — both now ingest
cleanly (1,494 and 1,806 chunks respectively, 2 upsert batches each).

**The result, after both fixes**: 744 of 762 documents (97.6%) ingested
successfully; 12 were correctly rejected as unreadable exactly where
predicted; 6 more were flagged "already uploaded" by the system's own
duplicate-content guard after an unrelated mid-run restart during
testing — not corpus duplicates, confirmed by checksum. No fact-based
accuracy claims are made on this corpus yet; this pass covers ingestion
only, the step every later accuracy number depends on.

Fact-based accuracy, evidence-chunk-level recall, and a conditional
generator benchmark (local Qwen vs. an evaluated-only DeepSeek cloud
backend, on cases where retrieval is already confirmed correct) were
measured in later passes on this same corpus — see
`eval/mixed_corpus/README.md`, not duplicated here.

## A real localization bug, found and fixed — then a real product decision

Document-level recall (above) answers "was the right document found?" —
not "was the specific fact on page 40 of it actually retrieved?" A later
pass measured that second, harder question directly and started at
19.9%: a genuine chunk-level localization failure document recall alone
can't see, hiding in plain sight behind a 100%-looking retrieval story.

Chasing it down found one precise, specific bug, not a vague "retrieval
needs tuning": a code path meant to guarantee an exact match (a cited
`Figure 7` or `Table 3`) survived reranking only worked when ordinary
search *hadn't* already found that same chunk on its own, however weakly
— backwards, since a weak initial match is exactly when the guarantee is
needed. Fixing that single guard raised the number to 91.5%, then 95.5%
once a second, related gap (the same guarantee silently skipped when a
question was scoped to one already-open document — the realistic
"user has a document open" workflow) was found and fixed the same way.

The generation side raised a different question: is a paid cloud model
(DeepSeek) worth adding at all? An honest answer needs a control most
one-off comparisons skip — identical evidence, scored identically, for
both models — otherwise "the cloud model won" can just mean "the cloud
model got easier questions." With that control in place, the cloud model
showed a real ~15-25 point accuracy edge on confirmed-evidence cases.
A second check, closer to how the product is actually used, made that
edge look like it had nearly vanished — and reading the disputed answers
by hand, rather than trusting the automated scorer, found the scorer
itself was behaving inconsistently on short questions, not the model. A
real edge remained once that was corrected, just a smaller one than the
first number suggested.

The product now ships that decision, not just the finding: local Qwen
stays the default and the only mode with zero external data flow; the
cloud model is available strictly opt-in, gated by both an administrator
setting and a per-request flag, with tests proving the second one can't
be skipped by accident and a request that asks for it gets a clear error
— never a silent answer from a different model — if either gate isn't
satisfied. Full detail, including the exact bug, the taxonomy that found
it, and the blind-adjudication method that resolved the generator
question: `eval/mixed_corpus/README.md` and `eval/mixed_corpus/
EXPERIMENT_HISTORY.md`.

## Why this is a meaningful proof point, and not just a bigger demo

- **The numbers are real pass rates on datasets built to be hard to
  game**, not a curated set of questions chosen because they're known to
  work — and, as of this pass, verified against a metric that was itself
  checked for the same kind of self-flattery it was designed to catch.
- **Every bug listed above was found by trying to break the system on
  purpose**, and fixed at its actual root cause — not patched around it.
- **When a result looked too good, that was treated as a reason to
  double-check, not celebrate — twice.** The CELEX-ID shortcut issue was
  the first example: a 100% score was investigated instead of accepted.
  The Recall@5 miscount was the second, and it was caught in the
  *reporting*, not the system — a flat 100% across a 140x growth in
  corpus size should have been the tell, and eventually was.
- **What wasn't tested is stated as plainly as what was.** Concurrency at
  scale was an explicit open question until this pass closed it (with a
  real, non-trivial finding — 1.18x, not the 2.68x seen at small scale).
  The citation-phrasing rank sensitivity found in check 4 above is *still
  open* — it is reported here specifically because it was found by trying
  to disprove an already-published "fixed" claim, not because it makes
  the system look good.

This is what "the system works" means here: not a good-looking chat
transcript, and not a self-consistent-but-wrong metric, but a chain of
tests specifically designed to catch the system — and the test suite
itself — lying, followed by fixing what that turned up and re-verifying
it live rather than assuming the fix.

For the full numbers, methodology, and every case-by-case detail,
including the per-format citation-phrasing breakdown and the concurrency
measurements, see `eval/README.md`.
