# What actually backs this project up (plain-language)

Most "I built a RAG app" portfolio pieces show a screenshot of one good
answer. That proves the happy path exists — it doesn't prove the system
holds up once you stop being gentle with it. This document is about the
second thing: what was actually checked, how, and why that's meaningful
evidence rather than a demo.

The short version: the system was checked the way a skeptical client
would check it, not the way a developer checks their own work. That
included deliberately trying to break it, finding real bugs and fixing
their actual cause, and — the part most write-ups skip — going back to
question whether the *test itself* was honest, finding out it wasn't
quite, and fixing that too.

## What was actually checked

**1. Does it get the basics right, reliably — not just once?**
On the system's real working set (~400 legal documents), it was tested
against two separate question sets: one used while tuning it, and a
second, untouched one built specifically to catch "it only works because
we accidentally tuned it to this exact test." Both came back with the
correct document found essentially every time, and correct answers
essentially every time, reproduced across repeated runs — not a single
lucky pass.

**2. Does it fall apart as the amount of data grows?**
The working set is small. To find out what happens at real scale, the
system was loaded with 57,000 unrelated real documents (EU legislation)
— picked specifically because that dataset is full of near-identical
documents (the same type of filing reissued monthly for decades), which
is about the hardest condition for a search system to stay accurate
under. Volume was ramped up in stages, testing at each step. Result: the
system never once lost the correct document out of the running, but it
gradually became less confident about which document was the best match
as the pile of near-duplicates grew — and the first small, real mistakes
(not many — one at a time) only showed up at the very top of the range
tested, not before. That's a meaningfully different, and more honest,
finding than "it works" or "it breaks" — it shows *where* the strain
starts to show, under a deliberately unfavorable condition.

**3. Was the test itself telling the truth?**
Partway through, it became clear that every test question happened to
include the exact filing number of the document it was asking about —
which meant the system could be quietly "cheating" via a shortcut built
for exactly that case, rather than genuinely searching. Instead of
leaving that unexamined, a second test was built from scratch that
specifically avoided giving the system that shortcut, using only the
kind of vague, natural phrasing a real person would actually type. That
test found a real, narrower weak spot: short-form references (like "the
regulation numbered 480 of '86") were sometimes confused with unrelated
documents that merely mention that number, and once even confused two
different documents from different years sharing the same short number.
That gap was fixed and re-tested — the failure rate on that specific
question type went from 1-in-4 wrong to 0.

**4. Does it change its mind if you ask the same thing twice?**
The same questions were repeated dozens of times each, checking not just
whether the final answer stayed correct, but whether the system was even
pulling up the same source document each time. It was — consistently,
including at large scale — which rules out the specific worry that a
system like this quietly becomes unreliable on repeat use even when a
single test looks fine.

## Why this is a meaningful proof point, and not just a bigger demo

- **The numbers are real pass rates on datasets built to be hard to
  game**, not a curated set of questions chosen because they're known to
  work.
- **Every bug listed above was found by trying to break the system on
  purpose**, and fixed at its actual root cause — not patched around it.
- **When a test result looked too good, that was treated as a reason to
  double-check, not celebrate.** The CELEX-ID shortcut issue is the clearest
  example: a 100% score was investigated instead of accepted, which is
  exactly the instinct a paying client would want in whoever built this,
  and exactly the instinct most portfolio projects don't demonstrate.
- **What wasn't tested is stated as plainly as what was** — for example,
  many people using the system at once *while* it holds a very large
  document set has never been tested together, and that's written down
  as an open question rather than glossed over.

This is what "the system works" means here: not a good-looking chat
transcript, but a chain of tests specifically designed to catch the
system lying to its own test, followed by fixing what that turned up.

For the full numbers, methodology, and every case-by-case detail, see
`eval/README.md`.
