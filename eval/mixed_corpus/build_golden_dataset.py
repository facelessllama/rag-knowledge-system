#!/usr/bin/env python3
"""Builds eval/mixed_corpus/golden_dataset.json — golden Q&A cases for the
762-document cross-domain corpus, discovery split ONLY (see README.md's
"Evaluation rule": calibration/heldout stay untouched until the approach
built from this pass is frozen).

Same discipline as eval/build_golden_dataset.py and eval/build_eu_golden_
dataset.py: every fact is extracted programmatically from the source PDF's
actual text/metadata, never hand-transcribed, so the expected answer is
guaranteed correct by construction. Different domains need different
extraction patterns (arXiv metadata + figure/table captions; DailyMed's two
label formats; FAA brochures), so this is one script per domain instead of
one shared regex the way the more uniform EU/UK corpora could use.

Question types, deliberately varied rather than one style repeated:
- fact_author / fact_figure_caption (scientific): checkable substring facts.
  fact_figure_caption specifically biases toward captions on LATE pages of
  LONG documents (>=20 pages) — the opening-chunk identity boost in
  rag/retriever.py already makes page-1 content easy to retrieve almost
  regardless of ranking quality; a caption from deep in a 100-page paper
  actually exercises real retrieval.
- fact_active_ingredient / fact_rx_approval (medical): DailyMed ships two
  incompatible label formats (OTC "Drug Facts" panel vs. Rx "Highlights of
  Prescribing Information") — see the two regexes below, confirmed against
  real documents of each kind before writing this.
- fact_table_caption (manuals): best-effort only, this slice is small (16
  docs) and mostly plain prose brochures.
- topic_search: a query built from distinctive terms already present in the
  title, without using the title as a literal substring — recall-only
  (expected_substring left unset), same convention as the EU corpus's
  ID-free/semantic set.
- comparison: two same-category documents in one question — recall-only,
  scored via run_eval.py's new expected_doc_filenames (plural) support.
- hard_negative: expect_answer=False, a plausible-sounding query about a
  topic verified absent from every title/source_id in the full manifest
  (not just discovery — a false "hard negative" whose answer actually sits
  in calibration/heldout would be silently wrong here).

Usage:
    source venv/bin/activate
    python eval/mixed_corpus/build_golden_dataset.py
"""
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from ingestion.pdf_parser import PDFParser
from ingestion.chunker import normalize_whitespace, SmartChunker

CORPUS_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = CORPUS_DIR / "manifest.json"
REPORT_PATH = CORPUS_DIR / "baseline_ingest_report.json"
OUT_PATH = CORPUS_DIR / "golden_dataset.json"

SEED = 20260720
LONG_DOC_PAGE_THRESHOLD = 20

# ── Figure/Table caption extraction ──────────────────────────────────────────
# Rewritten after a manual audit (see VALIDATION.md-adjacent session notes)
# found the previous single-pattern regex (a) never matched the "Fig."/"Fig"
# abbreviation at all — only literal "Figure" — silently mis-attributing a
# caption to the nearest unrelated "Figure N." in-text citation instead, and
# (b) truncated at [:80] mid-word/mid-sentence, which made the ground truth
# itself nearly impossible to credit via substring matching AND degraded the
# semantic judge's ability to fairly grade a correct paraphrase (it was
# judging against a broken fragment). Manual spot-check of 8 "model answered
# wrong despite the right page being retrieved" cases found 6 were actually
# correct, torpedoed by exactly this truncation.
#
# Any given "Figure N"/"Table N" mention on a page is one of: the caption
# itself, most often colon-terminated ("Figure 3: Comparison...") but some
# ML-paper templates use a pipe instead ("Table 3 | Ablation results..." —
# confirmed on a real arXiv PDF; grouped into the same tier as colon, since
# it's just as unambiguous a caption-opening convention), a period-
# terminated caption some venues use ("Fig. 3. Comparison..."), a bare
# caption with no punctuation before the descriptive text at all ("Fig. 1
# Mechanistic World Models..." — also confirmed on a real arXiv PDF), or
# just an in-text reference ("as shown in Figure 3, ..." / "...see Figure
# 3." / "Fig. 3 (b) of the main paper..."). The tiers below are tried
# strongest-signal-first; a candidate is rejected outright (not just
# skipped) if what follows doesn't look like an actual caption at all — see
# _looks_like_prose/_looks_like_caption_opener — so a stronger-tier false
# positive doesn't win over a weaker-tier real one.
_CAPTION_LABEL_COLON_RE = re.compile(r'\b(Fig(?:ure)?|Table)\.?\s+(\d+)\s*[:|]\s*')
_CAPTION_LABEL_PERIOD_RE = re.compile(r'\b(Fig(?:ure)?|Table)\.?\s+(\d+)\s*\.\s+(?=[A-Z(])')
_CAPTION_LABEL_BARE_RE = re.compile(r'\b(Fig(?:ure)?|Table)\.?\s+(\d+)\s+(?=[A-Z])')
# NOT "(?=[A-Z(])" — an opening paren here also matches a sub-panel
# cross-reference like "As shown in Fig. 3 (b) of the main paper, at each
# reasoning step..." (confirmed on arxiv_2607.11523, a supplementary
# document referencing a figure in a DIFFERENT main paper — "(b)" isn't a
# caption opener, the sentence right after it is ordinary body prose).
# Genuine multi-panel captions ("Figure 3: (a) shows... (b) shows...")
# still match fine via the COLON/PERIOD tiers, which require actual
# caption-opening punctuation first — this bare (no-punctuation-at-all)
# tier is the one with the least evidence to begin with, so it's the one
# that can least afford the extra ambiguity "(" adds.
_ANY_LABEL_RE = re.compile(r'\b(Fig(?:ure)?|Table)\.?\s+(\d+)\b')

# A real caption describes what the figure/table shows — it never opens with
# a discourse connective continuing an unrelated paragraph. This is what
# catches the false-positive shape "...as discussed in Figure 1. Finally, we
# turn to related work." — the label's punctuation looks exactly like a
# caption opener, but "Finally," is mid-paragraph prose, not a description
# of Figure 1 (confirmed: this exact shape produced a wrong ground truth on
# arxiv_2607.12474 before this rewrite).
_DISCOURSE_CONNECTIVES = (
    "finally", "however", "moreover", "therefore", "thus", "overall",
    "furthermore", "nevertheless", "meanwhile", "additionally",
    "nonetheless", "consequently", "hence", "similarly", "in addition",
    "in contrast", "on the other hand", "as a result",
)

# 2, not 3: a figure caption immediately followed by resumed body prose
# (no distinct heading, common in single-column layouts) can have that
# following discussion ALSO look like coherent caption-adjacent prose to
# _looks_like_prose — a 3rd sentence risked absorbing a whole unrelated
# appendix section (confirmed: "...prohibitive computational overhead. L
# Algorithms For completeness, we provide pseudocode..." — "L Algorithms"
# is a real section heading, not part of Figure 5's caption, on
# arxiv_2607.10131). Note this doesn't fully solve the analogous table
# case where a header/data row's cell VALUES happen to be full English
# sentences (not just numbers) — that's a known, documented residual gap,
# not something _looks_like_prose's alpha-ratio heuristic can catch by
# content alone without real page-layout information.
_MAX_CAPTION_SENTENCES = 2
_MAX_CAPTION_CHARS = 500
_MIN_CAPTION_CHARS = 15

# Only used for _split_sentences_with_offsets() — chunk_size/overlap are
# irrelevant here (captions are always far under 2x any real chunk_size, so
# _hard_wrap_oversized_spans never engages).
_SENTENCE_SPLITTER = SmartChunker()


def _looks_like_prose(s: str) -> bool:
    """Rejects a sentence-splitter span that's actually table data ("Metric
    Closed (n=11) Open (n=6) ∆ p sig Act 81.2% 79.6%...") rather than
    caption prose — a stretch of mostly digits/symbols with few real
    letters. This is what stops caption extension right at the boundary
    between a table's descriptive caption and its own header/data rows,
    which routinely follow immediately in the same normalized text with no
    other delimiter to key off."""
    letters = sum(ch.isalpha() for ch in s)
    return len(s) >= 10 and letters / len(s) >= 0.5


def _looks_like_caption_opener(s: str) -> bool:
    lowered = s.lower()
    return not any(lowered.startswith(w) for w in _DISCOURSE_CONNECTIVES)


# ingestion/chunker.py's sentence splitter (reused below for real chunk-
# offset compatibility) treats any ".<space>" as a sentence end, with no
# abbreviation awareness — correct for chunking (an oversplit costs nothing
# there), but wrong for caption extraction: a caption title like "AI Alone
# vs. Human-AI Collaboration" gets falsely cut right after "vs." (confirmed
# on arxiv_2607.10317's Table 1). Extending word-by-word past a false split
# (see _extend_across_abbreviation) rather than teaching the shared
# splitter about abbreviations generally, which is out of scope here and
# could change real chunk boundaries in production.
_ABBREVIATIONS_NOT_SENTENCE_ENDS = {
    "vs", "fig", "figs", "eq", "eqs", "no", "approx", "cf", "etc",
    "al", "resp", "eqn", "sec", "eg", "ie",
}

def _looks_like_prose_fragment(s: str) -> bool:
    """Same alpha-ratio idea as _looks_like_prose, but with no minimum
    length floor — _extend_across_abbreviation's trailing window starts
    short by construction (one word at a time from the split point), and
    would otherwise fail on the first word or two regardless of content,
    since _looks_like_prose's len>=10 floor exists for a different
    purpose (rejecting a short numeric fragment outright, not judging a
    growing window)."""
    if not s.strip():
        return True
    letters = sum(ch.isalpha() for ch in s)
    return letters / len(s) >= 0.5


def _ends_with_abbreviation(normalized: str, pos: int) -> bool:
    m = re.search(r'([A-Za-z]+)\.\s*$', normalized[:pos])
    return bool(m and m.group(1).lower() in _ABBREVIATIONS_NOT_SENTENCE_ENDS)


def _extend_across_abbreviation(normalized: str, start: int, ceiling: int) -> int:
    """Called when the sentence splitter stopped right at a false
    abbreviation split — extends word-by-word from `start`, judging each
    NEW word on its own (not a trailing window: a window stays dominated
    by earlier real words for several tokens after table data actually
    starts, since e.g. "System Accuracy 97.19%" is still mostly letters —
    confirmed absorbing 3-4 numeric tokens too many with a 60-char window
    before this switched to per-word). Stops the instant a word doesn't
    look like prose, or a REAL sentence end (not on the abbreviation list)
    is hit — better to cut a word early than absorb table data."""
    pos = start
    while pos < ceiling:
        m = re.match(r'\s*(\S+)', normalized[pos:ceiling])
        if not m:
            break
        word = m.group(1)
        word_end = pos + m.end()
        if not _looks_like_prose_fragment(word):
            break
        pos = word_end
        if re.search(r'[.!?]$', word) and not _ends_with_abbreviation(normalized, word_end):
            return word_end  # a genuine sentence end — stop here, not the abbreviation-extension caller's job to go further
    return pos


def _word_boundary_truncate(text: str, limit: int) -> str:
    """Display-only preview — never used for scoring, so cutting mid-word is
    merely ugly here, not a ground-truth defect."""
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return cut + "…"


def _find_label_candidates(normalized: str, kind: str, num: str) -> list[tuple[int, int]]:
    """Every occurrence of `kind num` on the page as (tier, end_offset),
    strongest tier first. Multiple mentions of the same figure/table are
    routine (the caption itself, plus later in-text references) — trying
    every one is what lets a weaker-but-correct match win when a
    stronger-looking match turns out to be a stray citation (see
    extract_caption's use of _looks_like_caption_opener)."""
    out = []
    for tier, pat in enumerate((_CAPTION_LABEL_COLON_RE, _CAPTION_LABEL_PERIOD_RE, _CAPTION_LABEL_BARE_RE)):
        for m in pat.finditer(normalized):
            is_table = m.group(1).lower().startswith("table")
            if is_table != (kind == "Table"):
                continue
            if m.group(2) != num:
                continue
            out.append((tier, m.end()))
    out.sort(key=lambda c: c[0])
    return out


def _all_caption_opener_starts(normalized: str) -> list[int]:
    """Start offsets of every position on the page that itself looks like
    the OPENING of some caption (colon/period/bare-tier match for ANY
    figure/table number, not just the one being resolved) — used as a hard
    ceiling on a caption's own extension. Deliberately NOT every bare
    mention of "Figure N" (that's _ANY_LABEL_RE, used only to discover
    which numbers exist at all in find_all_captions): a caption's own prose
    routinely cross-references another figure by number ("Recalculation of
    Figure 7 under the zero-padding accounting..." — confirmed on
    arxiv_2607.16117's Figure 9 caption), and that in-line reference must
    NOT be treated as a new caption boundary — it only matches here if
    IT is itself followed by caption-opening punctuation, which a mid-
    sentence reference never is."""
    starts = []
    for pat in (_CAPTION_LABEL_COLON_RE, _CAPTION_LABEL_PERIOD_RE, _CAPTION_LABEL_BARE_RE):
        starts.extend(m.start() for m in pat.finditer(normalized))
    return sorted(starts)


def extract_caption(normalized_page_text: str, kind: str, num: str) -> dict | None:
    """Finds the best caption for `kind num` (kind is "Figure" or "Table")
    in a single page's already-normalized text (normalize_whitespace()
    applied — the same coordinate space ingestion/chunker.py's char_start/
    char_end use, so the offsets returned here line up with real chunks at
    query time, letting a caller directly compute chunk-range ∩
    caption-range). Returns None rather than guessing if no candidate looks
    like an actual caption — see ingestion/chunker.py's OTC_ACTIVE_RE
    handling for the same "drop rather than emit a wrong ground truth"
    discipline applied elsewhere in this file."""
    sentences = _SENTENCE_SPLITTER._split_sentences_with_offsets(normalized_page_text)
    # Another caption's OPENING (not just any mention of its number) is a
    # hard ceiling on our own caption's extension — without this, a caption
    # immediately followed by another labeled table's caption with no
    # intervening sentence-ending period (routine: "...two-sided.\nTable
    # 11: Per-category...") lets the sentence-boundary loop below absorb
    # the next caption whole, since _looks_like_prose alone can't reject a
    # combined span that's prose-heavy only because of what it swallowed.
    all_opener_starts = _all_caption_opener_starts(normalized_page_text)

    for _tier, end_offset in _find_label_candidates(normalized_page_text, kind, num):
        next_label_start = next((s for s in all_opener_starts if s > end_offset), len(normalized_page_text))
        relevant = [(s, e, t) for s, e, t in sentences if e > end_offset and s < next_label_start]
        if not relevant:
            continue
        piece_end = end_offset
        for s, e, text in relevant[:_MAX_CAPTION_SENTENCES]:
            e = min(e, next_label_start)
            segment = normalized_page_text[max(s, end_offset):e]
            if not _looks_like_prose(segment):
                break
            if e - end_offset > _MAX_CAPTION_CHARS:
                break
            piece_end = e
        # The sentence splitter has no abbreviation awareness — if it
        # stopped right after one ("...AI Alone vs."), the caption almost
        # certainly continues past it. Extend word-by-word with a short
        # LOCAL window instead of re-evaluating the whole remainder as one
        # blob — a real caption title has enough genuine words to keep a
        # whole-remainder ratio looking prose-y even after it's absorbed an
        # entire table's worth of numbers behind it (confirmed: did exactly
        # that the first time this checked the cumulative span instead).
        if piece_end > end_offset and _ends_with_abbreviation(normalized_page_text, piece_end):
            piece_end = _extend_across_abbreviation(
                normalized_page_text, piece_end,
                min(next_label_start, end_offset + _MAX_CAPTION_CHARS),
            )
        if piece_end <= end_offset:
            continue
        raw = normalized_page_text[end_offset:piece_end]
        expected_text = raw.strip()
        if len(expected_text) < _MIN_CAPTION_CHARS:
            continue
        if not _looks_like_caption_opener(expected_text):
            continue
        # .strip() above can move the true start/end in from end_offset/
        # piece_end (trailing whitespace is routine off a word-extension or
        # sentence-span boundary) — char_start/char_end must point at
        # exactly expected_text, not the pre-strip raw span, or a caller
        # slicing normalize_whitespace(page_text)[char_start:char_end]
        # gets a different string than expected_text itself.
        lead = len(raw) - len(raw.lstrip())
        char_start = end_offset + lead
        char_end = char_start + len(expected_text)
        return {"expected_text": expected_text, "char_start": char_start, "char_end": char_end}
    return None


def _verify_caption_offsets(page_text: str, cap: dict) -> None:
    """Re-derives normalize_whitespace(page_text)[char_start:char_end] and
    asserts it equals expected_text — a caption case is worthless as ground
    truth if its own offsets don't actually point at its own text on its
    own claimed page, so this fails loudly (build-time, not silently
    emitted) rather than ever shipping a case that couldn't be trusted."""
    normalized = normalize_whitespace(page_text)
    sliced = normalized[cap["char_start"]:cap["char_end"]]
    assert sliced == cap["expected_text"], (
        f"caption offset verification failed: slice [{cap['char_start']}:{cap['char_end']}] "
        f"gave {sliced!r}, expected {cap['expected_text']!r}"
    )


def find_all_captions(page_text: str) -> list[dict]:
    """Every distinct Figure/Table caption resolvable on one page — each
    dict has kind/num/expected_text/char_start/char_end. Discovers WHICH
    (kind, num) pairs are mentioned at all via the loose _ANY_LABEL_RE
    (deliberately permissive — this pass is just for finding candidates to
    resolve, not for accepting one), then resolves each pair's actual
    caption via extract_caption(), which does the real strongest-signal-
    first, reject-if-implausible work."""
    normalized = normalize_whitespace(page_text)
    seen_pairs = set()
    results = []
    for m in _ANY_LABEL_RE.finditer(normalized):
        kind = "Table" if m.group(1).lower().startswith("table") else "Figure"
        num = m.group(2)
        if (kind, num) in seen_pairs:
            continue
        seen_pairs.add((kind, num))
        cap = extract_caption(normalized, kind, num)
        if cap:
            results.append({"kind": kind, "num": num, **cap})
    return results


OTC_ACTIVE_RE = re.compile(r'Active ingredient[s]?\s*\n([^\n]{3,150})')
RX_APPROVAL_RE = re.compile(r'Initial U\.S\. Approval:\s*(\d{4})')
RX_INDICATION_RE = re.compile(
    r'INDICATIONS AND USAGE\s*\n([A-Z][\w\-]+(?:\s+is|\s+are)\s+[^.]{15,180}\.)'
)
# Some (mostly homeopathic multi-ingredient) OTC labels lay the panel out as
# a header ROW ("Active ingredient  Purpose") followed by a value row
# ("Arnica Montana 30X HPUS  bruising, swelling...") instead of the simple
# "Active ingredient\n<value>" pairing OTC_ACTIVE_RE assumes — a naive match
# there grabs the literal word "Purpose" as the "active ingredient" (caught
# by hand on dailymed_85fcf1c8...ARNICA...pdf). Rejecting any capture that
# IS itself a known panel header, rather than trying to also parse the
# row-pair layout, keeps every emitted fact correct by construction at the
# cost of silently dropping the harder-to-parse label — better than a
# confidently wrong ground truth.
_OTC_HEADER_WORDS = {"purpose", "uses", "use", "warnings", "directions",
                      "other information", "inactive ingredients", "questions"}


def load_eligible_discovery_docs() -> list[dict]:
    """Discovery-split entries that actually made it into the live index
    (status "indexed" or "skipped" — "skipped" means already-indexed under
    a different doc_id from the interrupted first ingest run, still fully
    queryable, see VALIDATION.md) and aren't flagged as OCR noise on the
    hard/* slices (see extraction_verdict.md — the same flag is a known
    false positive on `medical`'s terse-but-legible labels, so it's only
    applied to hard_* categories here)."""
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))["detail"]

    eligible = []
    for entry in manifest:
        if entry["split"] != "discovery":
            continue
        r = report.get(entry["relative_path"])
        if not r or r["status"] not in ("indexed", "skipped"):
            continue
        if entry["category"].startswith("hard_") and entry.get("quality_flag") == "likely_garbage":
            continue
        eligible.append(entry)
    return eligible


def parse(entry: dict, parser: PDFParser):
    return parser.parse(str(CORPUS_DIR / entry["relative_path"]))


def build_scientific_cases(docs: list[dict], parser: PDFParser, rng: random.Random) -> list[dict]:
    cases = []
    long_docs = [d for d in docs if d.get("pages", 0) >= LONG_DOC_PAGE_THRESHOLD]
    short_docs = [d for d in docs if d.get("pages", 0) < LONG_DOC_PAGE_THRESHOLD]
    # Bias toward long documents per this pass's explicit brief ("especially
    # long files"), while keeping some short ones for domain balance.
    sample = rng.sample(long_docs, min(140, len(long_docs))) + rng.sample(short_docs, min(40, len(short_docs)))

    for entry in sample:
        parsed = parse(entry, parser)
        title = entry["title"]
        fname = Path(entry["relative_path"]).name
        full_text = "\n".join(p["text"] for p in parsed.pages)

        author = parsed.metadata.get("author", "")
        if author and author != "Unknown":
            first_author = author.split(";")[0].strip()
            # The PDF info-dict author field and the name as it actually
            # appears on the rendered page routinely disagree in ways a
            # blind trust in metadata would miss: "Lastname, Firstname" vs.
            # "Firstname Lastname" order, diacritics normalized differently,
            # or — caught by hand on 2607.15968 — a publisher string
            # ("Springer-SBM") sitting in the author field instead of a
            # person at all. Requiring the name to actually appear verbatim
            # in the page text this fact will be checked against is a cheap,
            # build-time version of the same "guaranteed correct by
            # construction" discipline as every regex-based fact below —
            # dropped 8/171 candidates the first time this ran without it.
            if first_author and len(first_author) < 60 and first_author in full_text:
                cases.append({
                    "id": f"sci_author_{entry['source_id']}",
                    "question": f'Who is a listed author of the paper "{title}"?',
                    "expect_answer": True,
                    "expected_doc_filename": fname,
                    "expected_substring": first_author,
                    "type": "fact_author",
                })

        captions = []
        total_pages = parsed.total_pages or 1
        for p in parsed.pages:
            for cap in find_all_captions(p["text"]):
                captions.append((p["page_num"], cap, p["text"]))
        if captions:
            # Prefer a caption from beyond 40% into the document when the
            # doc is long enough for that distinction to mean anything —
            # this is the "deep retrieval, not just page 1" test.
            deep = [c for c in captions if c[0] > total_pages * 0.4]
            page_num, cap, page_text = rng.choice(deep or captions)
            _verify_caption_offsets(page_text, cap)
            cases.append({
                "id": f"sci_caption_{entry['source_id']}_{cap['kind']}{cap['num']}",
                "question": f'What does {cap["kind"]} {cap["num"]} show in the paper "{title}"?',
                "expect_answer": True,
                "expected_doc_filename": fname,
                "expected_substring": cap["expected_text"],
                "caption_label": f'{cap["kind"]} {cap["num"]}',
                "display_preview": _word_boundary_truncate(cap["expected_text"], 80),
                "evidence_char_start": cap["char_start"],
                "evidence_char_end": cap["char_end"],
                "type": "fact_figure_caption",
                "source_page": page_num,
                "doc_pages": total_pages,
            })

    # Topic-search: a separate, smaller, non-overlapping sample — recall-only,
    # query built from title content words, never the literal title string.
    remaining = [d for d in docs if d not in sample]
    topic_sample = rng.sample(remaining, min(50, len(remaining)))
    for entry in topic_sample:
        words = [w for w in re.findall(r"[A-Za-z][A-Za-z\-]{3,}", entry["title"]) if w.lower() not in
                 {"with", "from", "into", "using", "under", "over", "that", "this", "their", "towards", "toward"}]
        if len(words) < 3:
            continue
        topic_terms = " ".join(rng.sample(words, min(4, len(words))))
        cases.append({
            "id": f"sci_topic_{entry['source_id']}",
            "question": f"Which paper in the corpus discusses {topic_terms}?",
            "expect_answer": True,
            "expected_doc_filename": Path(entry["relative_path"]).name,
            "expected_substring": None,
            "type": "topic_search",
        })
    return cases


def build_medical_cases(docs: list[dict], parser: PDFParser) -> list[dict]:
    cases = []
    for entry in docs:
        parsed = parse(entry, parser)
        text = "\n".join(p["text"] for p in parsed.pages)
        title = entry["title"]
        fname = Path(entry["relative_path"]).name

        m = OTC_ACTIVE_RE.search(text)
        if m and m.group(1).strip().lower() not in _OTC_HEADER_WORDS:
            cases.append({
                "id": f"med_active_{entry['source_id']}",
                "question": f'What is the active ingredient in "{title}"?',
                "expect_answer": True,
                "expected_doc_filename": fname,
                "expected_substring": m.group(1).strip(),
                "type": "fact_active_ingredient",
            })
            continue  # OTC label matched — don't also try the Rx pattern
        elif m:
            continue  # header-row layout, not the simple pairing this regex assumes — drop rather than guess

        m = RX_APPROVAL_RE.search(text)
        if m:
            cases.append({
                "id": f"med_approval_{entry['source_id']}",
                "question": f'What year was "{title}" initially approved by the FDA?',
                "expect_answer": True,
                "expected_doc_filename": fname,
                "expected_substring": m.group(1),
                "type": "fact_rx_approval",
            })
        m2 = RX_INDICATION_RE.search(text)
        if m2:
            cases.append({
                "id": f"med_indication_{entry['source_id']}",
                "question": f'What is "{title}" indicated for?',
                "expect_answer": True,
                "expected_doc_filename": fname,
                "expected_substring": m2.group(1).strip()[:100],
                "type": "fact_rx_indication",
            })
    return cases


def build_manuals_cases(docs: list[dict], parser: PDFParser) -> list[dict]:
    cases = []
    for entry in docs:
        parsed = parse(entry, parser)
        title = entry["title"]
        fname = Path(entry["relative_path"]).name
        for p in parsed.pages:
            caps = find_all_captions(p["text"])
            if caps:
                cap = caps[0]
                _verify_caption_offsets(p["text"], cap)
                cases.append({
                    "id": f"man_caption_{entry['source_id']}",
                    "question": f'What does {cap["kind"]} {cap["num"]} show in "{title}"?',
                    "expect_answer": True,
                    "expected_doc_filename": fname,
                    "expected_substring": cap["expected_text"],
                    "caption_label": f'{cap["kind"]} {cap["num"]}',
                    "display_preview": _word_boundary_truncate(cap["expected_text"], 80),
                    "evidence_char_start": cap["char_start"],
                    "evidence_char_end": cap["char_end"],
                    "type": "fact_table_caption",
                    "source_page": p["page_num"],
                })
                break
    return cases


def build_comparison_cases(docs_by_category: dict[str, list[dict]], rng: random.Random) -> list[dict]:
    """Two variants of the SAME pairs, deliberately not one: "unscoped" asks
    a plain natural-language question naming both documents — the way any
    free-text query would — and leans entirely on retrieval/reranking to
    find both from the question text alone. "scoped" sets "scoped": True,
    which eval/run_eval.py resolves into the request's document_ids field —
    the same mechanism the frontend's actual "Compare Documents" button
    uses (api/schemas.py's QueryRequest.document_ids), which fires a
    different code path (_augment_compare_queries's one guaranteed
    sub-query per named document) that an unscoped question never reaches
    at all. Reporting only the unscoped number and calling it "comparison
    accuracy" would silently never test the feature the UI actually ships."""
    cases = []
    for cat, docs in docs_by_category.items():
        if len(docs) < 2:
            continue
        pairs = list(zip(docs[::2], docs[1::2]))
        for a, b in rng.sample(pairs, min(8, len(pairs))):
            fa, fb = Path(a["relative_path"]).name, Path(b["relative_path"]).name
            base = {
                "expect_answer": True,
                "expected_doc_filenames": [fa, fb],
                "expected_substring": None,
            }
            cases.append({**base,
                "id": f"cmp_unscoped_{cat}_{a['source_id']}_{b['source_id']}",
                "question": f'How do "{a["title"]}" and "{b["title"]}" differ?',
                "type": "comparison_unscoped",
            })
            cases.append({**base,
                "id": f"cmp_scoped_{cat}_{a['source_id']}_{b['source_id']}",
                "question": f'How do these two documents differ?',
                "type": "comparison_scoped",
                "scoped": True,
            })
    return cases


# Verified absent from every title/source_id in the FULL manifest (not just
# discovery) at build time — see main(); a topic that happens to be the
# subject of a calibration/heldout document would make this a false "hard
# negative" invisible to anyone eyeballing the discovery split alone.
HARD_NEGATIVE_CANDIDATES = [
    "What is the recommended pediatric dosage of amoxicillin for otitis media?",
    "What does the FAA require for night vision goggle certification in helicopters?",
    "What is the published benchmark accuracy of GPT-2 on the GLUE dataset?",
    "What are the labeled side effects of ibuprofen in the corpus?",
    "What does the corpus say about supersonic aircraft noise regulations?",
    "What is the recommended treatment for anaphylaxis in the FAA manuals?",
    "What vaccine dosage schedule is described in the medical documents?",
    "What does the corpus say about lithium battery fire suppression on aircraft?",
    "What is the melting point of titanium described in any paper?",
    "What does the corpus say about parachute packing regulations?",
]


def build_hard_negative_cases(full_manifest: list[dict]) -> list[dict]:
    haystack = " ".join(e["title"].lower() for e in full_manifest)
    cases = []
    for i, q in enumerate(HARD_NEGATIVE_CANDIDATES):
        probe_terms = [w for w in re.findall(r"[a-z]{5,}", q.lower())]
        if any(t in haystack for t in probe_terms if t not in
               {"about", "which", "there", "would", "recommended", "described", "regulations", "corpus"}):
            continue  # a probe term coincidentally appears in some title — skip, don't risk a false negative
        cases.append({
            "id": f"hardneg_{i}",
            "question": q,
            "expect_answer": False,
            "expected_doc_filename": None,
            "expected_substring": None,
            "type": "hard_negative",
        })
    return cases


def main():
    rng = random.Random(SEED)
    full_manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    eligible = load_eligible_discovery_docs()
    print(f"{len(eligible)} eligible discovery-split documents (of {sum(1 for e in full_manifest if e['split']=='discovery')} total discovery)")

    by_cat: dict[str, list[dict]] = {}
    for e in eligible:
        by_cat.setdefault(e["category"], []).append(e)

    parser = PDFParser()  # reads OCR/page-limit env config the same as ingestion — see probe_extraction.py

    cases = []
    cases += build_scientific_cases(by_cat.get("scientific", []), parser, rng)
    cases += build_medical_cases(by_cat.get("medical", []), parser)
    cases += build_manuals_cases(by_cat.get("manuals", []), parser)
    cases += build_comparison_cases(
        {k: v for k, v in by_cat.items() if k in ("scientific", "medical", "manuals")}, rng
    )
    cases += build_hard_negative_cases(full_manifest)

    from collections import Counter
    print(f"\nBuilt {len(cases)} cases:")
    for t, n in sorted(Counter(c["type"] for c in cases).items()):
        print(f"  {t:24s} {n}")

    OUT_PATH.write_text(json.dumps(cases, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
