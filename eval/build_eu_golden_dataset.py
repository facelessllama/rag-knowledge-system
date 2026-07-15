"""
Builds eval/eu_golden_dataset.json + eval/eu_heldout_dataset.json +
eval/eu_cross_question_dataset.json — the golden test for the EU
legislation scale test (see eval/README.md, "Scale test: EU legislation
corpus"). Same discipline as build_golden_dataset.py: facts are extracted
programmatically from the source PDFs, not hand-transcribed, so expected
answers are correct by construction.

Reads directly from eval/eu_manifest.json (see scripts/bulk_ingest_pdfs.py)
— the first GOLDEN_N manifest entries become the golden set, the next
HELDOUT_N become heldout, and CROSS_QUESTION_N after that become the
cross-question-per-document sample. All three slices are within the first
few hundred manifest entries, which scripts/bulk_ingest_pdfs.py's pilot
ingest (--up-to 400) already uploaded — and, being a prefix of the fixed
manifest order, stay part of the corpus at every later scale-ladder
checkpoint (--up-to 5000, 15000, ...) without re-ingestion.

Filenames are "<CELEX_ID> - <description>.pdf" (EURLEX57K dataset) — the
description is sometimes truncated (long titles cut mid-word), so facts
are extracted from the PDF's own first-page text, not the filename, using
fitz (PyMuPDF) directly (this is a build-time script over local files, not
part of the ingestion pipeline — mirrors how build_golden_dataset.py reads
test_corpus/*.txt directly rather than going through TxtParser).

Usage:
    python eval/build_eu_golden_dataset.py
"""
import json
import re
from pathlib import Path

import fitz

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = Path(__file__).resolve().parent
SOURCE_DIR = Path("/mnt/c/Users/serg/Downloads/PDFS LEGAL RAG TEST/pdfs")
MANIFEST_PATH = EVAL_DIR / "eu_manifest.json"

GOLDEN_N = 150
HELDOUT_N = 150
CROSS_QUESTION_N = 30  # documents, x4 question types each

# "<CELEX_ID> - <description>.pdf" — description may be truncated, celex_id never is.
_FILENAME_RE = re.compile(r"^(\S+)\s*-\s*(.+)$")

# Document type from the bold header line, e.g. "COMMISSION REGULATION (EC) No 955/97"
# or "COUNCIL DECISION of 21 February 2011" or "2011/126/EU: Council Decision of...".
_TYPE_RE = re.compile(r"\b(REGULATION|DECISION|DIRECTIVE|RECOMMENDATION)\b", re.IGNORECASE)

# The document's own adoption date always appears as "of <date>" within the
# first few hundred characters (title line + repeated in the type/date/
# subject header block) — take the first match.
_DATE_RE = re.compile(r"\bof\s+(\d{1,2}\s+[A-Z][a-z]+\s+\d{4})\b")

# Cross-references: the phrase immediately BEFORE a "<Type> (EEC|EC|EU) No
# N/YY" citation tells us the *relationship*, not just the number — asking
# "what does X reference?" without that context made the model consistently
# answer with the document's own number instead (confirmed directly on 19
# pilot failures — see eval/README.md, "Scale test" section). Two buckets:
# action verbs, where "is X <verb>ing?" reads naturally in a question, and
# citation phrases (the standard EU legal-basis formula "Having regard to
# ..."), where a generic-but-explicit "excluding its own number" phrasing
# is used instead. Both exclude the document's own restated number.
_XREF_VERB_RE = re.compile(
    r"\b(amending|repealing|implementing|replacing|correcting|supplementing)\s+"
    r"(?:for the \w+ time\s+)?(?:Council |Commission )?"
    r"(?:Regulation|Directive|Decision)\s*\((?:EEC|EC|EU)\)\s*No\s*(\d+/\d+)",
    re.IGNORECASE,
)
_XREF_CITATION_RE = re.compile(
    r"\b(?:having regard to|provided for in|referred to in|pursuant to|"
    r"issued in|as last amended by)\s+(?:Council |Commission )?"
    r"(?:Regulation|Directive|Decision)\s*\((?:EEC|EC|EU)\)\s*No\s*(\d+/\d+)",
    re.IGNORECASE,
)
_REG_NUMBER_RE = re.compile(
    r"\b(?:Regulation|Directive|Decision)\s*\((?:EEC|EC|EU)\)\s*No\s*(\d+/\d+)", re.IGNORECASE
)


def extract_facts(pdf_path: Path) -> dict:
    doc = fitz.open(str(pdf_path))
    text = doc[0].get_text() if len(doc) else ""
    doc.close()
    head = text[:600]
    body = text[:1200]

    type_match = _TYPE_RE.search(head)
    date_match = _DATE_RE.search(head)

    own_numbers = set(_REG_NUMBER_RE.findall(head))  # own number always restated in the header

    # (auxiliary, verb-phrase) so the question reads as "What ... <aux>
    # <celex_id> <verb>?" — "is X amending" / "does X cite" both need
    # different auxiliaries to stay grammatical. The question can only ask
    # about one relationship, so it's phrased around the first verb/citation
    # match — but EU legal-basis preambles routinely cite *several* earlier
    # regulations ("Having regard to ... No 2777/75 ..., Having regard to
    # ... No 2967/85 ..."), and a model answering with any one of the
    # document's genuine cross-references is correct even if it's not the
    # first one found. xref_numbers below collects every distinct number
    # the body actually supports (same two patterns, matched across the
    # whole body, not just the first hit) so the ground truth doesn't
    # penalize a correct-but-different answer.
    xref_aux, xref_verb, xref_number = None, None, None
    verb_match = _XREF_VERB_RE.search(body)
    if verb_match and verb_match.group(2) not in own_numbers:
        xref_aux, xref_verb = "is", verb_match.group(1)  # already -ing: "amending", "repealing", ...
        xref_number = verb_match.group(2)
    else:
        citation_match = _XREF_CITATION_RE.search(body)
        if citation_match and citation_match.group(1) not in own_numbers:
            xref_aux, xref_verb = "does", "cite (excluding its own number)"
            xref_number = citation_match.group(1)

    xref_numbers = []
    if xref_number:
        for n in (*_XREF_VERB_RE.findall(body), *_XREF_CITATION_RE.findall(body)):
            n = n[-1] if isinstance(n, tuple) else n  # verb regex group is (verb, number)
            if n not in own_numbers and n not in xref_numbers:
                xref_numbers.append(n)

    return {
        "type": type_match.group(1).title() if type_match else None,
        "date": date_match.group(1) if date_match else None,
        "xref_aux": xref_aux,
        "xref_verb": xref_verb,
        "xref_number": xref_number,
        "xref_numbers": xref_numbers,
    }


def _cases_for_doc(filename: str, celex_id: str, facts: dict, id_prefix: str) -> list[dict]:
    """All question types this document's extracted facts support — order
    matters, callers pick a subset."""
    base_id = f"{id_prefix}_{celex_id}"
    out = [{
        "id": f"{base_id}_summary",
        "question": f"What is {celex_id} about?",
        "expect_answer": True,
        "expected_doc_filename": filename,
        "expected_substring": None,  # free-form — Recall@K only, like case_summary
        "type": "celex_summary",
    }]
    if facts["type"]:
        out.append({
            "id": f"{base_id}_lookup",
            "question": f"What does {facts['type']} {celex_id} concern?",
            "expect_answer": True,
            "expected_doc_filename": filename,
            "expected_substring": None,
            "type": "celex_lookup",
        })
    if facts["date"]:
        out.append({
            "id": f"{base_id}_date",
            "question": f"On what date was {celex_id} adopted?",
            "expect_answer": True,
            "expected_doc_filename": filename,
            "expected_substring": facts["date"],
            "type": "date_fact",
        })
    if facts["xref_number"]:
        out.append({
            "id": f"{base_id}_xref",
            "question": f"What earlier regulation, directive, or decision {facts['xref_aux']} "
                        f"{celex_id} {facts['xref_verb']}?",
            "expect_answer": True,
            "expected_doc_filename": filename,
            # list of every genuine cross-reference the body supports, not
            # just the one the question was phrased around — see
            # extract_facts()'s xref_numbers comment above.
            "expected_substring": facts["xref_numbers"],
            "type": "cross_reference_fact",
        })
    return out


def build_cases(manifest_slice: list[str], id_prefix: str, cross_question: bool = False) -> list[dict]:
    """cross_question=False (golden/heldout): one question type per
    document, rotating through whichever types that document's facts
    support — mirrors build_golden_dataset.py's case_summary/
    case_lookup_by_number alternation, avoiding 3-4 correlated cases
    piling up on the same document. cross_question=True: emit every
    supported type for each document (this is the point of that dataset —
    testing whether all facts about the SAME doc are independently
    findable)."""
    cases = []
    for i, filename in enumerate(manifest_slice):
        m = _FILENAME_RE.match(Path(filename).stem)
        if not m:
            continue
        celex_id, _description = m.groups()
        facts = extract_facts(SOURCE_DIR / filename)
        doc_cases = _cases_for_doc(filename, celex_id, facts, id_prefix)

        if cross_question:
            cases.extend(doc_cases)
        else:
            cases.append(doc_cases[i % len(doc_cases)])
    return cases


ADVERSARIAL_EU = [
    {"id": "eu_adv_capital_brazil", "question": "What is the capital of Brazil?", "expect_answer": False, "type": "out_of_domain"},
    {"id": "eu_adv_stock_price", "question": "What is the current share price of Apple Inc.?", "expect_answer": False, "type": "out_of_domain"},
    {"id": "eu_adv_recipe", "question": "How do I make lasagna?", "expect_answer": False, "type": "out_of_domain"},
    {"id": "eu_adv_weather", "question": "Will it rain in Berlin tomorrow?", "expect_answer": False, "type": "out_of_domain"},
    {"id": "eu_adv_meaning_of_life", "question": "What is the meaning of life according to these documents?", "expect_answer": False, "type": "out_of_domain"},
    {"id": "eu_adv_us_law", "question": "What does the US Constitution say about free speech?", "expect_answer": False, "type": "adjacent_wrong_jurisdiction"},
    {"id": "eu_adv_uk_law", "question": "What does UK company law require for annual filings?", "expect_answer": False, "type": "adjacent_wrong_jurisdiction"},
    {"id": "eu_adv_gdpr_specific", "question": "What are the penalties under the California Consumer Privacy Act?", "expect_answer": False, "type": "adjacent_wrong_topic"},
    {"id": "eu_adv_nonsense", "question": "asdkfj qpwoe zzz what banana regulation seventeen", "expect_answer": False, "type": "nonsense"},
    {"id": "eu_adv_fake_celex", "question": "What does Regulation 99999999 about intergalactic trade say?", "expect_answer": False, "type": "plausible_but_fake"},
]

HELDOUT_ADVERSARIAL_EU = [
    {"id": "ho_eu_adv_capital_egypt", "question": "What is the capital of Egypt?", "expect_answer": False, "type": "out_of_domain"},
    {"id": "ho_eu_adv_crypto", "question": "What is the current price of Bitcoin?", "expect_answer": False, "type": "out_of_domain"},
    {"id": "ho_eu_adv_recipe2", "question": "What ingredients go into a Greek salad?", "expect_answer": False, "type": "out_of_domain"},
    {"id": "ho_eu_adv_weather2", "question": "Is it snowing in Oslo right now?", "expect_answer": False, "type": "out_of_domain"},
    {"id": "ho_eu_adv_philosophy", "question": "What did Kant say about the categorical imperative?", "expect_answer": False, "type": "out_of_domain"},
    {"id": "ho_eu_adv_canada_law", "question": "What does the Canadian Charter of Rights guarantee?", "expect_answer": False, "type": "adjacent_wrong_jurisdiction"},
    {"id": "ho_eu_adv_india_law", "question": "What does the Indian Penal Code say about bail?", "expect_answer": False, "type": "adjacent_wrong_jurisdiction"},
    {"id": "ho_eu_adv_hipaa", "question": "What are the requirements for HIPAA compliance?", "expect_answer": False, "type": "adjacent_wrong_topic"},
    {"id": "ho_eu_adv_nonsense2", "question": "purple triangle seventeen quantum banana treaty", "expect_answer": False, "type": "nonsense"},
    {"id": "ho_eu_adv_fake_celex2", "question": "What does Directive 88888/ZZ about lunar mining say?", "expect_answer": False, "type": "plausible_but_fake"},
]


def main():
    manifest = json.loads(MANIFEST_PATH.read_text())

    golden_docs = manifest[:GOLDEN_N]
    heldout_docs = manifest[GOLDEN_N:GOLDEN_N + HELDOUT_N]
    cross_q_docs = manifest[GOLDEN_N + HELDOUT_N:GOLDEN_N + HELDOUT_N + CROSS_QUESTION_N]

    golden = build_cases(golden_docs, "eu") + ADVERSARIAL_EU
    heldout = build_cases(heldout_docs, "ho_eu") + HELDOUT_ADVERSARIAL_EU
    cross_q = build_cases(cross_q_docs, "xq_eu", cross_question=True)

    (EVAL_DIR / "eu_golden_dataset.json").write_text(json.dumps(golden, indent=2, ensure_ascii=False))
    (EVAL_DIR / "eu_heldout_dataset.json").write_text(json.dumps(heldout, indent=2, ensure_ascii=False))
    (EVAL_DIR / "eu_cross_question_dataset.json").write_text(json.dumps(cross_q, indent=2, ensure_ascii=False))

    print(f"golden: {len(golden)} cases ({len(golden_docs)} docs + {len(ADVERSARIAL_EU)} adversarial)")
    print(f"heldout: {len(heldout)} cases ({len(heldout_docs)} docs + {len(HELDOUT_ADVERSARIAL_EU)} adversarial)")
    print(f"cross_question: {len(cross_q)} cases ({len(cross_q_docs)} docs)")


if __name__ == "__main__":
    main()
