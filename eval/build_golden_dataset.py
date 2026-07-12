"""
Builds eval/golden_dataset.json — a golden Q&A set for regression-testing
retrieval and answer quality (see eval/README.md).

Facts are extracted programmatically from the source documents (not
hand-transcribed) so the expected answers are guaranteed correct by
construction — the "come into force on <date>" clause is present in the
large majority of these UK statutory instruments and gives a clean,
unambiguous fact per document.
"""
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TXT_DIR = REPO_ROOT / "test_corpus" / "text_legal_docs"
PDF_DIR = REPO_ROOT / "test_corpus" / "pdf_court_cases"
OUT_PATH = Path(__file__).resolve().parent / "golden_dataset.json"

FORCE_DATE_RE = re.compile(r"come into force on ([^.]{5,45}?)\.")
# UK statutory instruments in this corpus use a space as the thousands
# separator ("£75 000"), not a comma — match either.
MONEY_RE = re.compile(r"(?:is|of|sum of|amount of) £(\d+(?:[,\s]\d{3})*)(?:\s|\.)")
# Broader match (no preceding-phrase requirement) used only to count how many
# *distinct* amounts a document mentions, to detect ambiguous fee-schedule
# documents — see build_answerable_from_txt.
ALL_MONEY_RE = re.compile(r"£(\d+(?:[,\s]\d{3})*)")


def title_from_filename(stem: str) -> str:
    m = re.match(r"EN_\d+_(.+)", stem)
    title = (m.group(1) if m else stem).replace("_", " ")
    if title.lower().startswith("the "):
        title = title[4:]
    return title


def build_answerable_from_txt(txt_dir=TXT_DIR):
    cases = []
    for f in sorted(txt_dir.glob("EN_*.txt")):
        text = f.read_text(encoding="utf-8")
        title = title_from_filename(f.stem)

        m = FORCE_DATE_RE.search(text)
        if m:
            date = m.group(1).strip()
            cases.append({
                "id": f"date_{f.stem}",
                "question": f"When does the {title} come into force?",
                "expect_answer": True,
                "expected_doc_filename": f.name,
                "expected_substring": date,
                "type": "fact_date",
            })

        # Only ask a money question when the document mentions exactly one
        # distinct amount — fee schedules and allowance tables with many
        # different figures make "what monetary amount is prescribed"
        # genuinely ambiguous (there's no single correct answer), which
        # showed up as false "failures" where the model's refusal to pick
        # one number was actually the more honest response.
        distinct_amounts = set(ALL_MONEY_RE.findall(text))
        m2 = MONEY_RE.search(text)
        if m2 and len(distinct_amounts) == 1:
            amount = m2.group(1)
            cases.append({
                "id": f"money_{f.stem}",
                "question": f"What monetary amount is prescribed in the {title}?",
                "expect_answer": True,
                "expected_doc_filename": f.name,
                "expected_substring": amount,
                "type": "fact_money",
            })
    return cases


# Filenames are "<TYPE>_<NUM>_<YEAR>_<parties>.pdf" (see
# eval prep scripts) — e.g. "ABLAPL_3648_2020_LIPU_PRADHAN_vs_STATE_OF_ODISHA.pdf".
# Type prefix itself may contain underscores (e.g. "CR._MISC."), so match it
# non-greedily up to the first "_<digits>_<4-digit year>_" run instead of
# constraining its character set.
_PDF_NAME_RE = re.compile(r"^(.+?)_(\d+)_(\d{4})_(.+?)_vs_(.+)$")


def build_answerable_from_pdf(pdf_dir=PDF_DIR, limit=40):
    """Generates two question styles per document, split evenly:
    - case_summary: party names only ("What is the case X vs Y about?") —
      how you'd search if you don't remember the case number.
    - case_lookup_by_number: includes the case number ("What is case CRM
      10691 of 2020 ... about?") — how you'd search from a docket/cause
      list, where the number is what you actually have on hand.
    Both are realistic; testing only the party-only style (as this used to)
    silently never exercises case-number-based retrieval at all.
    """
    cases = []
    files = sorted(pdf_dir.glob("*.pdf"))[:limit]
    for i, f in enumerate(files):
        m = _PDF_NAME_RE.match(f.stem)
        if not m:
            continue
        case_type, num, year, p1, p2 = m.groups()
        party = f"{p1.replace('_', ' ')} vs {p2.replace('_', ' ')}"

        if i % 2 == 0:
            question = f"What is the case {party} about?"
            case_type_suffix = "case_summary"
        else:
            question = f"What is case {case_type} {num} of {year} ({party}) about?"
            case_type_suffix = "case_lookup_by_number"

        cases.append({
            "id": f"{case_type_suffix}_{f.stem[:40]}",
            "question": question,
            "expect_answer": True,
            "expected_doc_filename": f.name,
            "expected_substring": None,  # court rulings are free-form — Recall@K only, no substring check
            "type": case_type_suffix,
        })
    return cases


ADVERSARIAL = [
    {"id": "adv_capital_france", "question": "What is the capital of France?", "expect_answer": False, "type": "out_of_domain"},
    {"id": "adv_fed_rate", "question": "What interest rate does the Federal Reserve currently set?", "expect_answer": False, "type": "out_of_domain"},
    {"id": "adv_recipe", "question": "How do I bake a chocolate cake?", "expect_answer": False, "type": "out_of_domain"},
    {"id": "adv_weather", "question": "What is the weather forecast for tomorrow in London?", "expect_answer": False, "type": "out_of_domain"},
    {"id": "adv_meaning_of_life", "question": "What is the meaning of life according to these documents?", "expect_answer": False, "type": "out_of_domain"},
    {"id": "adv_us_law", "question": "What does the United States Constitution say about free speech?", "expect_answer": False, "type": "adjacent_wrong_jurisdiction"},
    {"id": "adv_gdpr", "question": "What are the penalties under the EU GDPR for data breaches?", "expect_answer": False, "type": "adjacent_wrong_topic"},
    {"id": "adv_crypto", "question": "What is the current price of Bitcoin?", "expect_answer": False, "type": "out_of_domain"},
    {"id": "adv_nonsense", "question": "Purple elephants dance quietly beneath the number seven.", "expect_answer": False, "type": "nonsense"},
    {"id": "adv_math", "question": "What is the square root of 144?", "expect_answer": False, "type": "out_of_domain"},
    {"id": "adv_medical", "question": "What is the recommended dosage of ibuprofen for adults?", "expect_answer": False, "type": "out_of_domain"},
    {"id": "adv_fake_regulation", "question": "What does the Martian Colonization Amendment Regulations 2099 specify?", "expect_answer": False, "type": "plausible_but_fake"},
    {"id": "adv_fake_case", "question": "What was the ruling in Smith vs Jones regarding intellectual property in California?", "expect_answer": False, "type": "plausible_but_fake"},
    {"id": "adv_sports", "question": "Who won the World Cup in 2022?", "expect_answer": False, "type": "out_of_domain"},
    {"id": "adv_programming", "question": "How do I write a for loop in Python?", "expect_answer": False, "type": "out_of_domain"},
]


def main():
    dataset = build_answerable_from_txt() + build_answerable_from_pdf() + ADVERSARIAL
    n_answerable = sum(1 for c in dataset if c["expect_answer"])
    n_refuse = sum(1 for c in dataset if not c["expect_answer"])
    print(f"Built {len(dataset)} cases: {n_answerable} answerable, {n_refuse} should-refuse")
    OUT_PATH.write_text(json.dumps(dataset, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
