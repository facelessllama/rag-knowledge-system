"""
Tests for ingestion/chunker.py — sentence/paragraph-aware chunking.
"""
from types import SimpleNamespace

from ingestion.chunker import (
    SmartChunker, normalize_whitespace, chunk_context_text, extract_case_metadata, extract_celex_id,
    extract_citation_number,
)


def _pages(*texts):
    return [{"page_num": i + 1, "text": t, "has_ocr": False} for i, t in enumerate(texts)]


def test_chunk_document_respects_sentence_boundaries():
    chunker = SmartChunker(chunk_size=50, chunk_overlap=10, min_chunk_size=10)
    text = "First sentence here. Second sentence follows. Third one too. Fourth sentence caps it off."
    chunks = chunker.chunk_document(_pages(text), doc_id="doc1")
    assert len(chunks) >= 2
    for c in chunks:
        assert c.text.strip()
        assert c.document_id == "doc1"
        assert c.page_num == 1


def test_chunk_document_applies_overlap_between_chunks():
    chunker = SmartChunker(chunk_size=40, chunk_overlap=15, min_chunk_size=5)
    text = "Alpha bravo charlie delta. Echo foxtrot golf hotel. India juliet kilo lima."
    chunks = chunker.chunk_document(_pages(text), doc_id="doc1")
    assert len(chunks) >= 2
    # Overlap text from the end of chunk N should reappear at the start of chunk N+1
    tail = chunks[0].text[-15:].strip()
    assert any(tail.split()[-1] in chunks[1].text for tail in [tail] if tail.split())


def test_chunk_document_skips_short_pages():
    chunker = SmartChunker(chunk_size=500, chunk_overlap=50, min_chunk_size=100)
    chunks = chunker.chunk_document(_pages("Too short."), doc_id="doc1")
    assert chunks == []


def test_chunk_document_indexes_are_sequential_across_pages():
    chunker = SmartChunker(chunk_size=30, chunk_overlap=5, min_chunk_size=5)
    page1 = "One sentence here now. Another sentence follows soon."
    page2 = "Third sentence on page two. Fourth sentence closes it out."
    chunks = chunker.chunk_document(_pages(page1, page2), doc_id="doc1")
    indices = [c.chunk_index for c in chunks]
    assert indices == list(range(len(chunks)))
    page_nums = {c.page_num for c in chunks}
    assert page_nums == {1, 2}


def test_chunk_document_russian_sentence_splitting():
    chunker = SmartChunker(chunk_size=200, chunk_overlap=20, min_chunk_size=5)
    text = "Первое предложение. Второе предложение! Третье предложение?"
    chunks = chunker.chunk_document(_pages(text), doc_id="doc1")
    assert len(chunks) >= 1
    assert "Первое" in chunks[0].text


def test_chunk_ids_are_unique_and_deterministic():
    chunker = SmartChunker(chunk_size=30, chunk_overlap=5, min_chunk_size=5)
    text = "One sentence here now. Another sentence follows soon. Third one too here."
    chunks = chunker.chunk_document(_pages(text), doc_id="docA")
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    assert all(cid.startswith("docA_p1_c") for cid in ids)


def test_char_offsets_slice_back_to_exact_chunk_text():
    """The whole TXT-viewer highlighting feature depends on this invariant:
    normalize_whitespace(page_text)[chunk.char_start:chunk.char_end] == chunk.text
    for every chunk, including overlap-prefixed ones."""
    chunker = SmartChunker(chunk_size=80, chunk_overlap=20, min_chunk_size=10)
    text = (
        "First sentence here is short. Second sentence follows right after. "
        "Third one is a bit longer than the others. Fourth sentence closes it off nicely. "
        "Статья 15.1 гласит о пункте три. Договор аренды заключён между сторонами добросовестно."
    )
    chunks = chunker.chunk_document(_pages(text), doc_id="doc1")
    normalized = normalize_whitespace(text)
    assert len(chunks) >= 2
    for c in chunks:
        assert c.char_start is not None and c.char_end is not None
        assert normalized[c.char_start:c.char_end] == c.text


def test_chunk_document_hard_wraps_run_on_text_with_no_sentence_punctuation():
    """Russian statutes routinely list clauses separated by semicolons/commas
    with no '.', '!', '?' for thousands of chars. Without a fallback, that
    whole stretch becomes one oversized atomic chunk, which is both slow to
    embed and semantically diluted."""
    chunker = SmartChunker(chunk_size=100, chunk_overlap=10, min_chunk_size=10)
    run_on = " ".join(f"пункт{i} слово{i}" for i in range(200))  # no . ! ? at all
    chunks = chunker.chunk_document(_pages(run_on), doc_id="doc1")
    assert len(chunks) >= 2
    for c in chunks:
        assert c.char_count <= 200  # <= 2x chunk_size safety margin, no pathological outliers


def test_char_offsets_are_page_relative_not_document_relative():
    chunker = SmartChunker(chunk_size=20, chunk_overlap=5, min_chunk_size=5)
    page1 = "First page sentence one. First page sentence two here."
    page2 = "Second page sentence one. Second page sentence two here."
    chunks = chunker.chunk_document(_pages(page1, page2), doc_id="doc1")
    normalized1 = normalize_whitespace(page1)
    normalized2 = normalize_whitespace(page2)
    for c in chunks:
        expected = normalized1 if c.page_num == 1 else normalized2
        assert expected[c.char_start:c.char_end] == c.text


# ── chunk_context_text ───────────────────────────────────────────────────────

def test_chunk_context_text_prepends_filename_as_title():
    chunk = SimpleNamespace(filename="ABLAPL_3648_2020_LIPU_PRADHAN_vs_STATE_OF_ODISHA.pdf",
                             text="Heard learned counsel for the petitioner.")
    result = chunk_context_text(chunk)
    assert result == "ABLAPL 3648 2020 LIPU PRADHAN vs STATE OF ODISHA: Heard learned counsel for the petitioner."


def test_chunk_context_text_falls_back_to_bare_text_without_filename():
    chunk = SimpleNamespace(filename="", text="some chunk text")
    assert chunk_context_text(chunk) == "some chunk text"


def test_chunk_context_text_handles_missing_filename_attribute():
    chunk = SimpleNamespace(text="some chunk text")  # no .filename at all
    assert chunk_context_text(chunk) == "some chunk text"


def test_chunk_context_text_does_not_mutate_original_text():
    chunk = SimpleNamespace(filename="doc.txt", text="original")
    chunk_context_text(chunk)
    assert chunk.text == "original"


def test_chunk_context_text_accepts_dict_with_filename():
    chunk = {"filename": "ABLAPL_3648_2020_LIPU_PRADHAN_vs_STATE_OF_ODISHA.pdf",
              "text": "Heard learned counsel for the petitioner."}
    result = chunk_context_text(chunk)
    assert result == "ABLAPL 3648 2020 LIPU PRADHAN vs STATE OF ODISHA: Heard learned counsel for the petitioner."


def test_chunk_context_text_accepts_dict_without_filename():
    assert chunk_context_text({"text": "some chunk text"}) == "some chunk text"
    assert chunk_context_text({"filename": "", "text": "some chunk text"}) == "some chunk text"


# ── extract_case_metadata ─────────────────────────────────────────────────────

def test_extract_case_metadata_parses_number_year_and_parties():
    result = extract_case_metadata("ABLAPL_3648_2020_LIPU_PRADHAN_vs_STATE_OF_ODISHA.pdf")
    assert result == {
        "case_number": "3648",
        "case_year": "2020",
        "parties": ["LIPU PRADHAN", "STATE OF ODISHA"],
    }


def test_extract_case_metadata_handles_multi_underscore_type_prefix():
    """Type prefix (e.g. 'CR. MISC.') may itself contain underscores — the
    non-greedy match must not eat into the number/year."""
    result = extract_case_metadata("CR._MISC._100_2019_A_B_vs_C_D.pdf")
    assert result["case_number"] == "100"
    assert result["case_year"] == "2019"
    assert result["parties"] == ["A B", "C D"]


def test_extract_case_metadata_returns_empty_for_non_case_filenames():
    """Plain-text UK statutory instruments have no case number/parties."""
    assert extract_case_metadata("The_Localism_Act_2011.txt") == {}
    assert extract_case_metadata("") == {}


# ── extract_celex_id ─────────────────────────────────────────────────────────

def test_extract_celex_id_parses_plain_id():
    assert extract_celex_id("31997R0955 - Commission Regulation (EC) No 955 97.pdf") == "31997R0955"


def test_extract_celex_id_parses_id_with_suffix():
    assert extract_celex_id("31958D1127(01) - EEC Council Rules of the Transport Committee.pdf") == "31958D1127(01)"


def test_extract_celex_id_returns_none_for_non_celex_filenames():
    assert extract_celex_id("ABLAPL_3648_2020_LIPU_PRADHAN_vs_STATE_OF_ODISHA.pdf") is None
    assert extract_celex_id("The_Localism_Act_2011.txt") is None
    assert extract_celex_id("") is None


# ── extract_citation_number ────────────────────────────────────────────────────

def test_extract_citation_number_parses_four_digit_year():
    result = extract_citation_number(
        "32008R1021 - Commission Regulation (EC) No 1021 2008 of 17 October 2008 amending Annexes.pdf"
    )
    assert result == {"citation_number": "1021", "citation_year": "2008"}


def test_extract_citation_number_uses_full_date_year_not_short_form_year():
    """The preamble's own leading year token can be 2-digit ("No 3833 88"),
    but the trailing full date ("of 3 December 1984") always carries the
    unambiguous 4-digit year — that's the one that must end up in the
    result, not whatever digit count the short form happened to use."""
    result = extract_citation_number(
        "31988R3833 - Commission Regulation (EEC) No 3833 88 of 3 December 1988 amending Regulation.pdf"
    )
    assert result == {"citation_number": "3833", "citation_year": "1988"}


def test_extract_citation_number_returns_empty_for_non_matching_filenames():
    assert extract_citation_number("ABLAPL_3648_2020_LIPU_PRADHAN_vs_STATE_OF_ODISHA.pdf") == {}
    assert extract_citation_number("The_Localism_Act_2011.txt") == {}
    assert extract_citation_number("") == {}
    # older "<N> <YY> EEC" filenames with no "No ... of <date>" preamble
    assert extract_citation_number("31958D1127(01) - EEC Council Rules of the Transport Committee.pdf") == {}
