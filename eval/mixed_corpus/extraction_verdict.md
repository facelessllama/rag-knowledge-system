# Extraction verdict — which documents will actually make it in

Generated from `extraction_probe.json` (762 documents, same `PDFParser`/`SmartChunker` code path as the live `/upload` API — see README.md's "Extraction/chunking probe" section for methodology and the medical false-positive caveat). Regenerate via `generate_verdict.py`, don't hand-edit.

## Bottom line

- **750 / 762** will be accepted by a real `/upload` call (produce ≥1 chunk).
- **12 / 762** will be rejected outright (HTTP 422 "Could not extract text from document" or a parse error) — these never enter the index, there is no chunk to almost-retrieve, don't build questions expecting them to be findable.
- Of the accepted ones, **21** more (all in `hard/*`) read as OCR noise, not real text — technically indexed, but building a fact-based question against them would test Tesseract's inability to read handwriting/dense formulas, not the RAG system. Exclude from golden questions; keep as documented OCR-limitation cases.
- **729 / 762 (95.7%)** is the number to build fact-based golden questions against.
- 13 more (`medical`, all short structured DailyMed labels) hit the same OCR-noise heuristic but were spot-checked and are **real, legible English text** — a genre false positive (low connective-word density is normal for ingredient/warning lists). Not excluded from the usable count above; listed below only so a manual skim can double-check the rest.
- 3 documents have too little extracted text (<15 word-tokens) for the stopword heuristic to judge either way — manual look recommended.

## Will NOT be ingested (12)

Real `/upload` raises `ValueError("Could not extract text from document")` → HTTP 422 → rolled back, never stored. These simply won't exist in the index; asking a question that expects one of them to be findable is testing nothing.

| category | file | chars | reason |
|---|---|---|---|
| hard_headers_footers | olmocr_headers_footers_b4c3c4ac3d6f7b52a993cec7ca8b3ad43cecabad_page_3.pdf | 64 | 0 chunks (all pages < 100 chars) |
| hard_old_scans | olmocr_old_scans_50.pdf | 0 | 0 chunks (all pages < 100 chars) |
| hard_old_scans | olmocr_old_scans_40.pdf | 54 | 0 chunks (all pages < 100 chars) |
| hard_old_scans | olmocr_old_scans_76.pdf | 0 | 0 chunks (all pages < 100 chars) |
| hard_old_scans | olmocr_old_scans_14.pdf | 0 | 0 chunks (all pages < 100 chars) |
| hard_old_scans | olmocr_old_scans_21.pdf | 76 | 0 chunks (all pages < 100 chars) |
| hard_old_scans | olmocr_old_scans_29.pdf | 72 | 0 chunks (all pages < 100 chars) |
| hard_old_scans | olmocr_old_scans_38.pdf | 0 | 0 chunks (all pages < 100 chars) |
| hard_old_scans | olmocr_old_scans_41.pdf | 0 | 0 chunks (all pages < 100 chars) |
| hard_old_scans | olmocr_old_scans_90.pdf | 81 | 0 chunks (all pages < 100 chars) |
| hard_old_scans | olmocr_old_scans_37.pdf | 49 | 0 chunks (all pages < 100 chars) |
| hard_old_scans_math | olmocr_old_scans_math_1_pg63.pdf | 89 | 0 chunks (all pages < 100 chars) |

## Ingests, but reads as OCR noise — exclude from golden questions (21)

Clears the chunker's 100-char gate, so it *will* sit in the index, but English-stopword ratio < 0.12 says it's garbled recognition output, not readable text — verified by hand against raw OCR text on several of these (pure noise, e.g. `old_scans/43.pdf`: `"| a ariel\nes ae i ee AC gn..."`). All from the olmOCR-bench `hard/*` slices, i.e. exactly the adversarial-scan content this corpus deliberately includes to find this failure mode.

| category | file | chars | stopword_ratio | tokens |
|---|---|---|---|---|
| hard_headers_footers | olmocr_headers_footers_78643402ed01cc7523be74da4652e7a8e81bd426_page_1.pdf | 977 | 0.058 | 69 |
| hard_long_tiny_text | olmocr_long_tiny_text_17_pg34_pg1.pdf | 332 | 0.021 | 48 |
| hard_long_tiny_text | olmocr_long_tiny_text_17_pg4_pg1.pdf | 850 | 0.035 | 114 |
| hard_multi_column | olmocr_multi_column_02a3e23da54b82c414d272051b0b5d8f44d8_page_12_pg1.pdf | 3808 | 0.0 | 489 |
| hard_multi_column | olmocr_multi_column_09f801e3a2ec90ef456d34ad571a46f36fce_page_40_pg1.pdf | 3046 | 0.003 | 343 |
| hard_multi_column | olmocr_multi_column_00187fe0533b1e8ddc748adab4924b6f7099_page_10_pg1.pdf | 6799 | 0.113 | 644 |
| hard_old_scans | olmocr_old_scans_77.pdf | 1609 | 0.0 | 124 |
| hard_old_scans | olmocr_old_scans_13.pdf | 161 | 0.0 | 20 |
| hard_old_scans | olmocr_old_scans_7.pdf | 180 | 0.0 | 18 |
| hard_old_scans | olmocr_old_scans_84.pdf | 1529 | 0.005 | 185 |
| hard_old_scans | olmocr_old_scans_57.pdf | 1762 | 0.006 | 165 |
| hard_old_scans | olmocr_old_scans_92.pdf | 1100 | 0.009 | 106 |
| hard_old_scans | olmocr_old_scans_98.pdf | 2221 | 0.013 | 229 |
| hard_old_scans | olmocr_old_scans_49.pdf | 484 | 0.02 | 50 |
| hard_old_scans | olmocr_old_scans_73.pdf | 177 | 0.059 | 17 |
| hard_old_scans_math | olmocr_old_scans_math_1_pg72.pdf | 323 | 0.04 | 25 |
| hard_tables | olmocr_tables_9e6a5b213e4fd5906842ebe6b0c5538e9e7b_pg12.pdf | 1315 | 0.022 | 135 |
| hard_tables | olmocr_tables_46274c2c7e7e925edf6541914f1841fbb4f0_pg139.pdf | 1089 | 0.026 | 114 |
| hard_tables | olmocr_tables_8160caa0f0e21fbd0ef674de8759df133ccf_pg12_pg1.pdf | 614 | 0.057 | 53 |
| hard_tables | olmocr_tables_137297b4b1d29d3ff3b0eb2f7295670c4735_pg4_pg1.pdf | 1077 | 0.084 | 83 |
| hard_tables | olmocr_tables_94f7559a72a6cc9affb8a2487c983304a95e_pg18_pg1.pdf | 463 | 0.095 | 42 |

## Flagged by the same heuristic, but a confirmed false positive — keep (13)

All `medical` (DailyMed drug labels): short, structured, born-digital English (ingredient lists, dosage tables, warnings) with naturally few connective words. Spot-checked `COLGATE_KIDS...pdf` (stopword_ratio 0.101) by hand — perfectly legible, correctly extracted "Drug Facts" label text. Trust this list less than the `hard/*` one above; a quick skim of the rest (13 documents, cheap) is worth doing before building questions from them, but don't drop them on the heuristic's say-so alone.

| file | chars | stopword_ratio | tokens |
|---|---|---|---|
| dailymed_2dcd4816-caa5-c006-e063-6394a90ac487_FRESHEN_UP_SODIUM_FLUORIDE_GEL_DENTIFRICE_MKJ_BRANDS_LLC.pdf | 2682 | 0.063 | 285 |
| dailymed_56b8d0cc-2f35-7d38-e063-6294a90ad06d_OWELL_NATURALS_NEUROPATHY_MAXIMUM_STRENGTH_MENTHOL_CREAM_OWELL_N.pdf | 2771 | 0.07 | 314 |
| dailymed_56aacf53-f9ed-78c5-e063-6394a90aa4ac_FRESHEN_UP_CHAI_LATTE_MINT_SODIUM_FLUORIDE_GEL_DENTIFRICE_DABUR.pdf | 2770 | 0.074 | 311 |
| dailymed_56aad223-db88-7cf6-e063-6394a90a6b72_FRESHEN_UP_VANILLA_MINT_SODIUM_FLUORIDE_GEL_DENTIFRICE_DABUR_IND.pdf | 2725 | 0.076 | 301 |
| dailymed_56ab500f-83fe-f3d7-e063-6394a90a14b1_BENZALKONIUM_CHLORIDE_LIQUID_H_E_B.pdf | 3002 | 0.097 | 319 |
| dailymed_56a3ad5f-c34a-6b90-e063-6394a90a06e9_HAND_SANITIZER_01_ALCOHOL_SPRAY_COSMUSES_COSMETICS_NINGBO_CO._LT.pdf | 2081 | 0.097 | 227 |
| dailymed_e8398535-60eb-46fa-8f9f-dff59045a192_COLGATE_KIDS_MILD_BUBBLE_FRUIT_FLAVOR_SODIUM_FLUORIDE_GEL_DENTIF.pdf | 5066 | 0.101 | 565 |
| dailymed_56b26868-833c-3d72-e063-6394a90aea4f_NON_-ALCOHOL_PEACH_SCENTED_HAND_SANITIZER_BENZALKONIUM_CHLORIDE.pdf | 2279 | 0.11 | 255 |
| dailymed_56b5c19e-cce5-16f8-e063-6394a90a1cbd_NON_-ALCOHOL_GRAPE_SCENTED_HAND_SANITIZER_BENZALKONIUM_CHLORIDE.pdf | 2279 | 0.11 | 255 |
| dailymed_56b67ff4-5785-106e-e063-6294a90af4ec_NON_-ALCOHOL_BLUEBERRY_SCENTED_HAND_SANITIZER_BENZALKONIUM_CHLOR.pdf | 2292 | 0.11 | 255 |
| dailymed_569dee15-e68d-220e-e063-6294a90af7c5_OAKDOLCHE_TEA_TREE_OIL_ANTIFUNGAL_TOLNAFTATE_SOAP_NANJING_QIAOTU.pdf | 3894 | 0.112 | 421 |
| dailymed_56b9a769-08fb-738f-e063-6294a90a9eb2_GELFOS-M_ALUMINUM_PHOSPHATE_COLLOIDAL_MAGNESIUM_HYDROXIDE_SIMETH.pdf | 2906 | 0.113 | 309 |
| dailymed_b7b3c15a-bb28-5c25-e053-2a95a90ad3b1_PENTREXCILINA_DAYTIME_ACETAMINOPHEN_CHLORPHENIRAMINE_MALEATE_PHE.pdf | 10214 | 0.116 | 1124 |

## Too little text for the heuristic to judge (3)

| category | file | chars | tokens |
|---|---|---|---|
| hard_old_scans | olmocr_old_scans_43.pdf | 102 | 4 |
| hard_old_scans | olmocr_old_scans_89.pdf | 302 | 12 |
| hard_headers_footers | olmocr_headers_footers_b329b8083ee2abe62999617011b888e980924905_page_3.pdf | 117 | 14 |

