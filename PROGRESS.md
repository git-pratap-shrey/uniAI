# Project Progress — uniAI

Last updated: 2026-04-05

## Pipeline: Syllabus Extraction & Ingestion

| Subject | PDFs | Extracted | Ingested | Status |
|---|---|---|---|---|
| CYBER_SECURITY | 1 | 7 JSONs | Pending | ✅ Extracted |
| DIGITAL_ELECTRONICS | 1 | 7 JSONs | Pending | ✅ Extracted |
| OOPS_JAVA | 1 | 7 JSONs (subject_name fixed) | Pending | ✅ Extracted |
| OPERATING_SYSTEM | 1 | 7 JSONs | Pending | ✅ Extracted |
| TAFL | 1 | 7 JSONs | Pending | ✅ Extracted |
| UHV | 1 | 7 JSONs | Pending | ✅ Extracted |

- Total: 6/6 syllabus PDFs extracted, 42/42 JSON files produced
- OOPS_JAVA `subject_name` manually corrected
- 3 files have minor VLM truncation artifacts (low priority, deferred to re-extract cycle)
- `get_syllabus_topics()` glob bug fixed in code (`chunk_*.json` → `syllabus_unit_*.json`)

## Pipeline: PYQ Extraction & Ingestion

| Subject | PDFs | Processed | Questions | Status |
|---|---|---|---|---|
| CYBER_SECURITY | 5 | 5 | ~118 | ✅ Done |
| DIGITAL_ELECTRONICS | 4 | 3 | ~80 | ⚠️ 1 zero-yield |
| OOPS_JAVA | 1 | 1 | ~28 | ✅ Done |
| OPERATING_SYSTEM | 5 | 5 | ~103 | ✅ Done |
| TAFL | 5 | 5 | ~143 | ✅ Done |
| UHV | 8 | 7 | ~171 | ✅ Done |

- Total: 26/28 PYQ PDFs processed, ~643 valid questions
- **2 zero-yield PDFs**: `DIGITAL_ELECTRONICS/pyqs/22.pdf`, `UHV/pyqs/22.pdf` — fill-in-the-blank format doesn't match question regex

### PYQ Code Fixes Applied
- Unique question IDs (running counter to prevent collisions)
- Sub-question prefix stripping (`(a)`, `(b)`, `(i)`) from question_text
- Ingestion dedup guard (silent overwrite protection)
- All 25 existing processed files cleaned of duplicate IDs and trailing dots

### Remaining PYQ Issues
| Issue | Severity | Action |
|---|---|---|
| DE/22, UHV/22 zero-yield | Low | Requires new regex pattern or manual extraction |
| 73% marks=null | Info | Older paper formats don't include marks — not a bug |
| No automated PYQ extraction tests | Medium | Add test for extraction + validation |

## Pipeline: Notes Extraction & Ingestion

| Subject | PDFs | Status |
|---|---|---|
| CYBER_SECURITY | 10 | ❌ Not started |
| DIGITAL_ELECTRONICS | 8 | ❌ Not started |
| OOPS_JAVA | 10 | ❌ Not started |
| OPERATING_SYSTEM | 10 | ❌ Not started |
| TAFL | 10 | ❌ Not started |
| UHV | 10 | ❌ Not started |

- Total: 58 notes PDFs, none extracted yet
- Pipeline scripts exist: `extract/extract_multimodal_notes.py`, `ingest/ingest_multimodal.py`

## Pipeline: ChromaDB

| Collection | Populated |
|---|---|
| `multimodal_notes` | ❌ Empty |
| `multimodal_syllabus` | ❌ Populated |
| `multimodal_pyq` | ❌ Populated |

- ChromaDB path: `source_code/chroma/`
- Syllabus and PYQ ingestion scripts are ready; need to be run after data validation

## Code: Fixes Applied This Session

| File | Fix | Description |
|---|---|---|
| `extract/extract_multimodal_pyq.py` | A | Unique question IDs with running counter |
| `extract/extract_multimodal_pyq.py` | B | Strip `(a)/(b)` sub-question prefixes from question_text |
| `extract/extract_multimodal_pyq.py` | C | Fixed syllabus glob: `chunk_*.json` → `syllabus_unit_*.json` |
| `ingest/ingest_multimodal_pyq.py` | D | Added `seen_ids` dedup guard on upsert |
| `data/year_2/OOPS_JAVA/syllabus/*.json` | Data | Fixed `subject_name` across 7 files |
| `data/year_2/CYBER_SECURITY/syllabus/` | Data | Patched truncated topics (2 files) |
| `data/year_2/OOPS_JAVA/syllabus/` | Data | Patched truncated topics + book citation artifacts |
| `data/year_2/*/pyqs/pyqs_processed/*` | Data | Cleaned duplicate IDs, trailing dots, header garbage in 25 files |

## Code: Fixes Pending / Improvements

| File | Priority | Description |
|---|---|---|
| `extract/extract_multimodal_pyq.py` | Low | Add regex pattern for `Q1(a)` fill-in-the-blank format |
| `ingest_multimodal_pyq.py` | Low | Add `--force` flag support |
| `extract/extract_multimodal_syllabus.py` | Low | Add validation/warning on `subject_name` mismatch |
| Tests | Medium | Add automated PYQ extraction test |
| Tests | Medium | Add syllabus extraction validation test |

## Next Steps (Recommended Order)

1. Run PyQ ingestion into ChromaDB (data is ready)
2. Run syllabus ingestion into ChromaDB (data is ready)
3. Run notes extraction for all 58 PDFs
4. Run notes ingestion
5. Run full RAG pipeline tests
6. Fix the 2 remaining zero-yield PYQs
