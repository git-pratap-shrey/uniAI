# Extract: VLM OCR Extraction Pipelines

## Module Overview

Three VLM-based scripts convert PDFs into structured JSON. Each targets one data type:

| Script | Input | Output | Next Stage |
|---|---|---|---|
| `extract_multimodal_notes.py` | Notes PDFs | `chunk_N_N.json` + `*.txt` | `ingest/ingest_multimodal.py` |
| `extract_multimodal_pyq.py` | PYQ PDFs | `pyqs_processed/*_processed.json` | `ingest/ingest_multimodal_pyq.py` |
| `extract_multimodal_syllabus.py` | Syllabus PDFs | 7 chunk JSONs per PDF | `ingest/ingest_multimodal_syllabus.py` |

Common patterns: render PDF pages to images via PyMuPDF, call VLM via `models.vision()`, retry failed calls with exponential backoff, parse JSON via `extract_first_json()`, skip already-processed files.

---

## Per-File Documentation

### `extract_multimodal_notes.py`

Processes lecture notes page-by-page. `CHUNK_SIZE=1` (one page per chunk).

**Functions:**
- `infer_metadata_from_path(pdf_path) -> dict` -- Parses path to get subject, type, unit from `year_2/<SUBJECT>/notes/<unit>/` structure.
- `render_pages_to_images(doc, start_page, end_page, return_bytes=False, scale=2.0) -> list` -- Renders pages to PIL Images or PNG bytes using `fitz.Matrix(scale, scale)`.
- `process_pdf(pdf_path) -> None` -- Opens PDF, for each page: renders image (scale=1.0 JPEG for Ollama, raw bytes for HF), calls `models.vision()` with `NOTES_EXTRACTION` prompt, retries 3x (15s*attempt backoff), parses JSON, writes `chunk_N_N.json` and appends to `<pdf_stem>.txt`. Output dir: `<pdf.parent>/<pdf_stem>/`.
- `process_all_folders(base_path_str) -> None` -- Finds all PDFs where `"notes" in p.parts`, processes each.

### `extract_multimodal_pyq.py`

Most complex pipeline: VLM OCR + LLM unit classification per question.

**Functions:**
- `get_syllabus_topics(subject) -> str` -- Finds syllabus JSON for subject, extracts unit topics. Fallback to generic titles.
- `load_pdf(pdf_path) -> str` -- Renders each page, calls VLM with `PYQ_VLM_TRANSCRIPTION` prompt per page. Scale=1.0 for Ollama, 1.5 for HF. Retries 3x.
- `normalize_text(text) -> str` -- Strips blank lines, merges continuation lines. Preserves newlines before question patterns (`Q1.`, `1.`, `(a)`) and section headers. Handles hyphen continuations.
- `clean_question_text(q_text) -> tuple(str, int|None)` -- Strips marks from 5 formats: inline `(10 marks)`/`[10]`, pipe-separated `| 2`, trailing numbers, watermarks. Returns `(cleaned, marks)`.
- `detect_metadata(text, pdf_path) -> tuple` -- Extracts `(subject, subject_code, year, program)` from path and text regex. Defaults: year=2023, program="B.Tech".
- `get_unit_classification(question_text, syllabus_text) -> int` -- Calls `models.chat()` with `pyq_unit_classification()` prompt. Returns unit 1-5, default 1 on failure.
- `section_slug(section_label) -> str` -- `"SECTION B"` -> `"sec_b"`.
- `process_pyq(pdf_path) -> None` -- Full pipeline: OCR -> normalize -> detect metadata -> split by sections -> extract each question (clean text, classify unit via LLM, build collision-free ID) -> save JSON array.
- `process_pyq_folders(base_path_str) -> None` -- Finds all PDFs in `pyqs` folders, processes each.

### `extract_multimodal_syllabus.py`

Produces exactly 7 JSON chunks per syllabus PDF.

**Functions:**
- `render_pdf_to_images(pdf_path, scale=2.0) -> list` -- Renders all pages to PIL Images.
- `call_vlm(images, max_retries=3) -> dict|None` -- Calls `models.vision()` with `SYLLABUS_EXTRACTION` prompt, parses JSON.
- `_base_meta(subject, syllabus_version, source_pdf, model) -> dict` -- Base metadata dict.
- `build_unit_chunk(unit_data, base) -> dict` -- Unit chunk with `chunk_type: "unit_N"`, topics, full_text.
- `build_co_chunk(cos, base) -> dict` -- CO chunk with formatted course outcomes.
- `build_books_chunk(textbooks, reference_books, base) -> dict` -- Books chunk with both lists.
- `infer_subject_from_path(pdf_path) -> str` -- Extracts subject from path.
- `process_syllabus(pdf_path, force=False) -> None` -- Renders pages, calls VLM per page, accumulates data, writes 7 JSON chunk files. Skips if all 7 exist (unless forced).
- `process_all_syllabuses(base_path_str, force=False) -> None` -- Finds `*syllabus*.pdf` files, processes each.

---

## Inter-File Relationships

All three scripts share dependencies on `config`, `models.vision()`, `utils` (image encoding, JSON parsing), and `prompts`. Data flow:

```
extract_notes.py  -> chunk_*.json   -> ingest_multimodal.py
extract_pyq.py    -> *_processed.json -> ingest_multimodal_pyq.py
extract_syllabus.py -> syllabus_*.json -> ingest_multimodal_syllabus.py
```

Only the PYQ pipeline uses a second LLM call during extraction (for unit classification).
