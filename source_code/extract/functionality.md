# Extract: VLM OCR Extraction Pipelines

## Module Overview

Three VLM-based scripts convert PDFs into structured JSON. Each targets one data type:

| Script | Input | Output | Next Stage |
|---|---|---|---|
| `extract_multimodal_notes.py` | Notes PDFs | `<pdf_stem>/<pdf_stem>_p{page}-{page}_{chunk_idx}.json` + `*.txt` | `ingest/ingest_multimodal.py` |
| `extract_multimodal_pyq.py` | PYQ PDFs | `pyqs_processed/*_processed.json` | `ingest/ingest_multimodal_pyq.py` |
| `extract_multimodal_syllabus.py` | Syllabus PDFs | 7 chunk JSONs per PDF | `ingest/ingest_multimodal_syllabus.py` |

Common patterns: render PDF pages to images via PyMuPDF, call VLM via `models.vision()`, retry failed calls with exponential backoff, parse JSON via `extract_first_json()`, skip already-processed files.

---

## Per-File Documentation

### `extract_multimodal_notes.py`

Processes lecture notes via **semantic sectioning with a running topic list feedback loop**. One VLM call per page; each page is split into one or more topic-level section JSONs.

**Configuration:**
- `BACKEND` -- vision provider from `CONFIG["providers"]["vision"]` (e.g. `"ollama"`)
- `MODEL_NAME` -- vision model from `CONFIG["providers"]["vision_model"]`

**Functions:**
- `infer_metadata_from_path(pdf_path) -> dict` -- Parses flattened path `<SUBJECT>/notes/unit<N>/*.pdf` to get subject, type, unit. Normalizes unit to numeric string.
- `normalize_unit(unit) -> str` -- Normalizes unit to a clean numeric string (e.g., "unit4" -> "4").
- `render_pages_to_images(doc, start_page, end_page, return_bytes=False, scale=2.0) -> list` -- Renders pages to PIL Images or PNG bytes using `fitz.Matrix(scale, scale)`. Ollama cloud uses scale=1.0 + JPEG to avoid Cloudflare 524 timeouts; HuggingFace uses scale=2.0 PNG.
- `process_pdf(pdf_path) -> None` -- Main per-PDF logic:
  1. Creates output dir at `<pdf.parent>/<pdf.stem>/`
  2. Maintains a **running topic list** (`existing_topics`) across pages — starts empty on page 1
  3. For each page: checks if section JSONs already exist (by glob `<pdf_stem>_p{N}-{N}_*.json`). If yes, **rehydrates** `existing_topics` from those files and skips re-processing.
  4. If page is new: renders image (JPEG for Ollama, PNG bytes for HF), calls `notes_extraction(existing_topics)` prompt, retries 3× with 5s×attempt backoff on failure.
  5. Parses response JSON; extracts `sections[]` array. If no valid JSON, creates a fallback single-section structure.
  6. For each section: appends newly discovered topics to `existing_topics`, writes one JSON file named `<pdf_stem>_p{page}-{page}_{chunk_idx}.json`.
  7. Appends section full_text to running `.txt` file.
- `process_all_folders(base_path_str) -> None` -- Finds all PDFs where `"notes" in p.parts`, processes each.

**Output JSON schema (per section):**
```json
{
  "subject": "COA",
  "type": "notes",
  "unit": "4",
  "source_pdf": "hand_unit1.pdf",
  "page_start": 1,
  "page_end": 1,
  "extracted_metadata": {
    "section_title": "Karnaugh Map",
    "is_new_topic": true,
    "full_text": "...",
    "topics": ["4-variable K-Map", "cell adjacencies"],
    "key_concepts": ["adjacency property"],
    "has_diagram": false,
    "confidence": 0.85,
    "content_quality": "clear"
  },
  "processed_by": "<model>",
  "chunk_size": 1,
  "section_index": 0,
  "chunk_idx": 0
}
```

**Skip logic:** Page is skipped if any `<pdf_stem>_p{N}-{N}_*.json` files exist in the output dir for that page. Topics and text from those files are rehydrated into the running state so subsequent pages get correct context.

**Entry point:** `python -m source_code.extract.extract_multimodal_notes [--path <dir>]`

---

### `extract_multimodal_pyq.py`

Most complex pipeline: VLM OCR + LLM unit classification per question.

**Functions:**
- `get_syllabus_topics(subject) -> str` -- Finds syllabus JSON for subject, extracts unit topics. Glob pattern: `syllabus_unit_*.json`. Fallback to generic titles.
- `load_pdf(pdf_path) -> str` -- Renders each page, calls VLM with `PYQ_VLM_TRANSCRIPTION` prompt per page. Scale=1.0 for Ollama (to avoid Cloudflare timeouts), 1.5 for HF. Retries 3× with **15s×attempt** backoff on failure.
- `normalize_text(text) -> str` -- Strips blank lines, merges continuation lines. Preserves newlines before question patterns (`Q1.`, `1.`, `(a)`) and section headers. Handles hyphen continuations.
- `clean_question_text(q_text) -> tuple(str, int|None)` -- Strips marks from 5 formats: inline `(10 marks)`/`[10]`, watermarks, pipe-separated `| 2`, trailing bare numbers, sub-question prefixes `(a)`. Returns `(cleaned, marks)`.
- `detect_metadata(text, pdf_path) -> tuple` -- Extracts `(subject, subject_code, year, program)` from flattened path `<SUBJECT>/pyqs/` and text regex. Subject is the folder immediately before `pyqs`. Defaults: year=2023, program="B.Tech".
- `get_unit_classification(question_text, syllabus_text) -> int` -- Calls `models.chat()` with `pyq_unit_classification()` prompt. Retries 3× with 15s×attempt backoff. Returns unit 1-5, default 1 on failure.
- `section_slug(section_label) -> str` -- `"SECTION B"` -> `"sec_b"`.
- `process_pyq(pdf_path) -> None` -- Full pipeline: OCR -> normalize -> detect metadata -> split by sections -> extract each question (clean text, classify unit via LLM, build collision-free ID) -> save JSON array. Skips if `pyqs_processed/<stem>_processed.json` already exists.
- `process_pyq_folders(base_path_str) -> None` -- Finds all PDFs in `pyqs` folders (excluding `pyqs_processed`), processes each.

---

### `extract_multimodal_syllabus.py`

Produces exactly 7 JSON chunks per syllabus PDF.

**Functions:**
- `render_pdf_to_images(pdf_path, scale=2.0) -> list` -- Renders all pages to PIL Images.
- `call_vlm(images, max_retries=3) -> dict|None` -- Calls `models.vision()` with `SYLLABUS_EXTRACTION` prompt, parses JSON.
- `_base_meta(subject, syllabus_version, source_pdf, model) -> dict` -- Base metadata dict.
- `build_unit_chunk(unit_data, base) -> dict` -- Unit chunk with `chunk_type: "unit_N"`, topics, full_text.
- `build_co_chunk(cos, base) -> dict` -- CO chunk with formatted course outcomes.
- `build_books_chunk(textbooks, reference_books, base) -> dict` -- Books chunk with both lists.
- `infer_subject_from_path(pdf_path) -> str` -- Extracts subject from flattened path. Subject is the folder immediately before `syllabus` (e.g., `<SUBJECT>/syllabus/file.pdf`). Returns "unknown" if not found.
- `process_syllabus(pdf_path, force=False) -> None` -- Renders pages, calls VLM per page, accumulates data, writes 7 JSON chunk files. Skips if all 7 exist (unless forced).
- `process_all_syllabuses(base_path_str, force=False) -> None` -- Finds `*syllabus*.pdf` files, processes each.

---

## Inter-File Relationships

All three scripts share dependencies on `config`, `models.vision()`, `utils` (image encoding, JSON parsing), and `prompts`. Data flow:

```
extract_notes.py    -> <stem>/<stem>_p*_*_*.json  -> ingest_multimodal.py
extract_pyq.py      -> pyqs_processed/*_processed.json -> ingest_multimodal_pyq.py
extract_syllabus.py -> syllabus_*.json             -> ingest_multimodal_syllabus.py
```

Only the PYQ pipeline uses a second LLM call during extraction (for unit classification).

Data layout is flattened: `<SUBJECT>/notes/unit<N>/*.pdf`, `<SUBJECT>/pyqs/*.pdf`, `<SUBJECT>/syllabus/*.pdf`. No `year_2` nesting.

**Retry timing:**
- Notes: 5s × attempt (up to 3 attempts)
- PYQ (both OCR and classification): 15s × attempt (up to 3 attempts) — conservative to handle Cloudflare rate limits on cloud Ollama
