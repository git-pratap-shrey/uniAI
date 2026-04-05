# Ingest: ChromaDB Ingestion Pipelines

## Module Overview

The `ingest/` package contains three scripts that load structured JSON (produced by the `extract/` pipeline) into three isolated ChromaDB collections. Each script handles one data type and targets a specific collection:

| Script | Input JSONs | Target Collection | Document IDs |
|---|---|---|---|
| `ingest_multimodal.py` | `<pdf_stem>_p*-*.json` (notes sections) | `multimodal_notes` | `{SUBJECT}_unit{unit}_notes_{pdf_stem}_p{start}-{end}_{chunk_idx}` |
| `ingest_multimodal_pyq.py` | `*_processed.json` (PYQs) | `multimodal_pyq` | `{question_id}` from extraction |
| `ingest_multimodal_syllabus.py` | `syllabus_*.json` | `multimodal_syllabus` | `syllabus_{SUBJECT}_{source_pdf}_{chunk_type}` |

All three scripts follow the same pattern:
1. Scan for JSON files using `pathlib.rglob()`
2. Load each JSON, build a rich embedding text string from the content
3. Generate an embedding via `utils.get_embedding()` (wraps `models.embed` -> Ollama)
4. Upsert into ChromaDB with metadata, skipping already-existing documents
5. Report counts of ingested/skipped/errored items

The collection isolation is foundational to the RAG system: the three data types are never mixed in the same vector space, enabling targeted retrieval (e.g., query notes for explanations, syllabus for scope, PYQs for exam patterns).

---

## Per-File Documentation

### `ingest_multimodal.py`

Ingests lecture notes section JSONs (produced by `extract_multimodal_notes.py`) into the `multimodal_notes` ChromaDB collection.

**Constants:**
- `BASE_PATH` -- from `CONFIG["paths"]["base_data"]`

**Functions:**

`normalize_unit(unit) -> str`
- Input: Unit value (any type: int, string like "unit1", "Unit 3", "UNIT-1", None)
- Output: Clean numeric string like "1", "2", or "unknown"
- Logic: Converts to string, lowercases, extracts first digit sequence via regex, strips leading zeros.

`is_garbage_chunk(section_meta: dict) -> bool`
- Input: Section-level extracted metadata dict
- Output: True if the section should be skipped
- Logic: Checks three criteria (any one is sufficient to reject):
  1. `section_title` is in `_GARBAGE_TITLES` blocklist
  2. `full_text` contains 2+ promotional keywords from `_PROMO_KEYWORDS`
  3. `full_text < 80` chars with no topics and no key concepts

`build_embedding_text(data: dict) -> str`
- Input: Full JSON data dict (with `extracted_metadata` wrapper containing section-level fields)
- Output: Rich embedding string: "Subject: COA | Unit: 4 | Type: notes | Topics: a, b | Concepts: c, d | Title: RISC Architecture\n\n<full_text (truncated to 4000)>"
- Logic: Topics moved before title for more embedding weight. `Type: notes` added for document context. Title demoted to last position (display hint only). Falls back to empty string if no content.

`ingest_descriptions() -> None`
- Input: None (reads from `BASE_PATH` using config)
- Output: Side effects -- upserts documents into `multimodal_notes` ChromaDB collection
- Logic:
  1. Opens the `multimodal_notes` collection via `utils.get_chroma_collection()`
  2. Finds all `<pdf_stem>_p*_*_*.json` files under `BASE_PATH` (section-level output from the new extractor)
  3. For each JSON: skips if file contains a `"sections"` key (per-page file), if confidence < 0.3, if garbage, if empty text, or if ID already exists
  4. Builds info-rich document ID: `{SUBJECT}_unit{unit}_notes_{pdf_stem}_p{start}-{end}_{chunk_idx}`
  5. Builds embedding text, generates vector, upserts with updated metadata schema
  6. Reports final counts

**Metadata stored per document:**
`source`, `page_start`, `page_end`, `page_count` (NEW), `unit`, `subject`, `title` (section_title), `document_type`, `chunk_idx` (NEW), `section_index` (NEW), `confidence`

**NOT stored (removed):** `topics` and `topics_str` -- ChromaDB metadata fields must be scalar. Topics are captured in the embedding text and retrievable semantically.

**Entry point:** `python ingest_multimodal.py`

---

### `ingest_multimodal_pyq.py`

Ingests processed PYQ JSONs (produced by `extract_multimodal_pyq.py`) into the `multimodal_pyq` ChromaDB collection.

**Constants:**
- `BASE_PATH` -- from `CONFIG["paths"]["base_data"]`

**Functions:**

`build_pyq_embedding_text(q: dict) -> str`
- Input: Single question dict from the processed PYQ JSON
- Output: "Subject: COA | Unit: 3 | Year: 2023\n\nQuestion:\n<question_text>"
- Logic: Prefixes available metadata (subject, unit, year), then the actual question text.

`ingest_pyqs() -> None`
- Input: None
- Output: Side effects -- upserts each question into `multimodal_pyq` collection
- Logic:
  1. Opens `multimodal_pyq` collection
  2. Finds all `pyqs_processed/*_processed.json` files
  3. For each JSON, iterates question list: skips if no question_id, if ID exists, if text too short
  4. Generates embedding, upserts with metadata
  5. Reports counts

**Metadata stored per document:**
`source`, `unit`, `subject`, `document_type: "pyq"`, `year`, `marks`

**Entry point:** `python ingest_multimodal_pyq.py`

---

### `ingest_multimodal_syllabus.py`

Ingests syllabus chunk JSONs (produced by `extract_multimodal_syllabus.py`) into the `multimodal_syllabus` ChromaDB collection.

**Functions:**

`build_syllabus_embedding_text(data: dict) -> str`
- Input: Syllabus chunk dict (flat schema, no `extracted_metadata` wrapper)
- Output: "Subject: Computer Org | Syllabus: BCS302 | Unit: 3 | Title: X | Section: Unit 3 | Topics: a, b\n\n<full_text>"
- Logic: Builds prefix from subject, syllabus_version, unit, unit_title, chunk_type (title case), topics. Appends full_text truncated to 4000 chars.

`ingest_syllabuses() -> None`
- Logic:
  1. Opens `multimodal_syllabus` collection
  2. Finds all `syllabus_*.json` files
  3. Skips if `type != "syllabus"`, empty text, or ID exists
  4. Generates embedding, upserts with metadata
  5. Reports counts (ingested, skipped, errors)

**Metadata stored per document:**
Standard: `source`, `page_start: 0`, `page_end: 0`, `unit`, `subject`, `title`, `document_type: "syllabus"`, `confidence: 1.0`
Syllabus-specific: `syllabus_version`, `chunk_type`

**Entry point:** `python ingest_multimodal_syllabus.py`

---

## Inter-File Relationships

All three scripts share the same dependency pattern:
- `utils.py` provides `get_embedding()` -> `models.embed()` -> Ollama and `get_chroma_collection()` -> chromadb.PersistentClient
- `config.py` provides CONFIG (paths, collection names, thresholds)

**Data flow from extraction to ingestion:**
- extract_multimodal_notes.py -> <pdf_stem>_p*_*_*.json -> ingest_multimodal.py -> multimodal_notes
- extract_multimodal_pyq.py -> *_processed.json -> ingest_multimodal_pyq.py -> multimodal_pyq
- extract_multimodal_syllabus.py -> syllabus_*.json -> ingest_multimodal_syllabus.py -> multimodal_syllabus

The garbage filter (`is_garbage_chunk`) is unique to notes ingestion -- only notes PDFs are prone to promotional watermarking. The `normalize_unit()` function only appears in the notes ingestion script; the PYQ and syllabus scripts receive already-normalized units from their extraction pipelines.

The ingest pipeline now handles section-level JSON files (one per semantic topic section per page) instead of page-level chunks. Document IDs include unit, pdf_stem, page range, and chunk index for debugging. Metadata uses only scalar fields -- topic arrays are stored in the embedding text, not in metadata.
