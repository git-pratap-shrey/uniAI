# Not In Use Scripts

## Module Overview

Legacy scripts from earlier iterations of the uniAI project. Superseded by the structured `extract/`, `ingest/`, `pipeline/`, and `rag/` packages. None are imported or executed by the current codebase.

---

## Per-File Documentation

### `OCRconvert.py`
Early local OCR script. Rendered PDF pages via PyMuPDF (2x zoom) and requested OCR from a local Ollama vision model, saving output as raw text. **Retired** because it only produced plain text without metadata, unlike later iterations that produce structured JSON.

### `convert.py`
Used Google Cloud Vision API for PDF-to-text OCR. Rendered pages via PyMuPDF (2x zoom), called `document_text_detection()`, and hardcoded a Google credentials path. **Retired** in favor of local vision models to avoid API costs and reliance on external services.

### `extract_text.py`
Local OCR extraction with basic metadata inference. Used PyMuPDF and an Ollama vision model for OCR, and inferred metadata (year, subject, doc type, unit) from the folder structure, saving text and metadata separately. **Retired** because VLM extraction in the main pipeline replaced plain text extraction with structured JSON output directly containing title, unit, topics, etc.

### `ingest_python.py`
Early text-based chunking script. Read extracted `.txt` files and classified blocks by content type (definition, advantages, algorithm, steps, comparison, formula, etc.), assigned exam priority, split on academic signals, and merged tiny chunks. Output to `chunks_ready_for_embedding.jsonl`. **Retired** because the structured VLM extraction replaced these text heuristics with better multimodal understanding.

### `query_python.py`
Early ChromaDB query script hardcoded for a single collection (`python`) with Windows paths. Features `detect_unit_query()` (regex unit detection), and `chroma_query()` (metadata filtering with dummy zero-vectors for exact unit queries, semantic search otherwise). **Retired** -- replaced by modular and generic `rag/` search functions.

### `rag_chat.py`
Early monolithic RAG chatbot: keyword scoring subject detection with LLM fallback routing, threshold-filtered retrieval, PYQ lookup with 0.72 similarity threshold, prompt assembly, Ollama generation, and a `/switch` command. **Retired** -- replaced by the modular `rag/` package with a cleaner routing waterfall.

### `subject_keywords.json`
Legacy flat-format keyword map: `{"COA": [...], "PYTHON": [...]}` with noise like "unit 11", "unit none". **Retired** -- replaced by hierarchical LLM-generated map with notes/syllabus/pyq buckets and core/specific splits.

### `cleanup_data.py`
Deleted ChromaDB and all non-PDF files from the data directory. Required a `--force` flag or interactive confirmation. **Retired** -- replaced by manual resets or the newer ingestion pipeline's skip-existing-id behavior.

---

## Evolution Timeline

1. Google Vision OCR (`convert.py`) -> Local Ollama Vision text OCR (`OCRconvert.py` / `extract_text.py`) -> Structured VLM JSON extraction (`extract/`)
2. Manual text chunking heuristics (`ingest_python.py`) -> VLM-extracted structured JSON chunks
3. Single-subject hardcoded search (`query_python.py`) -> Monolithic chat (`rag_chat.py`) -> Modular multi-subject RAG package (`rag/`)
4. Flat manual keyword map (`subject_keywords.json`) -> Hierarchical LLM-generated map
