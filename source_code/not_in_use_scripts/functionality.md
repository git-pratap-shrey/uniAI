# Not In Use Scripts

## Module Overview

Legacy scripts from earlier iterations of the uniAI project. Superseded by the structured `extract/`, `ingest/`, `pipeline/`, and `rag/` packages. None are imported or executed by the current codebase.

---

## Per-File Documentation

### `OCRconvert.py`
Used Google Cloud Vision API for PDF-to-text OCR. Rendered pages via PyMuPDF (2x zoom), called `document_text_detection()`, hardcoded Google credentials path. **Retired** because it only produced raw text without metadata, unlike VLM extraction which produces structured JSON with title, unit, topics, and confidence.

### `extract_text.py`
Early text-based chunking. Classified blocks by content type (definition, advantages, algorithm, steps, comparison, formula, etc.), assigned exam priority (high/medium/low), split on academic signals, merged tiny chunks. **Retired** because VLM extraction replaced text heuristics with image understanding.

### `convert.py`
Similar to `extract_text.py` -- structure-aware text splitter with academic signal detection, chunk type classification, exam priority assignment, weak chunk merging. Output to `chunks_ready_for_embedding.jsonl`. **Retired** for same reasons.

### `ingest_python.py`
Early ChromaDB script hardcoded for single subject (Python) with Windows paths. Had `detect_unit_query()` (regex unit detection), `chroma_query()` (metadata filtering with dummy zero-vectors for unit queries, semantic search otherwise), and interactive CLI loop. **Retired** -- replaced by generic multi-subject `ingest/` scripts.

### `query_python.py`
Early monolithic RAG chatbot: keyword scoring subject detection with LLM fallback, threshold-filtered retrieval, PYQ lookup with 0.72 similarity threshold, prompt assembly, Ollama generation, `/switch` command. **Retired** -- replaced by modular `rag/` package.

### `subject_keywords.json`
Legacy flat-format keyword map: `{"COA": [...], "PYTHON": [...]}` with noise like "unit 11", "unit none". **Retired** -- replaced by hierarchical LLM-generated map with notes/syllabus/pyq buckets and core/specific splits.

### `rag_chat.py`
Empty/blank file. Likely a placeholder whose content was already migrated elsewhere.

### `cleanup_data.py`
Deleted ChromaDB and all non-PDF files from data directory. Required `--force` flag or confirmation. **Retired** -- replaced by manual reset or pipeline's skip-existing-id behavior.

---

## Evolution Timeline

1. Google Vision OCR -> Ollama VLM -> Structured VLM JSON extraction (`extract/`)
2. Manual text chunking -> VLM-extracted structured JSON (`extract/`)
3. Single-subject hardcoded -> Multi-subject configurable (`ingest/`)
4. Monolithic chat -> Modular RAG package (`rag/`)
5. Flat manual keyword map -> Hierarchical LLM-generated map (`data/subject_keywords.json`)
