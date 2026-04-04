# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Working Rules

### Always Read Functionality Files Before Code Changes

**Rule:** Before planning or making any code changes in a directory inside `source_code/`, you MUST first read the `functionality.md` file in that directory (or the root-level file for cross-directory changes).

- If modifying files in `source_code/rag/` → read `source_code/rag/functionality.md`
- If modifying files in `source_code/tests/` → read `source_code/tests/functionality.md`
- If modifying files in `source_code/config/` → read `source_code/config/functionality.md`
- For root-level files in `source_code/` → read `source_code/functionality_root_files.md`
- For any other subdirectory → read its `functionality.md` if one exists

This ensures you understand the module's purpose, function signatures, inputs/outputs, inter-file relationships, and data flow before touching any code.

## Project Overview

uniAI is a syllabus-aware, exam-focused RAG (Retrieval-Augmented Generation) system for university students. It ingests PDF notes, syllabus documents, and past-year exam papers (PYQs) via VLM OCR, stores them in three isolated ChromaDB collections, and answers queries through a multi-stage routing pipeline with cross-encoder reranking and hallucination gating.

## Development Commands

### Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

For WSL/Ubuntu with GPU support, follow `requirements_linux.txt`.

### Environment

```bash
cp .env.example .env   # then edit with API keys, paths, model settings
```

### Ingestion Pipeline

```bash
# Extract (VLM OCR)
python source_code/extract/extract_multimodal_notes.py
python source_code/extract/extract_multimodal_syllabus.py
python source_code/extract/extract_multimodal_pyq.py

# Ingest into ChromaDB
python source_code/ingest/ingest_multimodal.py
python source_code/ingest/ingest_multimodal_syllabus.py
python source_code/ingest/ingest_multimodal_pyq.py

# Build router artifacts
python source_code/pipeline/generate_keyword_map.py
python source_code/pipeline/generate_unit_embeddings.py
```

### Start Server (Django)

```bash
cd rag_project
python manage.py runserver
```

API: `GET /api/health`, `POST /api/query`

### CLI Chat

```bash
python source_code/rag/chat_cli.py
```

Commands: `/switch <SUBJECT>`, `/subjects`, `/history`, `/clear`

### Testing

```bash
# Run all pytest tests
pytest source_code/tests/

# Run individual test files
pytest source_code/tests/ci/test_router.py
pytest source_code/tests/ci/test_chat.py
pytest source_code/tests/ci/test_db.py

# Router-specific tests
python source_code/tests/router/run_router_tests.py

# Complete system tests
python source_code/tests/complete_system/run_test.py
```

## Architecture

### High-Level Flow

```
User query → Query Expansion → 3-Stage Hybrid Router → Collection-Isolated Retrieval → Cross-Encoder Rerank → Hallucination Gate → LLM Generation
```

### Core Modules

- **`source_code/config/`** — Unified configuration: `env.py` (secrets), `models.py` (provider profiles), `rag.py` (hyperparameters), `paths.py` (filesystem), `main.py` (assembles `CONFIG` dict)
- **`source_code/models.py`** — Unified model registry: `chat()`, `embed()`, `rerank()`, `vision()` abstract away Gemini/Ollama/Groq/HF backends
- **`source_code/prompts.py`** — All LLM prompts in one place
- **`source_code/rag/rag_pipeline.py`** — Main orchestrator: the `answer_query()` function is the single entry point for the RAG pipeline
- **`source_code/rag/hybrid_router.py`** — Coordinates 3-stage routing waterfall (keyword → embedding → LLM fallback)
- **`source_code/rag/search.py`** — Collection-isolated retrieval (`retrieve_notes`, `retrieve_syllabus`)
- **`source_code/rag/cross_encoder.py`** — Qwen3-Reranker-0.6B GPU reranker with hallucination gate (score < 0.65 → generic mode)

### Three Isolated ChromaDB Collections

| Collection | Content | Key Metadata |
|---|---|---|
| `multimodal_notes` | Lecture/handwritten notes, slides | `subject`, `unit`, `title`, `document_type`, `confidence` |
| `multimodal_syllabus` | Unit topics, course outcomes, references | `subject`, `unit`, `chunk_type`, `syllabus_version` |
| `multimodal_pyq` | Past year exam questions | `subject`, `unit`, `year`, `marks` |

### Django Backend

Located in `rag_project/rag_api/`. `views.py` has `query_view` (POST `/api/query`) and `health_view` (GET `/api/health`). CSRF is currently disabled on the query endpoint for development.

### Key Configurable Parameters (in `source_code/config/rag.py`)

| Parameter | Default | Description |
|---|---|---|
| `similarity_threshold` | `0.35` | Min cosine similarity for retrieval |
| `cross_encoder.min_score` | `0.65` | Below this → Generic AI Tutor Mode |
| `cross_encoder.candidates` | `6` | Max chunks sent to cross-encoder |
| `history_limit` | `4` | Conversation turns in context |
| `embedding_router_threshold` | `0.55` | Min similarity for Stage 2 routing |

### Data Layout

```
source_code/data/year_2/<SUBJECT>/
  notes/unit1/*.pdf, notes/unit2/*.pdf, ...
  pyqs/*.pdf
  syllabus/*.pdf
```
