# GEMINI.md - uniAI Project Instructions

This document provides foundational guidance for Gemini CLI when working in the uniAI repository. It supplements the project's existing `CLAUDE.md` and `README.md`.

## Project Overview
**uniAI** is a syllabus-aware, exam-focused Retrieval-Augmented Generation (RAG) system. Its primary goal is to provide students with answers strictly grounded in their specific syllabus and notes, prioritizing exam relevance over generic tutoring.

### Key Technologies
- **Backend:** Django 4.2+
- **Vector Database:** ChromaDB (3 isolated collections: notes, syllabus, pyq)
- **AI Models:**
  - **Generation:** Gemini API, Ollama, Groq (abstracted via `source_code/models.py`)
  - **Vision/OCR:** Qwen3-VL (via Ollama/OpenRouter/HuggingFace)
  - **Reranking:** Qwen3-Reranker-0.6B (Local HF Transformers)
  - **Embeddings:** Qwen3-Embedding:4B (Ollama)
- **Data Processing:** PyMuPDF (PDF parsing), Pillow (Image handling)

---

## Architectural Principles

### 1. Centralized Configuration
All configurations are managed in `source_code/config/` and assembled into a single `CONFIG` dictionary in `source_code/config/main.py`.
- **`env.py`**: Secrets and environment variables.
- **`models.py`**: Provider profiles and active model selection.
- **`rag.py`**: Hyperparameters (thresholds, K-values, etc.).
- **`paths.py`**: Filesystem paths and collection names.

**Constraint:** Always import `CONFIG` from `source_code.config.main`.

### 2. Unified Model Registry (`source_code/models.py`)
Direct calls to provider SDKs (Ollama, Gemini, etc.) are forbidden outside of `models.py`. Use the following abstractions:
- `models.chat()`: Text generation.
- `models.embed()`: Vector embeddings.
- `models.rerank()`: Cross-encoder scoring.
- `models.vision()`: VLM OCR tasks.

### 3. Collection Isolation
ChromaDB is split into three distinct collections to prevent cross-contamination:
- `multimodal_notes`: Lecture/handwritten notes.
- `multimodal_syllabus`: Structured unit topics and outcomes.
- `multimodal_pyq`: Past year questions with unit classification.

### 4. Hybrid Routing Waterfall
Queries are routed through four tiers:
1. **Regex**: Explicit unit mentions (e.g., "unit 3").
2. **Keyword Scoring**: Weighted matches against `subject_keywords.json`.
3. **Embedding Similarity**: Vector match against `unit_embeddings.pkl`.
4. **LLM Fallback**: deterministic LLM call for ambiguous queries.

---

## Development Workflow

### **Rule: Read Functionality Files**
Before modifying any code in `source_code/`, you **MUST** read the corresponding `functionality.md` file in that directory.
- `source_code/rag/functionality.md`
- `source_code/tests/functionality.md`
- `source_code/config/functionality.md`
- `source_code/functionality_root_files.md` (for root `models.py`, `prompts.py`, `utils.py`)

### Building and Running

#### Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Configure keys/paths
```

#### Ingestion Pipeline (Sequential)
1. **Extract**: `python -m source_code.extract.extract_multimodal_notes` (Repeat for `pyq` and `syllabus`)
2. **Ingest**: `python source_code/ingest/ingest_multimodal.py` (Repeat for `pyq` and `syllabus`)
3. **Artifacts**:
   - `python source_code/pipeline/generate_keyword_map.py`
   - `python source_code/pipeline/generate_unit_embeddings.py`

#### Running the Application
- **Django Server**: `cd rag_project && python manage.py runserver`
- **CLI Chat**: `python source_code/rag/chat_cli.py`

#### Testing
- **All Tests**: `pytest source_code/tests/`
- **Router Eval**: `python source_code/tests/router/run_router_tests.py`
- **System Eval**: `python source_code/tests/complete_system/run_test.py`

---

## Coding Conventions
- **Typing**: Use Python type hints for all function signatures.
- **Prompts**: Keep all LLM prompts in `source_code/prompts.py`.
- **Error Handling**: Use the `⚠` prefix for AI provider errors as per `models.py` convention.
- **Lazy Loading**: AI clients in `models.py` are lazy-loaded; maintain this pattern to avoid unnecessary initialization overhead.
- **Reranker**: The cross-encoder requires GPU (CUDA) for acceptable performance; ensure fallback to CPU is handled gracefully (implemented in `models.py`).
