# uniAI — Syllabus-Aware, Exam-Focused RAG Assistant

uniAI is a Retrieval-Augmented Generation (RAG) study assistant tailored for university exams. It ingests official notes, PYQs, and syllabus PDFs, builds a vector database, and answers questions with a syllabus-first, exam-oriented tone.

## What This Repo Contains

- `source_code/`: ingestion, extraction, pipeline utilities, and config
- `rag_project/`: Django API for chat + endpoints
- `cli_testingScripts/`: quick local API tests
- `requirements.txt`: cross-platform Python deps
- `requirements_linux.txt`: WSL/Ubuntu guide and deps

## Core Capabilities (Implemented)

- Multimodal OCR extraction from PDFs into structured JSON chunks
- Syllabus extraction into unit-wise, CO, and books chunks
- ChromaDB vector storage with embeddings from Ollama
- Subject detection and unit filtering
- RAG chat via a Django API
- Keyword map generation for subject routing

## Architecture Snapshot

1. Extraction
- `source_code/extract_multimodal.py` for notes/PYQs
- `source_code/extract_multimodal_syllabus.py` for syllabus PDFs

2. Ingestion
- `source_code/ingest_multimodal.py` for notes/PYQs
- `source_code/ingest_multimodal_syllabus.py` for syllabus

3. Retrieval
- `source_code/pipeline/retrieval_utils.py` provides thresholded retrieval
- `source_code/pipeline/embeddings/local_embedding.py` uses Ollama embeddings

4. API
- `rag_project/rag_api/views.py` exposes `/api/query` and `/api/health`

## Data Layout

By default, data is expected under:

```
source_code/data/year_2/<SUBJECT>/
  notes/unit1/*.pdf
  notes/unit2/*.pdf
  pyq/*.pdf
  syllabus/*.pdf
```

PDFs are chunked and stored alongside their extracted JSON files.

## Configuration

Environment variables are loaded from `.env` if present. Key settings live in `source_code/config.py`.

Common env overrides:

- `OLLAMA_BASE_URL` (default `http://localhost:11434`)
- `OLLAMA_LOCAL_URL` (default `http://localhost:11434`)
- `OLLAMA_API_KEY` (optional, for Ollama cloud)
- `MODEL_EMBEDDING` (default `qwen3-embedding:4B`)
- `MODEL_VISION_BACKEND` (`ollama`, `huggingface`)
- `MODEL_VISION` (default `qwen3-vl:235b-cloud`)
- `MODEL_VISION_HF` (default `Qwen/Qwen3-VL-235B-A22B-Instruct`)
- `HF_TOKEN` (required if using HuggingFace backend)
- `MODEL_CHAT` (default `gemma3:4b`)
- `MODEL_ROUTER` (default `mistral:7b-instruct`)
- `BASE_DATA_DIR` (default `/home/anon/PROJECTS/uniAI/source_code/data/year_2`)
- `CHROMA_DB_PATH` (default `/home/anon/PROJECTS/uniAI/source_code/chroma`)
- `CHROMA_COLLECTION_NAME` (default `multimodal_notes`)

## Setup

1. Create a venv and install deps

```
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

For WSL/Ubuntu, follow the step-by-step in `requirements_linux.txt`.

2. Make sure Ollama is running locally and models are available.

## Ingestion Pipeline

1. OCR + metadata extraction for notes/PYQs

```
python source_code/extract_multimodal.py
```

2. OCR + structured extraction for syllabus PDFs

```
python source_code/extract_multimodal_syllabus.py
```

3. Ingest extracted JSONs into ChromaDB

```
python source_code/ingest_multimodal.py
python source_code/ingest_multimodal_syllabus.py
```

4. Build subject keyword map (used for routing)

```
python source_code/pipeline/generate_keyword_map.py
```

## Running the API

From `rag_project/`:

```
python manage.py runserver
```

Endpoints:

- `GET /api/health`
- `POST /api/query` with JSON body `{ "query": "...", "history": [...] }`

## Quick Test

```
python cli_testingScripts/testing.py
```

## Cleanup

This will delete the ChromaDB directory and all non-PDF files inside the data directory:

```
python source_code/cleanup_data.py --force
```

## Notes

- The system supports follow-up questions and keeps a short session history.
- Unit filters are detected via queries like "unit 4 file handling".
- Out-of-syllabus requests are routed to a generic response mode in the API.

## Known Gaps / TODOs

- CSRF is disabled for the query endpoint. Secure it before production.
- Production auth and rate limiting are not implemented.
- There are optional Gemini paths in code, but the Gemini integration is currently disabled in `config.py`.
