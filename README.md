# uniAI — Syllabus-Aware, Exam-Focused Study Assistant

uniAI is a **Retrieval-Augmented Generation (RAG)** system built for university students with one clear priority: **exam scoring over generic learning**.

It is not a general-purpose AI tutor. It is built around syllabus boundaries, unit-level depth, and the kind of structured answers students are expected to write in exams. Every architectural decision reflects that constraint.

> *Students don't need more explanations — they need the **right** explanations, aligned exactly with their syllabus, units, and exam patterns.*

---

## What Makes uniAI Different

Most AI study tools try to *teach*. uniAI is designed to help students *score*.

- Answers are grounded strictly in **your own notes and syllabus PDFs**
- Explicitly flags questions that are **out of syllabus** instead of silently hallucinating
- Retrieval is **unit-scoped** — asking about Unit 3 only pulls Unit 3 content
- A **cross-encoder reranker** ensures the most relevant chunks reach the LLM, not just the most similar ones
- Exam tone: definitions first, keywords bolded, structured points

---

## Architecture Overview

```
PDF Notes / Syllabus / PYQs
        │
        ▼
┌─────────────────────────────────┐
│   VLM OCR Ingestion Pipeline    │  ← Qwen3-VL (Ollama / HuggingFace)
│   Parallel page-by-page OCR     │    PyMuPDF, metadata tagging, garbage filtering
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│   Three Isolated Collections    │
│   multimodal_notes              │
│   multimodal_syllabus           │  ← ChromaDB (cosine space)
│   multimodal_pyq                │
└────────────┬────────────────────┘
             │
        Query comes in
             │
             ▼
┌─────────────────────────────────┐
│   Query Expansion (3 layers)    │  ← Exam phrasing normalization
│                                 │    Abbreviation expansion
│                                 │    Syllabus keyword injection
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│   Hybrid Router (3 stages)      │  1. Keyword scoring
│                                 │  2. Unit embedding similarity
│                                 │  3. LLM fallback (Mistral)
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│   Metadata-Filtered Retrieval   │  ← Subject + Unit scoped ChromaDB query
│   Notes + Syllabus chunks       │    Cosine similarity threshold gating
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│   Cross-Encoder Reranker        │  ← Qwen3-Reranker-0.6B (HuggingFace)
│                                 │    AutoModelForSequenceClassification
│                                 │    GPU inference via PyTorch CUDA
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│   Hallucination Gate            │  ← If top cross-score < threshold → Generic Mode
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│   Generation                    │  ← Gemini API (cloud) or Ollama (local)
│   + Session Memory Injection    │    Exam-focused prompt assembly
└─────────────────────────────────┘
```

---

## Repository Structure

```
uniAI/
├── source_code/
│   ├── config.py                        # All model, path, and threshold config
│   ├── prompts.py                       # Single source of truth for all LLM prompts
│   ├── utils.py                         # Shared helpers: embeddings, ChromaDB, image encoding
│   │
│   ├── extract/
│   │   ├── extract_multimodal_notes.py  # VLM OCR ingestion for lecture notes
│   │   ├── extract_multimodal_pyq.py    # VLM OCR + LLM unit classification for PYQs
│   │   └── extract_multimodal_syllabus.py # Structured syllabus extraction (7 chunks/PDF)
│   │
│   ├── ingest/
│   │   ├── ingest_multimodal.py         # Notes → multimodal_notes
│   │   ├── ingest_multimodal_pyq.py     # PYQs → multimodal_pyq
│   │   └── ingest_multimodal_syllabus.py # Syllabus → multimodal_syllabus
│   │
│   ├── pipeline/
│   │   ├── embeddings/local_embedding.py  # Ollama embedding client (keep_alive)
│   │   ├── generate_keyword_map.py        # Builds subject_keywords.json for routing
│   │   ├── generate_unit_embeddings.py    # Builds unit_embeddings.pkl for embedding router
│   │   └── retrieval_utils.py             # Threshold-filtered retrieval helper
│   │
│   ├── rag/
│   │   ├── rag_pipeline.py              # Main orchestrator: routes → retrieves → reranks → generates
│   │   ├── hybrid_router.py             # 3-stage router: keyword → embedding → LLM
│   │   ├── router.py                    # Keyword scoring router with weighted signals
│   │   ├── embedding_router.py          # Pre-computed unit embedding similarity router
│   │   ├── unit_router.py               # Regex + keyword unit detection
│   │   ├── query_expander.py            # 3-layer query expansion
│   │   ├── search.py                    # Collection-isolated retrieval functions
│   │   ├── cross_encoder.py             # Qwen3-Reranker-0.6B reranker (GPU)
│   │   ├── reranker.py                  # Heuristic reranker (fallback / legacy)
│   │   ├── context_builder.py           # Formats chunks into LLM-ready context
│   │   └── chat_cli.py                  # CLI chat loop
│   │
│   └── tests/
│       ├── ci/                          # GitHub Actions test suite
│       ├── router/                      # Hybrid router accuracy evaluation (30 questions)
│       ├── retrieval/                   # Collection isolation + pipeline tests
│       └── chat/                        # End-to-end RAG sweep tests
│
├── rag_project/                         # Django backend
│   └── rag_api/
│       ├── views.py                     # /api/query and /api/health endpoints
│       ├── urls.py
│       └── templates/chat.html          # Minimal HTML/JS frontend
│
├── .github/workflows/ci.yml            # CI: syntax check, Django health, pytest
├── requirements.txt
├── requirements_linux.txt              # WSL/Ubuntu setup guide
└── .env.example
```

---

## Core Components

### 1. Ingestion Pipeline

PDFs go through a three-stage ingestion process before they reach ChromaDB.

**Notes & Syllabus OCR** (`extract_multimodal_notes.py`, `extract_multimodal_syllabus.py`)
- Renders each PDF page to an image (JPEG for Ollama cloud, PNG for HuggingFace)
- Sends page images to a Vision-Language Model (VLM) for OCR and structured extraction
- Supports two backends switchable via `MODEL_VISION_BACKEND`:
  - `ollama` — local or cloud Ollama (default: `qwen3-vl:235b-cloud`)
  - `huggingface` — HuggingFace Inference API (default: `Qwen/Qwen3-VL-235B-A22B-Instruct`)
- Extracts structured JSON per chunk: `full_text`, `title`, `topics`, `key_concepts`, `unit`, `confidence`
- Runs **parallel page extraction** for fast ingestion
- Syllabus PDFs produce exactly 7 structured chunks: `unit_1` through `unit_5`, `course_outcomes`, `books_references`

**PYQ Pipeline** (`extract_multimodal_pyq.py`)
- VLM transcribes exam question paper images page by page
- Each extracted question is classified into its syllabus unit using an LLM (`MODEL_CHAT`)
- Questions are structured with: `question_text`, `unit`, `marks`, `year`, `section`
- Stored in the dedicated `multimodal_pyq` collection for exam-pattern retrieval

**Ingestion Filtering** (`ingest_multimodal.py`)
- Skips chunks below OCR confidence threshold (`MIN_INGEST_CONFIDENCE = 0.3`)
- Filters promotional/garbage content (title blocklist + keyword density check)
- Normalizes unit strings to plain numeric format (`"unit1"` → `"1"`)

---

### 2. Three Isolated ChromaDB Collections

| Collection | Content | Key Metadata Fields |
|---|---|---|
| `multimodal_notes` | Lecture notes, handwritten notes, printed slides | `subject`, `unit`, `title`, `document_type`, `confidence` |
| `multimodal_syllabus` | Unit topics, course outcomes, reference books | `subject`, `unit`, `chunk_type`, `syllabus_version` |
| `multimodal_pyq` | Past year exam questions | `subject`, `unit`, `year`, `marks` |

Collection isolation is the foundation of specialized retrieval. Notes, syllabus, and PYQ data are never mixed at query time.

---

### 3. Hybrid Query Router

Before retrieval, every query goes through a three-stage routing pipeline to detect subject and unit.

**Stage 1 — Keyword Scoring** (`router.py`)

Scores the query against `subject_keywords.json` using weighted signals:

| Signal | Weight |
|---|---|
| PYQ keywords | 5 |
| Notes unit-level keywords | 4 |
| Syllabus unit-level keywords | 3 |
| Notes/Syllabus core keywords | 2 |
| Unknown/noise | 0 |

Returns subject + unit if score ≥ `KEYWORD_MIN_SCORE` with no tie.

**Stage 2 — Embedding Similarity** (`embedding_router.py`)

If keyword scoring fails or is ambiguous, the query is embedded and compared against pre-computed unit embeddings (`unit_embeddings.pkl`). Returns subject + unit if similarity ≥ `EMBEDDING_ROUTER_THRESHOLD`.

**Stage 3 — LLM Fallback** (`hybrid_router.py`)

For complex or ambiguous queries, a fast local router model (Mistral) classifies the query against the full `subject_unit` list. Temperature is set to 0 for deterministic output.

---

### 4. Query Expansion (`query_expander.py`)

Three layers applied before embedding, to bridge the gap between student phrasing and syllabus terminology:

1. **Exam phrasing normalization** — strips question-format tokens (`define`, `explain`, `write a note on`) so embeddings focus on the actual concept
2. **Abbreviation expansion** — maps known abbreviations (`k-map` → `karnaugh map`, `cia triad` → `confidentiality integrity availability`) plus subject aliases from `subject_aliases.json`
3. **Syllabus keyword injection** — appends unit-specific and core keywords from `subject_keywords.json` to anchor the embedding in syllabus vocabulary

---

### 5. Cross-Encoder Reranker (`cross_encoder.py`)

After initial cosine-similarity retrieval, chunks are reranked using a sequence classification model.

- **Model:** `tomaarsen/Qwen3-Reranker-0.6B-seq-cls`
- **Loaded via:** HuggingFace `AutoModelForSequenceClassification`
- **Inference:** GPU (CUDA) with `torch.float16`, eager-loaded at startup
- **Process:** Top `CROSS_ENCODER_CANDIDATES` (default: 6) chunks are scored as `(query, chunk)` pairs; scores are sigmoid-normalized to 0–1
- **Hallucination gate:** If the top cross-encoder score falls below `MIN_CROSS_SCORE` (default: 0.65), the pipeline switches to **Generic AI Tutor Mode** and returns no chunks

---

### 6. Generation

- **Primary:** Gemini API (`gemini-3-flash-preview` or configurable)
- **Local fallback:** Ollama with `gemma3:4b` or any configured `MODEL_CHAT`
- **Session memory:** Last `MAX_HISTORY_TURNS` (default: 4) conversation turns injected at generation time
- **Prompt mode:** `syllabus` (strict, exam-focused) or `generic` (labeled general knowledge)
- **Follow-up detection:** Queries like `repeat`, `summarize`, `again` skip retrieval and use history directly

---

## Setup & Installation

### 1. Clone and create environment

```bash
git clone https://github.com/git-pratap-shrey/uniAI.git
cd uniAI
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

*For WSL/Ubuntu, follow the step-by-step guide in `requirements_linux.txt` (includes PyTorch CUDA setup).*

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your keys and paths
```

Key variables:

```env
MODEL_VISION_BACKEND=ollama          # ollama | huggingface
MODEL_VISION=qwen3-vl:235b-cloud
MODEL_EMBEDDING=qwen3-embedding:4B
MODEL_CHAT=gemma3:4b                 # or gemini-3-flash-preview
MODEL_ROUTER=mistral:7b-instruct
GEMINI_API_KEY=...                   # if using Gemini
HF_TOKEN=...                         # if using HuggingFace backend
```

Make sure Ollama is running locally (`ollama serve`) and required models are pulled.

### 3. Place your data

```
source_code/data/year_2/<SUBJECT>/
  notes/unit1/*.pdf
  notes/unit2/*.pdf
  pyqs/*.pdf
  syllabus/*.pdf
```

### 4. Run ingestion pipeline

```bash
# Step 1: OCR extraction (notes and PYQs run in parallel)
python source_code/extract/extract_multimodal_notes.py
python source_code/extract/extract_multimodal_pyq.py

# Step 2: Syllabus extraction
python source_code/extract/extract_multimodal_syllabus.py

# Step 3: Ingest into ChromaDB
python source_code/ingest/ingest_multimodal.py
python source_code/ingest/ingest_multimodal_pyq.py
python source_code/ingest/ingest_multimodal_syllabus.py

# Step 4: Build keyword map and unit embeddings for router
python source_code/pipeline/generate_keyword_map.py
python source_code/pipeline/generate_unit_embeddings.py
```

### 5. Start the server

```bash
cd rag_project
python manage.py runserver
```

**Endpoints:**

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | System health + active model |
| `POST` | `/api/query` | Main RAG query endpoint |

**Query payload:**
```json
{
  "query": "Explain buffer overflow",
  "history": [],
  "subject": "CYBER_SECURITY"
}
```

### 6. CLI chat (optional)

```bash
python source_code/rag/chat_cli.py
```

Commands: `/switch <SUBJECT>`, `/subjects`, `/history`, `/clear`

---

## Testing

```bash
# CI suite (syntax, Django health, router unit test)
pytest source_code/tests/ci/ -v

# Router accuracy (30 questions across 3 subjects)
python source_code/tests/router/test_30_questions.py

# Retrieval isolation tests
pytest source_code/tests/retrieval/ -v

# Full pipeline evaluation (80 questions, JSONL output)
python source_code/tests/chat/testing.py

# Cross-encoder threshold sweep
python source_code/tests/chat/sweep.py
```

CI runs automatically on every push via GitHub Actions (`.github/workflows/ci.yml`).

---

## Configuration Reference

All tuneable parameters live in `source_code/config.py`.

| Parameter | Default | Description |
|---|---|---|
| `SIMILARITY_THRESHOLD` | `0.35` | Min cosine similarity to keep a retrieval result |
| `MIN_STRONG_SIM` | `0.6` | Min similarity the top chunk must have |
| `CROSS_ENCODER_MODEL` | `tomaarsen/Qwen3-Reranker-0.6B-seq-cls` | Reranker model |
| `MIN_CROSS_SCORE` | `0.65` | Below this → Generic AI Tutor Mode |
| `CROSS_ENCODER_CANDIDATES` | `6` | Max pairs sent to cross-encoder |
| `PIPELINE_CROSS_RERANK_TOP_N` | `4` | Chunks kept after reranking |
| `MAX_HISTORY_TURNS` | `4` | Conversation turns injected into context |
| `KEYWORD_MIN_SCORE` | `2` | Min keyword score for router Stage 1 |
| `EMBEDDING_ROUTER_THRESHOLD` | `0.55` | Min similarity for router Stage 2 |

---

## Cleanup

```bash
# Delete ChromaDB + all non-PDF files in data directory
python source_code/not_in_use_scripts/cleanup_data.py --force
```

---

## Current Limitations

- Fixed-size chunking (semantic/structure-aware chunking planned)
- CSRF disabled on `/api/query` — must be secured before any deployment
- No production auth or rate limiting
- Cloud LLM rate limits during heavy ingestion
- Cross-encoder loads eagerly at startup — requires GPU for reasonable speed

## Roadmap

- Semantic chunking
- Answer citations with source page references
- Confidence indicators in responses
- Local generation fallback to reduce cloud cost
- Unit-level summaries and topic index
- College-wide deployment (subject to feasibility)

---

## Tech Stack

| Layer | Stack |
|---|---|
| Backend | Python, Django, FastAPI (testing) |
| AI / ML | RAG, VLM OCR, Cross-encoder reranking, Embeddings |
| Models | Qwen3-VL, Qwen3-Reranker, Gemma3, Mistral, Gemini API |
| Vector DB | ChromaDB (3 isolated collections) |
| Inference | Ollama (local/cloud), HuggingFace Transformers, PyTorch CUDA |
| Data Processing | PyMuPDF, custom chunking and cleaning |
| Testing | pytest, GitHub Actions CI |
| Dev & Infra | Git/GitHub, `.env` config, Cloudflare Tunnel, local-first design |

---

## Status

**Stage:** Active development / prototype  
**Target users:** Self + small group of classmates  
**Future goal:** College-wide deployment

The focus of uniAI is not novelty — it is **alignment with real academic needs** and **practical engineering trade-offs**. Every component exists because a simpler version failed a real retrieval or accuracy problem.
