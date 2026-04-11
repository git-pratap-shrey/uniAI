# uniAI — Syllabus-Aware, Exam-Focused Study Assistant

> *Students don't need more explanations — they need the **right** explanations, aligned exactly with their syllabus, units, and exam patterns.*

uniAI is a **Retrieval-Augmented Generation (RAG)** system built for university students with one clear priority: **exam scoring over generic learning**. It is not a general-purpose AI tutor. Every architectural decision — from how PDFs are ingested to how the LLM prompt is structured — reflects the constraint that answers must be grounded in the student's actual syllabus, unit by unit.

---

## Why uniAI is Different

Most AI study tools try to *teach*. uniAI is designed to help students *score*.

It is intentionally less creative, more constrained, and more exam-oriented than a general assistant. Concretely, this means it answers strictly from your own uploaded notes and syllabus PDFs, it explicitly flags out-of-syllabus questions instead of silently hallucinating, retrieval is unit-scoped so asking about Unit 3 only surfaces Unit 3 content, and a cross-encoder reranker ensures the most semantically relevant chunks reach the LLM rather than just the most cosine-similar ones.

---

## Architecture Overview

```
PDF Notes / Syllabus / PYQs
        │
        ▼
┌────────────────────────────────────┐
│   VLM OCR Ingestion Pipeline       │  ← Qwen3-VL (Ollama / OpenRouter / HuggingFace)
│   Semantic sectioning per page     │    PyMuPDF, running topic list, garbage filtering
│   One JSON per topic section       │    Rate-limit safe (exponential backoff)
└──────────────┬─────────────────────┘
               │
               ▼
┌────────────────────────────────────┐
│   Three Isolated ChromaDB          │
│   Collections                      │  ← cosine similarity space
│   multimodal_notes                 │
│   multimodal_syllabus              │
│   multimodal_pyq                   │
└──────────────┬─────────────────────┘
               │
          Query arrives
               │
               ▼
┌────────────────────────────────────┐
│   Query Expansion (3 layers)       │  ← Exam phrasing normalization
│                                    │    Abbreviation expansion
│                                    │    Syllabus keyword injection
└──────────────┬─────────────────────┘
               │
               ▼
┌────────────────────────────────────┐
│   Hybrid Router (4 tiers)          │  1. Regex for explicit unit mention
│                                    │  2. Weighted keyword scoring
│                                    │  3. Pre-computed unit embedding similarity
│                                    │  4. LLM fallback (Qwen3.5 / Gemini)
└──────────────┬─────────────────────┘
               │
               ▼
┌────────────────────────────────────┐
│   Metadata-Filtered Retrieval      │  ← Subject + Unit scoped ChromaDB query
│   Notes + Syllabus chunks          │    Cosine similarity threshold gating
└──────────────┬─────────────────────┘
               │
               ▼
┌────────────────────────────────────┐
│   Cross-Encoder Reranker           │  ← Qwen3-Reranker-0.6B (HuggingFace)
│                                    │    GPU inference via PyTorch CUDA
│                                    │    Sigmoid-normalized 0–1 relevance scores
└──────────────┬─────────────────────┘
               │
               ▼
┌────────────────────────────────────┐
│   Hallucination Gate               │  ← top cross-score < 0.65 → Generic Mode
└──────────────┬─────────────────────┘
               │
               ▼
┌────────────────────────────────────┐
│   Generation                       │  ← Gemini API / Ollama / Groq
│   + Session Memory Injection       │    Exam-focused prompt assembly
└────────────────────────────────────┘
```

---

## Repository Structure

```
uniAI/
├── source_code/
│   ├── config/
│   │   ├── env.py              # Secrets and machine-specific settings from .env
│   │   ├── models.py           # AI provider profiles (Gemini, Ollama, Groq)
│   │   ├── rag.py              # RAG hyperparameters (thresholds, K values, etc.)
│   │   ├── paths.py            # Filesystem paths, ChromaDB collection names
│   │   └── main.py             # Assembles CONFIG dict — single import for everything
│   │
│   ├── models.py               # Unified provider abstraction (chat, embed, rerank, vision)
│   ├── utils.py                # Shared helpers: image encoding, ChromaDB, JSON parsing
│   ├── prompts.py              # Single source of truth for all LLM prompts
│   │
│   ├── extract/
│   │   ├── extract_multimodal_notes.py     # VLM OCR: semantic sectioning with topic loop
│   │   ├── extract_multimodal_pyq.py       # VLM OCR + LLM unit classification for PYQs
│   │   └── extract_multimodal_syllabus.py  # Structured syllabus extraction (7 chunks/PDF)
│   │
│   ├── ingest/
│   │   ├── ingest_multimodal.py            # Notes → multimodal_notes
│   │   ├── ingest_multimodal_pyq.py        # PYQs → multimodal_pyq
│   │   └── ingest_multimodal_syllabus.py   # Syllabus → multimodal_syllabus
│   │
│   ├── pipeline/
│   │   ├── embeddings/local_embedding.py   # Ollama embedding client (keep_alive)
│   │   ├── generate_keyword_map.py         # Builds subject_keywords.json for routing
│   │   ├── generate_unit_embeddings.py     # Builds unit_embeddings.pkl for Stage 3 router
│   │   └── retrieval_utils.py              # Threshold-filtered retrieval helper
│   │
│   ├── rag/
│   │   ├── rag_pipeline.py        # Main orchestrator: route → retrieve → rerank → generate
│   │   ├── hybrid_router.py       # Coordinates 4-tier routing waterfall
│   │   ├── router.py              # Tier 2: weighted keyword scoring
│   │   ├── embedding_router.py    # Tier 3: pre-computed unit embedding similarity
│   │   ├── unit_router.py         # Regex + keyword unit detection
│   │   ├── query_expander.py      # 3-layer query expansion
│   │   ├── search.py              # Collection-isolated retrieval functions
│   │   ├── cross_encoder.py       # Qwen3-Reranker-0.6B reranker (GPU)
│   │   ├── reranker.py            # Heuristic reranker (fallback / legacy)
│   │   ├── context_builder.py     # Formats chunks into LLM-ready context
│   │   └── chat_cli.py            # CLI chat loop
│   │
│   └── tests/
│       ├── test_glm.py            # Standalone GLM-OCR capability smoke test
│       ├── chat/                  # Manual chat session scripts and question sets
│       ├── retrieval/             # Retrieval accuracy and routing tests
│       ├── router/                # Router evaluation with trace logs
│       ├── complete_system/       # Full pipeline integration tests
│       ├── ci/                    # CI/CD tests (syntax, Django, pytest)
│       ├── db/                    # ChromaDB audit and dump utilities
│       ├── others/                # Miscellaneous unit tests
│       └── api/                   # API provider smoke tests
│
├── rag_project/                   # Django backend
│   └── rag_api/
│       ├── views.py               # /api/query and /api/health endpoints
│       ├── urls.py
│       └── templates/chat.html    # Minimal HTML/JS frontend
│
├── PROGRESS.md                    # Pipeline status tracker
├── .github/workflows/ci.yml       # CI: syntax check, Django health, pytest
├── requirements.txt
├── requirements_linux.txt         # WSL/Ubuntu setup guide
└── .env.example
```

---

## Core Components

### 1. Configuration System

The config was designed as a proper Python package with four files that each own one concern. `env.py` loads secrets from `.env`. `models.py` defines AI provider profiles and which one is active. `rag.py` holds every tunable hyperparameter. `paths.py` resolves filesystem locations using `pathlib`. The `main.py` assembles these into a single `CONFIG` dictionary that every other module imports, ensuring one consistent access pattern throughout the codebase.

### 2. Models Registry (`models.py`)

This is the architectural core. Instead of every script calling `ollama.chat()` or `genai.generate_content()` directly, they all go through `models.chat()`, `models.embed()`, `models.rerank()`, or `models.vision()`. Switching the generation backend from Gemini to Groq is a one-line change in `config/models.py`. Provider clients are lazily initialized — they are only created on first use, which avoids import-time failures if a provider library is not installed.

| Function | Purpose | Providers |
|---|---|---|
| `models.chat()` | Text generation | Gemini, Ollama, Groq |
| `models.embed()` | Vector embeddings | Ollama |
| `models.rerank()` | Cross-encoder scoring | HuggingFace Transformers (local) |
| `models.vision()` | VLM OCR | Ollama, OpenRouter, HuggingFace |

### 3. Ingestion Pipelines

Three parallel pipelines handle the three data types, each depositing into its own isolated ChromaDB collection.

**Notes pipeline** uses a **semantic sectioning** approach: per page, the VLM identifies distinct topic sections and returns a `sections[]` array. Each section is written as its own JSON file. A **running topic list** is maintained across pages so the VLM can reuse consistent section names rather than creating duplicates. Already-processed pages are detected by file glob and skipped, with existing topic names rehydrated from disk to preserve continuity. Images are rendered as JPEG at 1× scale for Ollama cloud (to avoid Cloudflare 524 timeouts) or PNG at 2× for HuggingFace.

**Syllabus pipeline** processes each syllabus PDF into exactly seven structured JSON files — one per unit plus course outcomes and a books/references chunk. This granularity is what makes unit-scoped retrieval precise later.

**PYQ pipeline** is the most involved. It transcribes exam papers page-by-page via VLM, then for each extracted question calls the chat LLM a second time to classify which syllabus unit the question belongs to. Questions are cleaned of marks annotations, pipe separators, and trailing numbers via regex before ingestion. Both the OCR step and the classification step use 15s × attempt exponential backoff to handle cloud rate limits.

### 4. Hybrid Query Router

Every query goes through a **four-tier** waterfall before any retrieval happens.

**Tier 1 — Regex Unit Detection** checks for an explicit unit mention (`unit 3`, `unit-4`) and extracts it immediately.

**Tier 2 — Keyword Scoring** scores the query against `subject_keywords.json` using a weighted system. PYQ keywords carry the most signal (weight 5), followed by unit-specific notes keywords (4), syllabus unit keywords (3), and core subject keywords (2). If one subject wins with no tie and meets the minimum threshold, routing completes in milliseconds without any LLM call.

| Signal | Weight |
|---|---|
| PYQ keywords | 5 |
| Notes unit-level keywords | 4 |
| Syllabus unit-level keywords | 3 |
| Core subject keywords | 2 |

**Tier 3 — Embedding Similarity** embeds the query and computes cosine similarity against pre-computed unit embeddings stored in `unit_embeddings.pkl`. These reference embeddings are generated offline from the keyword map and represent each subject/unit as a dense vector. If similarity exceeds `EMBEDDING_ROUTER_THRESHOLD` (0.55), routing is decided.

**Tier 4 — LLM Fallback** invokes a fast router model with a strict prompt that must reply with exactly one `SUBJECT_UNIT` string. Temperature is fixed at 0.0 for deterministic output. This tier only runs for genuinely ambiguous queries that escaped all previous stages.

### 5. Query Expansion

Three layers are applied before embedding to bridge the vocabulary gap between how students phrase questions and how lecture notes are written.

The first layer strips exam-style phrasing so "write a short note on buffer overflow" becomes "buffer overflow" and the embedding captures the concept, not the question format. The second layer expands known abbreviations using a hardcoded map and a loaded `subject_aliases.json`. The third layer appends syllabus keywords for the detected subject and unit, anchoring the query embedding in academic vocabulary.

### 6. Cross-Encoder Reranker

After cosine-similarity retrieval, the top candidates are reranked using `tomaarsen/Qwen3-Reranker-0.6B-seq-cls`. Unlike the bi-encoder used for initial retrieval, a cross-encoder processes the query and each document *together*, which allows it to detect semantic relationships that independent embeddings miss. Scores are sigmoid-normalized to a 0–1 range.

The **hallucination gate** sits immediately after reranking: if the top cross-encoder score falls below `MIN_CROSS_SCORE` (0.65), the pipeline discards all retrieved chunks and switches to Generic AI Tutor Mode. This is the mechanism that prevents the LLM from producing confident-sounding answers from irrelevant context.

### 7. Three Isolated ChromaDB Collections

| Collection | Content | Key Metadata |
|---|---|---|
| `multimodal_notes` | Lecture notes, handwritten notes, slides | `subject`, `unit`, `title`, `chunk_idx`, `section_index`, `confidence` |
| `multimodal_syllabus` | Unit topics, course outcomes, book lists | `subject`, `unit`, `chunk_type`, `syllabus_version` |
| `multimodal_pyq` | Past year exam questions | `subject`, `unit`, `year`, `marks` |

Collection isolation is foundational. The `retrieve_notes()` function applies an explicit `document_type != "syllabus"` filter to prevent syllabus chunks from appearing in notes results, even though both live under the same ChromaDB path.

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

For WSL/Ubuntu, follow the step-by-step guide in `requirements_linux.txt` (includes PyTorch CUDA setup).

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your keys and paths
```

Key variables to set:

```env
OLLAMA_BASE_URL=http://localhost:11434   # or cloud Ollama URL
OLLAMA_API_KEY=...                       # if using authenticated cloud Ollama
BASE_DATA_DIR=/path/to/your/data        # flattened: SUBJECT/notes/unitN/*.pdf
CHROMA_DB_PATH=/path/to/your/chroma
GEMINI_API_KEY=...                       # if using Gemini for generation
OPENROUTER_API_KEY=...                   # if using OpenRouter for vision fallback
HF_TOKEN=...                             # if using HuggingFace for vision or reranking
USE_OLLAMA_CLOUD=true                    # true = use OLLAMA_BASE_URL, false = OLLAMA_LOCAL_URL
```

Make sure Ollama is running locally (`ollama serve`) and required models are pulled.

### 3. Place your data

Data layout is **flat** — no year nesting:

```
<BASE_DATA_DIR>/<SUBJECT>/
  notes/unit1/*.pdf
  notes/unit2/*.pdf
  pyqs/*.pdf
  syllabus/*.pdf
```

### 4. Run the ingestion pipeline

```bash
# OCR extraction (run as modules from project root)
python -m source_code.extract.extract_multimodal_notes
python -m source_code.extract.extract_multimodal_pyq
python -m source_code.extract.extract_multimodal_syllabus

# Ingest into ChromaDB
python source_code/ingest/ingest_multimodal.py
python source_code/ingest/ingest_multimodal_pyq.py
python source_code/ingest/ingest_multimodal_syllabus.py

# Build router artifacts
python source_code/pipeline/generate_keyword_map.py
python source_code/pipeline/generate_unit_embeddings.py
```

All extraction scripts are resumable — already-processed files are detected and skipped automatically.

### 5. Start the server

```bash
cd rag_project
python manage.py runserver
```

**API Endpoints:**

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | System health and active model |
| `POST` | `/api/query` | Main RAG query endpoint |

**Query payload:**
```json
{
  "query": "Explain buffer overflow attack",
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

## Configuration Reference

All tuneable parameters live in `source_code/config/rag.py`.

| Parameter | Default | Description |
|---|---|---|
| `similarity_threshold` | `0.35` | Min cosine similarity to keep a retrieval result |
| `min_strong_sim` | `0.6` | Min similarity the top chunk must have |
| `cross_encoder.model` | `tomaarsen/Qwen3-Reranker-0.6B-seq-cls` | Reranker model |
| `cross_encoder.min_score` | `0.65` | Below this score → Generic AI Tutor Mode |
| `cross_encoder.candidates` | `6` | Max chunks sent to cross-encoder |
| `cross_encoder.pipeline_top_n` | `4` | Chunks kept after reranking |
| `history_limit` | `4` | Conversation turns injected into context |
| `keywords.min_score` | `2` | Min keyword score to trust Tier 2 routing |
| `embedding_router_threshold` | `0.55` | Min similarity to trust Tier 3 routing |

Chat model selection lives in `source_code/config/models.py` via `ACTIVE_CHAT_MODEL`.

---

## Tech Stack

| Layer | Stack |
|---|---|
| Backend | Python, Django |
| AI / ML | RAG, VLM OCR, Cross-encoder reranking, Embeddings |
| Models | Qwen3-VL, Qwen3-Reranker-0.6B, Qwen3-Embedding:4B, Qwen3.5:2b, Gemini API |
| Vector DB | ChromaDB (3 isolated collections, cosine space) |
| Inference | Ollama (local/cloud), OpenRouter, HuggingFace Transformers, PyTorch CUDA |
| Data Processing | PyMuPDF, semantic sectioning, custom cleaning |
| Testing | pytest, GitHub Actions CI |
| Dev & Infra | Git/GitHub, `.env` config, Cloudflare Tunnel, local-first design |

---

## Current Limitations

The cross-encoder loads on first call and blocks until it is warm, meaning the first request after a cold server start will be noticeably slow. CSRF is currently disabled on `/api/query` for development convenience and must be re-enabled before any public deployment. Only one academic year is fully ingested in the current prototype. There is no persistent long-term memory across sessions — conversation history is stateless and lives in the frontend.

## Roadmap

Answer citations with source page references so students can trace answers back to their notes. A background warm-up thread for the cross-encoder to eliminate cold-start latency. Automated ingestion triggers for new subject data. Unit-level summaries and topic index generation. Fix zero-yield PYQ PDFs (fill-in-the-blank regex). College-wide deployment once the system is hardened.

---

## Status

**Stage:** Active development / prototype — notes extraction running  
**Target users:** Self + small group of classmates  
**Future goal:** College-wide deployment

The focus of uniAI is not novelty — it is **alignment with real academic needs** and **practical engineering trade-offs**. Every component exists because a simpler version failed a real retrieval or accuracy problem.
