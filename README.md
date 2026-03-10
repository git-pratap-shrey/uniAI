# uniAI – Syllabus-Aware, Exam-Focused Study Assistant

uniAI is a **Retrieval-Augmented Generation (RAG)** based study assistant designed specifically for **university students**, with a clear priority: **exam scoring over generic learning**.

Unlike general-purpose AI tutors, uniAI is built around syllabus boundaries, unit-wise depth, and the kind of answers students are actually expected to write in exams.

---

## 🎯 Project Vision

> *Students don’t need more explanations — they need the **right explanations**, aligned exactly with their **syllabus**, **units**, and **exam patterns***.

### Core Goals

* Be **syllabus-aware**, not just topic-aware
* Answer in a **“what to write in exam”** tone
* Prioritize **definitions, keywords, and structured points**
* Respect **unit-level boundaries**
* Clearly distinguish between:
  * **In-syllabus answers** (strict, grounded, no inference)
  * **Out-of-syllabus questions** (explicitly labeled as *Generic AI Tutor Mode*)

---

## What This Repo Contains

- `source_code/`: ingestion, extraction, pipeline utilities, and config
- `rag_project/`: Django API for chat + endpoints
- `cli_testingScripts/`: quick local API tests
- `tests/router/`: test scripts to evaluate subject routing accuracy
- `requirements.txt`: cross-platform Python deps
- `requirements_linux.txt`: WSL/Ubuntu guide and deps

---

## 🧠 What uniAI Can Do (Current Features)

* Answer questions **only from official notes / PDFs** in syllabus mode
* Detect and handle:
  * Unit-specific queries (e.g. *"unit 4 file handling"*)
  * Follow-up questions (e.g. *"repeat"*, *"summarize"*)
  * Out-of-syllabus intent (e.g. *"write code"*, *"implementation"*)
* Support **multi-turn conversational context** (session-scoped)
* List **unit-wise topics** and explain **individual concepts**
* Prevent silent hallucination by:
  * Explicitly saying *"not found in syllabus"*
  * Switching to **Generic AI Tutor Mode** when required

---

## 🧱 Architecture Overview

### 1️⃣ Data Layer & Layout

* Source: University PDFs (notes, PYQs, syllabus)
* By default, data is expected under:

```text
source_code/data/year_2/<SUBJECT>/
  notes/unit1/*.pdf
  notes/unit2/*.pdf
  pyq/*.pdf
  syllabus/*.pdf
```
PDFs are chunked and stored alongside their extracted JSON files.

### 2️⃣ Ingestion Pipeline

* PDF → Text extraction (Vision language model - OCR) running in parallel, page by page.
* Text cleaning and normalization
* Fixed-size chunking
* Metadata tagging:
  * `unit`, `category` (notes / pyq), `source`
* Embedding generation (local)
* Storage in a persistent vector database

### 3️⃣ RAG & Retrieval

* **ChromaDB** used as the vector store
* Supports two retrieval strategies:
  * **Semantic retrieval** for concept-level queries
  * **Metadata-only retrieval** for unit overviews (topics listing)
* Correct handling of Chroma filter constraints (`$and` logic)

### 4️⃣ Query Pipeline & Routing

* **Query Expansion**: Expands queries with subject aliases and syllabus keywords to improve recall.
* **Hybrid Router**: Routes queries to the correct subject/unit using a combined approach:
  * Keyword matching (fast, exact)
  * Embedding similarity (semantic match)
  * LLM Fallback (complex queries)
* **Generation (Gemini / Local)**: Text generation with strict exam-focused prompting.
* **Session Memory**: Short-term, isolated conversational memory injected at generation time.

### 5️⃣ API & UI

* Backend: **Django**
* API endpoints:
  * `/api/query`
  * `/api/health`
* Frontend: Minimal HTML + CSS + Vanilla JS (`marked.js` for markdown rendering)
* Stateless backend (chat history sent from frontend)

---

## ⚙️ Configuration

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

---

## 🚀 Setup & Execution

### 1. Setup Environment
Create a venv and install deps:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
*For WSL/Ubuntu, follow the step-by-step in `requirements_linux.txt`.*
Make sure Ollama is running locally and models are available.

### 2. Ingestion Pipeline
1. **OCR + metadata extraction for notes/PYQs (can run in parallel):**
```bash
python source_code/extract_multimodal.py
```
2. **OCR + structured extraction for syllabus PDFs:**
```bash
python source_code/extract_multimodal_syllabus.py
```
3. **Ingest extracted JSONs into ChromaDB:**
```bash
python source_code/ingest_multimodal.py
python source_code/ingest_multimodal_syllabus.py
```
4. **Build subject keyword map (used for routing):**
```bash
python source_code/pipeline/generate_keyword_map.py
```

### 3. Testing & Validation
1. **Run the Router Tests to validate accuracy:**
```bash
python tests/router/test_30_questions.py
```
2. **Run local API testing scripts:**
```bash
python cli_testingScripts/testing.py
```

### 4. Running the API
From `rag_project/`:
```bash
python manage.py runserver
```
**Endpoints:**
- `GET /api/health`
- `POST /api/query` with JSON body `{ "query": "...", "history": [...] }`

### 5. Cleanup
This will delete the ChromaDB directory and all non-PDF files inside the data directory:
```bash
python source_code/cleanup_data.py --force
```

---

## ✅ What Has Been Implemented

### ✔ Core Process
End-to-end pipeline is functional:
`PDF → Parallel Page Extraction → Text/Images → Emb → ChromaDB → Query Expansion & Hybrid Router → Answer`

### ✔ Practical Engineering Work
* Correct handling of ChromaDB filter semantics, LLM/Vision API rate limits, null/error responses
* Environment-based configuration (`config.py`)
* Fully synced Django UI and CLI testing utilities
* Robust test suite for the Hybrid Router (`tests/router/`)
* Parallel multimodal extraction scripts for fast ingestion
* Cloudflare Tunnel support for sharing the dev server

---

## 🛠️ Tech Stack

* **Backend:** Python, Django, ChromaDB
* **AI / ML:** Retrieval-Augmented Generation (RAG), Local embeddings via **Ollama**, **Gemini API** for cloud generation
* **Data Processing:** PyMuPDF, custom chunking/cleaning
* **Dev & Infra:** `.env` configs, Git/GitHub, Cloudflare Tunnel, Local-first design

---

## 🚧 Current Limitations & Roadmap

### Limitations
* Fixed-size chunking (semantic chunking planned)
* Cloud LLM rate limits during heavy testing
* CSRF is disabled for the Django query endpoint (requires securing before production).
* Production auth and rate limiting are not implemented.

### Roadmap
* Semantic / structure-aware chunking
* Local generation fallback to reduce cloud cost
* More subjects and academic years
* Unit-level indexing and summaries
* Answer citations and confidence indicators
* Scalable hosting for college-wide deployment

---

## ✨ Why uniAI Is Different

Most AI study tools try to *teach*.
**uniAI is designed to help students *score*.**

It is intentionally:
* Less creative
* More constrained
* More exam-oriented

That trade-off is deliberate — and it defines the project.

---

## 📌 Status

* Project stage: **Active development / prototype**
* Target users (current): Self + small group of classmates
* Future goal: College-wide deployment (subject to feasibility)

If you’re reading this as a reviewer or collaborator: the focus of uniAI is not novelty, but **alignment with real academic needs** and **practical engineering trade-offs**.
