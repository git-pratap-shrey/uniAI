# Configuration System

## Module Overview

The `config/` package centralizes all configuration for the uniAI RAG system. It is split across five files, each owning a single concern:

| File | Concern |
|---|---|
| `env.py` | Secrets and machine-specific settings loaded from `.env` |
| `models.py` | AI provider profiles (Gemini, Ollama, Groq) for chat, embedding, routing, and vision |
| `rag.py` | RAG hyperparameters -- thresholds, K values, reranker settings |
| `paths.py` | Filesystem paths and ChromaDB collection names via `pathlib` |
| `main.py` | Assembles everything into a single `CONFIG` dict -- the one import point |
| `__init__.py` | Re-exports `CONFIG` so consumers can do `from source_code.config import CONFIG` |

---

## Per-File Documentation

### `env.py`
Loads environment variables from `.env` using `python-dotenv`. Only secrets and machine-specific settings.

**Exposed symbols:**
- `GROQ_API_KEY`, `GEMINI_API_KEY`, `HF_TOKEN`, `OLLAMA_API_KEY` -- API keys (default `""`)
- `OLLAMA_BASE_URL`, `OLLAMA_LOCAL_URL` -- default `"http://localhost:11434"`
- `USE_OLLAMA_CLOUD` -- bool, default `True`
- `APP_ENV` -- string, default `"dev"`

### `models.py`
Defines AI model profiles and active selection.

**Exposed symbols:**
- `MODEL_CONFIGS` (dict) -- Three chat profiles:
  - `"gemini"`: `gemini-3.1-flash-lite-preview`, temp 0.3, top_p 0.9, max_tokens 4096
  - `"ollama"`: `qwen3:8b`, temp 0.25, top_p 0.95, num_ctx 8192
  - `"groq"`: `qwen/qwen3-32b`, temp 0.6, top_p 0.95, max_tokens 4096
- `ACTIVE_CHAT_MODEL` -- currently `"gemini"`
- `get_active_model_config()` -- returns `MODEL_CONFIGS[ACTIVE_CHAT_MODEL]`
- `EMBEDDING_CONFIG` -- `{"provider": "ollama", "model": "qwen3-embedding:4B"}`
- `ROUTER_CONFIG` -- `{"provider": "ollama", "model": "gemini-3-flash-preview:latest", "temperature": 0.0, "num_predict": 50}`
- `VISION_CONFIG` -- `{"provider": "ollama", "model": "qwen3-vl:235b-cloud", "hf_model_id": "Qwen/Qwen3-VL-235B-A22B-Instruct"}`

### `rag.py`
Centralizes all RAG pipeline tuning parameters.

**Exposed symbols:**
- `RAG_CONFIG` -- similarity_threshold=0.35, min_strong_sim=0.6, notes_k=8, syllabus_k=7, pyq_k=5, pyq_threshold=0.60, all_notes_k=6, all_syllabus_k=7, rerank_top_n=7
- `CROSS_ENCODER_CONFIG` -- model=`tomaarsen/Qwen3-Reranker-0.6B-seq-cls`, min_score=0.65, candidates=6, pipeline_top_n=4
- `MAX_HISTORY_TURNS`=4, `KEYWORD_MIN_SCORE`=2, `EMBEDDING_ROUTER_THRESHOLD`=0.55, `MIN_INGEST_CONFIDENCE`=0.3, `QUERY_EXPANDER_MAX_KEYWORDS`=6

### `paths.py`
Filesystem paths and collection names.

**Exposed symbols:**
- `BASE_DIR` -- `source_code/` root
- `BASE_DATA_DIR` -- env or `BASE_DIR/data/year_2`
- `CHROMA_DB_PATH` -- env or `BASE_DIR/chroma`
- `UNIT_EMBEDDINGS_PATH` -- `BASE_DIR/pipeline/embeddings/unit_embeddings.pkl`
- `KEYWORDS_FILE_PATH` -- `BASE_DIR/data/subject_keywords.json`
- `CHROMA_COLLECTION_NAME` -- `"multimodal_notes"`
- `CHROMA_SYLLABUS_COLLECTION_NAME` -- `"multimodal_syllabus"`
- `CHROMA_PYQ_COLLECTION_NAME` -- `"multimodal_pyq"`

### `main.py`
Assembles all sub-modules into structured `CONFIG` dict with keys: `env`, `OLLAMA_BASE_URL`, `model`, `providers` (chat/embedding/router/vision), `rag` (thresholds, cross_encoder, keywords, embedding_router), `paths` (base_data, chroma, unit_embeddings, collections), `ingest` (min_confidence).

### `__init__.py`
Re-exports `CONFIG`: `from .main import CONFIG`

---

## Inter-File Relationships

```
env.py  ──────────┐
                   │
models.py ─(keys)──┤
                   ├──> main.py ──> CONFIG ──> __init__.py ──> consumers
rag.py    ─────────┤
paths.py  ─────────┘
```

env.py/rag.py are leaf nodes. models.py imports keys from env.py. paths.py imports Ollama URLs from env.py. main.py imports from all four and assembles CONFIG.
