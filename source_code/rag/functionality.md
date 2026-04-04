# `rag/` Module — Functionality Documentation

## Overview

The `rag/` module is the core query runtime of uniAI. It orchestrates every stage of a student's query from arrival to final answer. The pipeline follows a strict sequence:

1. **Query Expansion** — normalizes exam phrasing, expands abbreviations, injects syllabus keywords
2. **Hybrid Routing** — 4-tier cascading router (regex → keyword → embedding → LLM) to detect subject + unit
3. **Retrieval** — metadata-filtered cosine similarity search across 3 isolated ChromaDB collections
4. **Cross-Encoder Reranking** — Qwen3-Reranker-0.6B rescores candidates for deep semantic relevance
5. **Hallucination Gate** — if top score falls below threshold, switches to Generic AI Tutor Mode
6. **Context Building** — formats chunks + conversation history into LLM-ready prompts
7. **Generation** — calls the centralized models registry for text generation
8. **CLI Interface** — interactive chat loop with session management and slash commands

---

## Per-File Documentation

### `rag_pipeline.py` — Main Orchestrator

**Purpose:** The single entry point for the entire RAG pipeline. Every stage of a query passes through here.

#### Constants
- `MAX_HISTORY_TURNS` — conversation turn limit from config (default 4)
- `FOLLOWUP_PATTERNS` — regex patterns that detect follow-up questions ("repeat", "again", "summarize", "previous", etc.)
- `GENERIC_PATTERNS` — patterns that trigger generic mode ("write code", "implement", "beyond syllabus")

#### Intent Detection Functions

- `_is_followup(query: str) -> bool` — checks if query matches common follow-up phrasing to skip retrieval and rely on conversation history
- `_is_unit_overview(query: str, unit: str | None) -> bool` — detects if user is asking for a unit topic list
- `_detect_mode(query: str) -> str` — returns `"syllabus"` or `"generic"` based on query content

#### Generation

- `_trim_history(history: list[dict]) -> list[dict]` — keeps only the last `MAX_HISTORY_TURN * 2` turns to save tokens
- `_generate(prompt: str) -> str` — calls `models.chat()` with configured temperature

#### Public API

- `answer_query(query, history=None, session_subject=None) -> dict` — the main public entry point
  - **Args:** `query` (student's question), `history` (conversation turns list), `session_subject` (optional subject lock)
  - **Returns:** `{"answer": str, "subject": str, "unit": str, "mode": str, "sources": list[str], "chunks": list[dict], "expanded_query": str}`
  - **Pipeline flow:**
    1. Trim history
    2. Expand query via `query_expander.expand_query()`
    3. Route via `hybrid_router.route()` to get subject + unit
    4. Detect mode (syllabus vs generic)
    5. If follow-up + history exists → skip retrieval, build prompt from history only
    6. Retrieve notes via `search.retrieve_notes()` and syllabus via `search.retrieve_syllabus()`
    7. Merge chunks and rerank via `cross_encoder.rerank_cross_encoder()`
    8. If top score < `MIN_CROSS_SCORE` or no ranked results → switch to generic mode and clear ranked chunks
    9. Build context via `context_builder.build_context()`
    10. Construct prompt via `prompts.rag_answer()`
    11. Generate answer and return enriched result dict

---

### `hybrid_router.py` — Master Router

**Purpose:** Coordinates the 4-tier routing strategy to determine the subject and unit of a query.

#### Data Classes
- `RouteResult` — `dataclass(subject: str | None, unit: str | None, method: str)` where method is one of `"keyword"`, `"embedding"`, `"llm"`, `"none"`

#### Functions

- `_llm_classify_subject_unit(query: str) -> RouteResult` — fallback router that uses the LLM to classify both subject and unit simultaneously. Builds a list of subject and subject_unit combinations from the keyword map, prompts the router model, and parses the response. Returns `RouteResult(None, None, "none")` on failure.
- `route(query: str, session_subject: str | None = None) -> RouteResult` — main entry point
  - **Tier 1:** `detect_unit()` via regex for explicit unit mention
  - **Tier 2:** `detect_subject()` via keyword scoring (from `router.py`)
  - **Tier 3:** `embedding_router.route()` via cosine similarity to pre-computed unit embeddings
  - **Tier 4:** `_llm_classify_subject_unit()` LLM fallback
  - **Fallback:** returns session_subject and explicit_unit, both potentially None

**Key interaction:** `session_subject` (from CLI `/switch` command) overrides the keyword-detected subject but not unit. `explicit_unit` from regex overrides all unit detection methods.

---

### `router.py` — Stage 1: Keyword Scoring Router

**Purpose:** Detects the subject of a query using weighted keyword scoring against `subject_keywords.json`.

#### Scoring Weights
| Signal | Weight |
|---|---|
| PYQ keywords | 5 |
| Notes unit keywords | 4 |
| Syllabus unit keywords | 3 |
| Core keywords (notes/syllabus) | 2 |
| Unknown | 0 |

#### Constants
- `QUESTION_TOKENS` — set of common exam question prefixes to strip before scoring ("what", "define", "explain", "short note on", etc.)
- `KEYWORDS_FILE` — path to `subject_keywords.json` from config

#### Functions

- `_flatten_keywords(entry) -> list[str]` — consolidates nested subject keyword dict into flat list. Handles both legacy flat format and new nested format
- `_score_subject(query_lower: str, entry) -> float` — computes weighted score for one subject by matching query words against keywords from notes, syllabus, and PYQ collections
- `_llm_classify(query: str) -> str | None` — fallback LLM subject classification. Calls router model with `subject_router` prompt
- `detect_subject(query: str, debug: bool = False, allow_llm_fallback: bool = True)` — main entry point
  - Strips exam question prefixes from query
  - Scores all subjects via `_score_subject()`
  - If one subject wins with score >= `KEYWORD_MIN_SCORE`, also runs `score_units()` to detect unit
  - Falls back to `_llm_classify()` if no clear winner
- `list_subjects() -> list[str]` — returns all known subjects from the keyword map

---

### `embedding_router.py` — Stage 2: Embedding Similarity Router

**Purpose:** Routes queries via cosine similarity against pre-computed unit embeddings stored in `unit_embeddings.pkl`.

#### Functions

- `cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float` — computes cosine similarity between two vectors, returns 0.0-1.0
- `route(query: str) -> tuple[str | None, str | None, float]` — main entry point
  - Embeds query via `pipeline.embeddings.local_embedding.embed()`
  - Compares against all entries in `_unit_embeddings` dict
  - If best similarity exceeds `EMBEDDING_ROUTER_THRESHOLD` (0.55), returns `(subject, unit, score)`
  - Key format in pickle: `"SUBJECT_UNIT"` (e.g., `"CYBER_SECURITY_3"`)
  - Returns `(None, None, 0.0)` if no match exceeds threshold or embeddings file not found

---

### `unit_router.py` — Unit Detection

**Purpose:** Identifies which syllabus unit the user is asking about using regex and keyword scoring.

#### Functions

- `detect_unit(query: str) -> str | None` — regex extraction for explicit unit mentions. Pattern: `\bunit[\s\-]*([1-9]\d*)\b`. Returns numeric string like `"3"` or None
- `score_units(query_lower: str, subject_entry) -> tuple[str, float] | None` — keyword-based unit scoring within a subject. Matches query words against unit-level keywords (notes weight=4, syllabus weight=3). Returns best unit and score or None
- `format_unit_filter(unit: str) -> str` — pass-through normalization for ChromaDB compatibility

---

### `query_expander.py` — 3-Layer Query Expansion

**Purpose:** Bridges the vocabulary gap between how students phrase questions and how academic materials are written.

#### Layer Data
- `_EXAM_PHRASING` — compiled regex for stripping exam-style prefixes ("write a short note on", "define", "explain", etc.)
- `ABBREV_MAP` — hardcoded abbreviation expansions (19 entries including "ddos" → "distributed denial of service", "cpu" → "central processing unit", etc.)
- `_subject_aliases` — loaded from `subject_aliases.json` for domain-specific shorthand

#### Functions

- `normalize_exam_phrasing(query: str) -> str` — strips exam prefixes, collapses whitespace
- `expand_abbreviations(query: str) -> tuple[str, set[str]]` — matches abbreviations via word boundaries, returns (original_query, set_of_expansion_terms)
- `get_unit_keywords(subject: str, unit: str | None, top_n: int = 6) -> list[str]` — fetches unit-specific + core keywords from the keyword map
- `expand_query(user_query: str, subject: str | None = None, unit: str | None = None) -> str` — main entry point. Returns `normalized + " " + " ".join(abbrev_terms + syllabus_terms)` when expansions exist, otherwise just normalized query

---

### `search.py` — ChromaDB Retrieval

**Purpose:** All database queries flow through here. Provides collection-isolated retrieval with metadata filtering across 3 ChromaDB collections.

#### Types
- `Chunk` — `TypedDict` with `text`, `metadata`, `distance`, `similarity`, `collection` fields

#### Internal Functions

- `_get(alias: str) -> chromadb.Collection` — lazy-loads a collection by alias ("notes", "syllabus", "pyq")
- `normalize_unit(raw: str | int | None) -> str | None` — standardizes unit identifiers to plain numeric strings
- `_unit_filter(unit: str) -> dict` — builds ChromaDB `$or` clause for backward compatibility (matches both `"3"` and `"unit3"`)
- `_build_where(subject, unit, extra) -> dict | None` — composes nested `$and` filter from subject/unit/extra constraints
- `_query_collection(alias, query, where, k, threshold) -> list[Chunk]` — executes the actual query: embeds text, calls ChromaDB, filters by distance threshold, returns Chunk list

#### Public API

- `collection_exists(alias: str) -> bool` — checks if a collection exists in ChromaDB
- `retrieve_notes(query, subject, unit, k, threshold) -> list[Chunk]` — retrieves lecture notes with `document_type != "syllabus"` exclusion filter
- `retrieve_syllabus(query, subject, unit, k, threshold) -> list[Chunk]` — retrieves syllabus chunks from the syllabus collection
- `retrieve_pyq(query, subject, unit, k, threshold, marks, year) -> list[Chunk]` — retrieves past year questions with optional marks/year filters, uses higher default threshold (0.60)
- `retrieve_all(query, subject, unit, notes_k, syllabus_k, threshold) -> list[Chunk]` — combines notes + syllabus results for unit overview queries

#### Configuration Defaults
- Notes: `k=8`, `threshold=0.35`
- Syllabus: `k=7`, `threshold=0.35`
- PYQ: `k=5`, `threshold=0.60`

---

### `cross_encoder.py` — Neural Reranker

**Purpose:** Rescores retrieved chunks using a transformer cross-encoder that processes query and document together, capturing deep semantic relationships that independent embeddings miss.

#### Functions

- `rerank_cross_encoder(query: str, chunks: list[dict], top_n=None, candidates=None) -> list[dict]` — main entry point
  - Pre-sorts chunks by cosine similarity, keeps top `candidates` (default 6)
  - Calls `models.rerank()` which uses `tomaarsen/Qwen3-Reranker-0.6B-seq-cls`
  - Attaches `final_score` to each chunk, sorts descending, returns top `top_n` (default 4)
  - Returns empty list if no input chunks

#### Integration
- Used by `rag_pipeline.py` after retrieval
- Results feed into the hallucination gate: if `ranked[0]["final_score"] < 0.65`, mode switches to `"generic"` and chunks are cleared

---

### `reranker.py` — Heuristic Reranker (Fallback/Legacy)

**Purpose:** Fast, rule-based reranking alternative using metadata signals. Currently deprecated in favor of cross-encoder reranker but kept as fallback.

#### Boosting Logic
1. **OCR Confidence** — multiplier `0.5 + (confidence / 2.0)`, so confidence=1.0 → 1.0x, confidence=0.5 → 0.75x
2. **Unit Match** — 1.15x if chunk's unit matches predicted unit
3. **Doc Type** — 0.90x penalty for syllabus-type documents in general queries

#### Functions

- `rerank(chunks, predicted_unit=None, top_n=None) -> list[dict]` — applies heuristic boosts and returns top N scored chunks. Default `top_n` from `CONFIG["rag"]["rerank_top_n"]` (7)

---

### `context_builder.py` — Prompt Engineering Layer

**Purpose:** Transforms raw retrieval data into structured text for LLM consumption, preserving metadata for accurate citation.

#### Functions

- `build_context(chunks: list[dict]) -> str` — formats chunks into a single context string
  - Each chunk gets header: `[Source N | Title | Unit X | doc_type | relevance=0.XX]`
  - Chunks separated by `\n\n---\n\n`
- `build_history_block(history: list[dict]) -> str` — formats conversation turns as `"Previous conversation:\nUSER: ...\nASSISTANT: ..."`
- `format_sources_for_display(chunks: list[dict]) -> list[str]` — human-readable citation lines for CLI: `"filename.pdf (p.1) | Unit 3 | similarity=0.72"`

---

### `chat_cli.py` — Command Line Interface

**Purpose:** Interactive chat loop that serves as the thin View/Controller layer for the RAG system.

#### Display Functions

- `_print_header()` — welcome message and command reference
- `_print_answer(result: dict)` — formats pipeline response showing mode/subject/unit/sources
- `_print_history(history: list[dict])` — shows recent conversation turns (truncated to 120 chars)

#### Command Handler

- `_handle_command(query, session_subject, history) -> tuple[str | None, bool]` — processes slash commands:
  - `/switch <SUBJECT>` — locks session to a subject
  - `/switch` — clears subject lock
  - `/subject` — shows current session subject
  - `/subjects` — lists all known subjects
  - `/history` — shows conversation history
  - `/clear` — clears history

#### Main Loop

- `chat()` — interactive loop
  1. Reads user input
  2. Handles slash commands
  3. Calls `rag_pipeline.answer_query()`
  4. On first query, auto-detects and locks subject
  5. Updates conversation history (user + assistant turns)
  6. Prints formatted response with sources

---

## Data Flow / Inter-File Interactions

```
chat_cli.py
    │
    ▼
rag_pipeline.py ── answer_query()
    │
    ├── query_expander.py ── expand_query()
    │       ├── normalize_exam_phrasing()
    │       ├── expand_abbreviations()
    │       └── get_unit_keywords()
    │
    ├── hybrid_router.py ── route()
    │       ├── unit_router.py ── detect_unit() (regex)
    │       ├── router.py ── detect_subject() (keyword scoring)
    │       ├── embedding_router.py ── route() (embedding similarity)
    │       └── _llm_classify_subject_unit() (LLM fallback)
    │
    ├── search.py ── retrieve_notes() + retrieve_syllabus()
    │       └── pipeline/embeddings/local_embedding.py ── embed()
    │
    ├── cross_encoder.py ── rerank_cross_encoder()
    │       └── models.py ── rerank() (Qwen3-Reranker)
    │
    ├── context_builder.py ── build_context() + build_history_block()
    │
    ├── prompts.py ── rag_answer()
    │
    └── models.py ── chat() (generation)
```
