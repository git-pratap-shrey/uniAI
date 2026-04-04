# Pipeline: Embedding and Router Artifact Generation

## Module Overview

Generates offline artifacts used by the RAG query-time routing system. Two key files enable the hybrid router to classify queries without LLM calls:

| Script | Output | Used By |
|---|---|---|
| `generate_keyword_map.py` | `data/subject_keywords.json` | Hybrid router Stage 1 (keyword scoring) |
| `generate_unit_embeddings.py` | `pipeline/embeddings/unit_embeddings.pkl` | Hybrid router Stage 2 (embedding similarity) |
| `retrieval_utils.py` | N/A (library) | All retrieval operations |
| `embeddings/local_embedding.py` | N/A (library) | All embedding generation |

---

## Per-File Documentation

### `embeddings/local_embedding.py`

Thin wrapper delegating to the centralized `models` registry.

**`embed(texts: list[str]) -> list[list[float]]`** -- Calls `models.embed()` with configured embedding provider (ollama) and model (`qwen3-embedding:4B`).

### `generate_keyword_map.py`

Builds subject-to-keywords mapping from all three ChromaDB collections, using an LLM to extract search terms.

**Constants:** `STOP_WORDS`, `MAX_ITEMS_PER_UNIT=30`, `MAX_ITEMS_PER_SUBJECT=50`, `MAX_KEYWORD_WORDS=5`, `OUTPUT_FILE=data/subject_keywords.json`.

**Functions:**
- `clean_llm_output(raw_output) -> list[str]` -- Strips markdown/numbering, filters length (3-60 chars), removes stop words, digits, unit labels, multi-clause phrases.
- `split_core_and_specific(unit_kws) -> dict` -- Promotes keywords appearing in 2+ units to "core" bucket; removes core from unit-specific lists.
- `load_checkpoint() / save_checkpoint(final_map)` -- Enables resumable runs via `subject_keywords.json`.
- `fetch_collection(client, collection_name, include) -> dict` -- Safe collection getter.
- `collect_notes_syllabus(metadatas) -> dict[str, dict[str, set]]` -- Groups as `subject -> unit_label -> {titles}`.
- `collect_syllabus(metadatas, documents) -> dict[str, dict[str, set]]` -- Groups as `subject -> unit_label -> {topic_snippets}`, extracts from embedded document text.
- `collect_pyq(metadatas, documents) -> dict[str, set]` -- Groups as `subject -> {question_snippets}`, uses actual question text.
- `extract_keywords_for_unit(ollama_client, subject, items, unit, max_items) -> list[str]` -- Calls LLM via `prompts.keyword_extraction()`, falls back to raw items on failure.
- `generate_keyword_map() -> None` -- Orchestrates: fetch from ChromaDB, group data, extract keywords per subject (notes/syllabus/pyq), save with checkpointing.

**Output format:** `{"COA": {"notes": {"core": [...], "1": [...]}, "syllabus": {...}, "pyq": [...]}}`

### `generate_unit_embeddings.py`

Generates dense embeddings for each subject+unit from the keyword map.

**Functions:**
- `build_unit_texts() -> dict[str, str]` -- Reads `subject_keywords.json`, collects unit labels from notes+syllabus, concatenates keywords per unit into text blobs. Returns `{"SUBJECT_unit": "keyword1 keyword2 ..."}`.
- `main() -> None` -- Builds texts, generates embeddings via `embed()`, saves as pickle to `unit_embeddings.pkl`.

### `retrieval_utils.py`

**`retrieve_with_threshold(collection, query, n_initial=10, similarity_threshold=None, metadata_filter=None) -> dict`** -- Generates query embedding, queries ChromaDB, converts distances to similarity (`1.0 - distance` for cosine space), filters results below threshold. Returns filtered dict matching ChromaDB structure.

---

## Inter-File Relationships

**Execution order:** extract -> ingest -> generate_keyword_map -> generate_unit_embeddings

**Dependencies:**
- `generate_keyword_map.py` reads ChromaDB, uses `prompts.keyword_extraction`, writes `subject_keywords.json`
- `generate_unit_embeddings.py` reads `subject_keywords.json`, writes `unit_embeddings.pkl`
- Both use `embeddings/local_embedding.py` which delegates to `models.embed()`
- `retrieval_utils.py` uses `local_embedding` for query-time retrieval
