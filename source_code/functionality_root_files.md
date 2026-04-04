# Root-Level Source Files

## Module Overview

| File | Purpose |
|---|---|
| `models.py` | Unified provider abstraction for chat, embedding, reranking, and vision |
| `prompts.py` | Single source of truth for all LLM prompts |
| `utils.py` | Shared helpers: image encoding, JSON parsing, embedding, ChromaDB |
| `__init__.py` | Empty package marker |

---

## Per-File Documentation

### `models.py`

The architectural core. Every module calls `models.chat()`, `models.embed()`, `models.rerank()`, or `models.vision()` instead of provider SDKs directly.

**Lazy-loaded clients:** `_clients` dict initialized to None. `get_ollama_client()`, `get_gemini_client()`, `get_groq_client()` -- each creates client on first use with appropriate API key from CONFIG.

**`chat(prompt, system_prompt, messages, model, provider, **kwargs) -> str`**
- Resolves provider/model from CONFIG if not overridden. Supports simple prompt, system+prompt, or full messages array.
- **Gemini:** `client.models.generate_contents()` with config_args (temperature, max_output_tokens, top_p). Returns `response.text`.
- **Ollama:** `client.chat()` with full_messages (system prepended if provided), options (temperature, num_ctx, top_p). Returns `response["message"]["content"]`.
- **Groq:** `client.chat.completions.create()` with messages, temperature, max_tokens. Returns `completion.choices[0].message.content`.
- On error returns error string instead of raising.

**`embed(texts, model, provider) -> List[List[float]]`**
- Provider defaults to ollama, model to `qwen3-embedding:4B`. Calls `client.embeddings()` per text with `keep_alive="10m"`. Returns list of vectors.

**`rerank(query, documents, model) -> List[float]`**
- Loads `tomaarsen/Qwen3-Reranker-0.6B-seq-cls` lazily (thread-safe, auto-detects CUDA, float16 on GPU).
- Qwen3-Reranker format: formats each pair as chat-style system+instruct+query+document tags.
- Tokenizes, passes through model, applies `sigmoid()` normalization to 0-1 range. Max length 8192, left padding.

**`vision(images, prompt, model, provider) -> str`**
- **Ollama:** Accepts file paths or bytes, reads/casts to bytes, calls `client.generate()`.
- **HuggingFace:** Converts images to base64 data URIs (`pil_to_base64`), uses `InferenceClient` chat completions with image_url content type.

### `prompts.py`

Organized into five groups.

**EXTRACTION:**
- `NOTES_EXTRACTION` -- VLM prompt: OCR PDF page images, output structured JSON with full_text, title, unit, document_type, topics, key_concepts, diagrams_present, content_quality, confidence.
- `SYLLABUS_EXTRACTION` -- VLM prompt: parse syllabus tables, output JSON with syllabus_version, subject_name, units[], course_outcomes[], textbooks[], reference_books[].

**RAG CHAT (builder functions):**
- `rag_answer(query, notes_context, history_block, mode="syllabus", subject)` -- Syllabus mode: exam-focused assistant using notes as authoritative source. Generic mode: labels as "[General Knowledge]". Appends history, notes, and question (repeated twice).
- `topic_list(subject, unit)` -- Simple unit topic listing prompt.

**ROUTING (builder functions):**
- `subject_router(query, subjects_list)` -- Strict prompt, exact subject name only.
- `subject_unit_router(query, subjects_units_list)` -- Same for subject_unit combos.

**PIPELINE (builder function):**
- `keyword_extraction(subject, items_list, unit=None)` -- Unit-specific (8-12 terms) or subject-level (10-15 phrases). Comma-separated output only.

**PYQ EXTRACTION:**
- `pyq_unit_classification(question, syllabus_units)` -- Classifies question into unit 1-5. Single integer output.
- `PYQ_VLM_TRANSCRIPTION` -- VLM prompt for exam paper OCR transcription.

### `utils.py`

**Image helpers:**
- `pil_to_base64(img)` -- PNG base64 data-URI for HF API
- `pil_to_bytes(img)` -- Raw PNG bytes for Ollama
- `pil_to_jpeg_bytes(img, quality=85)` -- JPEG bytes (5-10x smaller than PNG)

**JSON parsing:**
- `extract_first_json(text) -> dict|None` -- Brace-counting to find first complete JSON in noisy VLM output.

**Embedding:**
- `get_embedding(text) -> list[float]` -- Wraps `models.embed([text])[0]`.

**ChromaDB:**
- `get_chroma_collection(collection_name) -> Collection` -- Returns or creates collection with cosine space. Cached per name in `_chroma_collections`. Defaults to `multimodal_notes`.

### `__init__.py`

Empty file, marks `source_code/` as a Python package.

---

## Inter-File Relationships

```
prompts.py  (no internal deps -- pure strings/functions)
    -> used by extract/*, rag/*, pipeline/generate_keyword_map.py

utils.py  --> models.py (get_embedding), config (chroma paths)
    -> used by extract/*, ingest/*, rag/*

models.py --> config (provider selection, API keys)
    -> used by extract/*, ingest/* (via utils), rag/*, pipeline/*
```

All AI calls flow through `models.py`, which reads provider/model from CONFIG, lazy-initializes the client, normalizes provider differences, and returns consistent output.
