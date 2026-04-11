# uniAI Retrieval Architecture — Structured Design Log

## 1. Data Sources & Pipelines

A)  Notes Pipeline Source:
    `<SUBJECT>/notes/unit<N>/<pdf_stem>/<pdf_stem>_p{page}-{page}_{chunk_idx}.json`
    Collection: `multimodal_notes`
    One JSON file per semantic section per page. The extractor writes multiple
    section JSONs per page (one per distinct topic identified by the VLM).

B)  PYQ Pipeline Source:
    `<SUBJECT>/pyqs/pyqs_processed/*_processed.json`
    Collection: `multimodal_pyq`
    One JSON array per PYQ PDF, containing all extracted questions.

C)  Syllabus Pipeline Source:
    `<SUBJECT>/syllabus/syllabus_*.json`
    Collection: `multimodal_syllabus`
    Exactly 7 chunk JSONs per syllabus PDF (5 unit chunks + CO chunk + books chunk).

Data layout is **flattened** — no `year_2/` nesting. Subject folders sit directly
under `BASE_DATA_DIR`.

Each collection is logically isolated. This separation is foundational
to specialized retrieval.

## 2. Metadata Philosophy: Deterministic vs VLM

**Deterministic Metadata (Trusted)**
Derived from:
- Folder structure
- Pre-processed structured JSON
- Ingestion logic

Authoritative fields:
- subject
- unit
- year (PYQ)
- marks (PYQ)
- chunk_type (syllabus)
- collection name

These fields drive routing and filtering.

**VLM Metadata (Advisory Only)**
Derived from vision-language model:
- title (section_title)
- topics (embedded in embedding text only — not stored as metadata)
- key_concepts (embedded in embedding text only)
- document_type (notes)
- content_quality
- confidence

These enrich semantic search but must never override structural metadata.
Topics and key_concepts are intentionally excluded from ChromaDB metadata
fields (scalar-only constraint) and are instead embedded into the embedding text.

## 3. Collection-Level Metadata Structure

A)  **`multimodal_notes`**

Fields:
- `source`, `page_start`, `page_end`, `page_count`, `unit` (normalized numeric), `subject` (uppercase), `title` (section_title), `document_type`, `chunk_idx`, `section_index`, `confidence`

Used for: Concept explanation, unit-scoped retrieval, exam-writing answers.

B)  **`multimodal_pyq`**

Fields:
- `source`, `unit`, `subject`, `document_type = "pyq"`, `year`, `marks` (0 if null)

Used for: Unit-based PYQ retrieval, marks filtering, year filtering, exam trend analysis.

C)  **`multimodal_syllabus`**

Fields:
- `source`, `unit` ("" or numeric), `subject`, `title`, `document_type = "syllabus"`, `syllabus_version`, `chunk_type` (unit_N, course_outcomes, books_references), `confidence = 1.0`

Used for: Course outcomes, unit syllabus, reference books, exam scope clarification.

## 4. Query Processing & Routing

Before retrieval, queries go through an expansion and routing system to ensure accurate context localization.

**A) Query Expansion (`query_expander.py`)**
- Strips exam-style phrasing (layer 1).
- Expands known abbreviations using hardcoded map + `subject_aliases.json` (layer 2).
- Appends syllabus keywords for detected subject/unit from `subject_keywords.json` (layer 3).

**B) Hybrid Subject & Unit Router (`hybrid_router.py`)**
A **four-tier** cascading router:
1. **Regex Detection (Tier 1):** Extracts explicit unit mentions (`\bunit[\s\-]*([1-9]\d*)\b`).
2. **Keyword Router (Tier 2):** Weighted scoring against `subject_keywords.json`. PYQ keywords weight=5, notes unit keywords=4, syllabus unit=3, core subject=2. Min score threshold = 2.
3. **Embedding Similarity Router (Tier 3):** Embeds query, compares against pre-computed unit embeddings (`unit_embeddings.pkl`). Threshold = 0.55.
4. **LLM Fallback (Tier 4):** Fast router model (temperature=0.0) classifies subject+unit simultaneously from a structured prompt. Last resort only.

## 5. Retrieval Strategy Architecture

**Layer 1: Intent Classification (Rule-Based & Hybrid Router)**
Examples:
- "course outcome" → deterministic syllabus filter
- "unit 3 syllabus" → deterministic syllabus filter
- "reference books" → deterministic syllabus filter
- "10 mark questions from unit 4" → deterministic PYQ filter
- Conceptual query → notes semantic retrieval

**Layer 2: Metadata Filtering**
- Course Outcomes: collection = multimodal_syllabus, where = { subject: SUBJECT, chunk_type: "course_outcomes" }
- Unit Syllabus: collection = multimodal_syllabus, where = { subject: SUBJECT, unit: "3" }
- PYQ by Unit: collection = multimodal_pyq, where = { subject: SUBJECT, unit: "2" }
- Notes by Unit: collection = multimodal_notes, where = { subject: SUBJECT, unit: "4" }

**Layer 3: Semantic Ranking Within Filtered Subset**
Process:
1. Detect subject (Hybrid Router)
2. Detect unit if present (Hybrid Router)
3. Filter collection
4. Embed extended query
5. Rank top K inside subset
6. Rerank top candidates with cross-encoder
7. Apply hallucination gate
8. Generate answer

**Layer 4: Cross-Encoder Reranking + Hallucination Gate**
- `tomaarsen/Qwen3-Reranker-0.6B-seq-cls` rescores up to 6 candidate chunks
- Scores sigmoid-normalized to 0–1 range (GPU via PyTorch CUDA)
- If top score < 0.65 → discard chunks, switch to Generic AI Tutor Mode

**Layer 5: LLM Generation**
LLM must:
- Use retrieved chunks only (when in RAG mode)
- Follow exam-writing tone
- Avoid generic tutoring unless switched to Generic Mode

Fallback to Generic Mode:
- Cross-encoder top score < 0.65
- No relevant chunks found
- Query outside syllabus scope

## 6. Specialized Retrieval Modes

- **Syllabus-Aware Strict Mode:** Only answer from syllabus collection.
- **Unit-Scoped Retrieval Mode:** Filter by subject + unit before ranking.
- **Exam Pattern Mode:** Use PYQ collection with unit, marks, and year filters.
- **Follow-up Mode:** Skip retrieval entirely, build prompt from conversation history only.
- **Unit Overview Mode:** Combine notes + syllabus with `retrieve_all()` for topic-listing queries.
- **Cross-Collection Hybrid Mode:** Combine syllabus topics + PYQ frequency + notes density to identify important topics.

## 7. Non-Generic Retrieval Principle

- Generic RAG: Embed everything → search everything → hope best
- uniAI Architecture: Query Expansion → Multi-stage Routing (Intent) → Deterministic Metadata → Constrained Semantic Ranking → Cross-Encoder Reranking → Hallucination Gate → Structured Generation

## 8. VLM Extraction Architecture

**Notes Extraction (semantic sectioning):**
- One VLM call per page using `notes_extraction(existing_topics)` prompt
- VLM returns `sections[]` — one entry per distinct topic found on the page
- Running topic list accumulated across pages prevents duplicate topic IDs
- Previously processed pages are detected and skipped; topics rehydrated from disk
- Output: one JSON per section, named `<stem>_p{N}-{N}_{chunk_idx}.json`

**PYQ Extraction:**
- Per-page VLM OCR → normalize text → split by sections → extract questions
- Second LLM call per question for unit classification
- Rate-limit safe: 15s × attempt exponential backoff (handles Cloudflare 524s on cloud Ollama)
- Collision-safe IDs include section slug + running counter

**Vision Providers:**
- Primary: Ollama (cloud or local, scale=1.0 JPEG for cloud, scale=2.0 PNG for HF)
- Fallback: OpenRouter (REST API, base64 PNG, 120s timeout)
- Alternative: HuggingFace InferenceClient (base64 data-URIs)

## 9. Current Strengths

- Deterministic structural metadata guarantees
- Multi-stage highly accurate Hybrid Subject/Unit routing (4 tiers)
- Normalized units and subjects (alias resolution)
- Garbage and hallucinatory extraction filtering
- Collection isolation (Syllabus, PYQ, Notes)
- Cross-encoder reranking (neural, not just cosine)
- Hallucination gate prevents confident wrong answers
- Section-level granularity in notes (one semantic topic per chunk)
- Rate-limit safe extraction with exponential backoff
- Multi-provider vision fallback (Ollama → OpenRouter → HuggingFace)
- Robust skip/resume logic in all extraction pipelines

## Final Architectural State

The system is a structured academic retrieval engine with:
- Deterministic structural formatting layer
- Semantic enrichment and semantic query expansion layer
- Isolated domain collections mapped efficiently via Hybrid Routing
- Clean metadata guarantees enabling unit-scoped retrieval
- Cross-encoder reranking + hallucination gating for answer quality
- Exam-focused constraint generation capability
- Section-level notes extraction for fine-grained retrieval
- Multi-provider vision infrastructure for extraction resilience

Active pipeline stages as of 2026-04-10: Notes extraction running, PYQ extraction complete, syllabus extraction complete.
