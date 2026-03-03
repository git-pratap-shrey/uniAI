# uniAI Retrieval Architecture -- Structured Design Log

## 1. Data Sources & Pipelines

A)  Notes Pipeline Source:
    `<year>`{=html}/`<subject>`{=html}/notes/unitX/chunk\_\*.json
    Collection: multimodal_notes

B)  PYQ Pipeline Source:
    `<subject>`{=html}/pyqs_processed/\*\_processed.json Collection:
    multimodal_pyq

C)  Syllabus Pipeline Source:
    `<subject>`{=html}/syllabus/syllabus\_\*.json Collection:
    multimodal_syllabus

Each collection is logically isolated. This separation is foundational
to specialized retrieval.

## 2. Metadata Philosophy: Deterministic vs VLM

Deterministic Metadata (Trusted) Derived from: - Folder structure -
Pre-processed structured JSON - Ingestion logic

Authoritative fields: - subject - unit - year (PYQ) - marks (PYQ) -
chunk_type (syllabus) - collection name

These fields drive routing and filtering.

VLM Metadata (Advisory Only) Derived from vision-language model: -
title - topics - key_concepts - document_type (notes) -
content_quality - confidence

These enrich semantic search but must never override structural
metadata.

## 3. Collection-Level Metadata Structure

A)  multimodal_notes

Fields: - source - page_start - page_end - unit (normalized numeric) -
subject (uppercase) - title - document_type - confidence

Used for: - Concept explanation - Unit-scoped retrieval - Exam-writing
answers

B)  multimodal_pyq

Fields: - source - unit - subject - document_type = "pyq" - year - marks

Used for: - Unit-based PYQ retrieval - Marks filtering - Year
filtering - Exam trend analysis

C)  multimodal_syllabus

Fields: - source - unit ("" or numeric) - subject - title -
document_type = "syllabus" - syllabus_version - chunk_type (unit_X,
course_outcomes, books_references) - confidence = 1.0

Used for: - Course outcomes - Unit syllabus - Reference books - Exam
scope clarification

## 4. Retrieval Strategy Architecture

Layer 1: Intent Classification (Rule-Based)

Examples: - "course outcome" → deterministic syllabus filter - "unit 3
syllabus" → deterministic syllabus filter - "reference books" →
deterministic syllabus filter - "10 mark questions from unit 4" →
deterministic PYQ filter - Conceptual query → notes semantic retrieval

Layer 2: Metadata Filtering

Course Outcomes: collection = multimodal_syllabus where = { subject:
SUBJECT, chunk_type: "course_outcomes" }

Unit Syllabus: collection = multimodal_syllabus where = { subject:
SUBJECT, unit: "3" }

PYQ by Unit: collection = multimodal_pyq where = { subject: SUBJECT,
unit: "2" }

Notes by Unit: collection = multimodal_notes where = { subject: SUBJECT,
unit: "4" }

Layer 3: Semantic Ranking Within Filtered Subset

Process: 1. Detect subject 2. Detect unit if present 3. Filter
collection 4. Embed query 5. Rank top K inside subset 6. Generate answer

Layer 4: LLM Generation

LLM must: - Use retrieved chunks only - Follow exam-writing tone - Avoid
generic tutoring unless requested

Fallback only when: - Retrieval confidence is low - No relevant chunks
found - Query outside syllabus scope

## 5. Specialized Retrieval Modes

Syllabus-Aware Strict Mode: Only answer from syllabus collection.

Unit-Scoped Retrieval Mode: Filter by subject + unit before ranking.

Exam Pattern Mode: Use PYQ collection with unit, marks, and year
filters.

Cross-Collection Hybrid Mode: Combine syllabus topics + PYQ frequency +
notes density to identify important topics.

## 6. Non-Generic Retrieval Principle

Generic RAG: Embed everything → search everything → hope best

uniAI Architecture: Intent → deterministic metadata → constrained
semantic ranking → structured generation

## 7. Current Strengths

-   Deterministic structural metadata
-   Normalized units
-   Subject normalization
-   Garbage filtering
-   Collection isolation
-   Syllabus chunk separation
-   Unit-scoped retrieval capability

## Final Architectural State

The system is now a structured academic retrieval engine with: -
Deterministic structural layer - Semantic enrichment layer - Isolated
domain collections - Clean metadata guarantees - Exam-focused retrieval
capability

Ingestion phase complete. Intelligence optimization phase begins.
