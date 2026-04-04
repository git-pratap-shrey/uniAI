# `tests/` Module — Functionality Documentation

## Overview

The test suite validates every layer of the uniAI system — from individual component units to complete end-to-end RAG pipeline runs. It is organized by what aspect of the system is being tested rather than by mirroring the source code structure.

**Test subdirectories:**

| Directory | Purpose |
|---|---|
| `chat/` | Manual chat session scripts, question sets, and sweep runners |
| `retrieval/` | Retrieval accuracy, subject/unit routing, and isolation tests |
| `router/` | Router evaluation, bug exploration, and preservation tests with trace logs |
| `complete_system/` | Full RAG pipeline integration tests with expected-answer comparison and rich reports |
| `ci/` | CI/CD pipeline tests (syntax checks, Django health, pytest) |
| `db/` | ChromaDB audit and dump utilities |
| `others/` | Miscellaneous unit tests (parsing, query expander, chunking, config verification) |
| `api/` | Individual API provider smoke tests (Gemini, Groq) |

---

## Per-Subdirectory Breakdown

### `chat/` — Chat Testing & Question Sets

Test data and scripts for manual chat session testing, primarily focused on Cyber Security stress testing.

#### Files

- **`testing.py`** — Main chat test framework. Connects to an existing Django server, sends questions, and records answers.
  - Sends predefined questions from `questions.txt` to the running server
  - Records responses for manual review
  - Used for interactive quality assurance sessions

- **`sweep.py`** — Batch question runner. Sends multiple questions in a single sweep and collects results into a JSON file.
  - Iterates over a question list, hits `/api/query`, writes results to JSONL
  - Used for quick regression sweeps after code changes

- **`questions.txt`** — 80-question Cyber Security test set organized into 10 sections (sanity checks, syllabus-based, unit-specific, boundary tests, hybrid, adversarial, retrieval confidence, non-academic, follow-up, edge cases). Also replicated as `cyber_security_rag_test_questions.txt`.

- **`results.jsonl`, `results_1.jsonl`, `results_2.jsonl`** — Historical test output files with question/answer pairs from previous sessions.

---

### `retrieval/` — Retrieval & Routing Tests

Tests the routing and retrieval subsystems in isolation.

#### Files

- **`test_01_subject_unit.py`** — Subject and unit routing test suite.
  - Verifies that queries route to the correct subject and unit
  - Tests keyword scoring, embedding similarity, and LLM fallback paths
  - Each test case provides a query string and expected routing result

- **`test_02_separation.py`** — Collection isolation test.
  - Ensures notes retrieval doesn't return syllabus chunks
  - Verifies that `document_type != "syllabus"` filter is working in `search.py`
  - Cross-collection contamination detection

- **`test_03_pipeline.py`** — End-to-end retrieval pipeline test.
  - Full retrieve → rerank → context build sequence without generation
  - Validates chunk count, relevance scores, and context formatting

---

### `router/` — Router Evaluation & Debugging

The most heavily tested subsystem due to routing being the single most critical component for answer quality.

#### Files

- **`router_evaluator.py`** — Main router evaluation tool. Reads test cases from JSON, runs each query through the hybrid router, and produces accuracy metrics.
  - Loads questions from `router_test_cases.json`
  - Compares expected route vs actual route (keyword/embedding/llm/none)
  - Outputs pass/fail rates and misclassification analysis

- **`test_30_questions.py`** — 30-question focused router test. Narrower scope than the full evaluator.
  - Quick smoke test for common query patterns
  - Validates tier ordering (keyword first, embedding second, LLM last)

- **`router_debug.py`** — Diagnostic tool for individual query routing. Prints detailed routing trace for a single query showing which tier matched, what score was returned, and why fallback was or wasn't triggered.

- **`run_router_tests.py`** — Test runner that executes router tests and outputs results summary.

- **`test_bug_exploration.py`** — Ad hoc bug investigation script. Used to reproduce specific routing failures and understand edge cases.

- **`test_preservation.py`** — Regression preservation test. Ensures routing improvements don't break previously-working queries.

- **`bug_exploration_results.md`**, **`preservation_test_summary.md`** — Human-readable test result summaries.

- **`router_test_cases.json`**, **`router_test_cases_backup.json`**, **`router_test_cases_debug.json`** — Router test case JSON files with expected routing results.

- **`logs/`** — Directory of trace JSON files from router evaluation runs. Each trace captures the full routing path for a query including all tier scores.
  - `trace_*.json` — Individual query routing traces (19 traces across 2 runs)
  - `results_*.json` — Aggregated results per run (2 result files)

---

### `complete_system/` — Full Pipeline Integration Tests

End-to-end tests that feed questions through the entire RAG pipeline and compare against expected answers.

#### Files

- **`run_test.py`** — Main test runner for complete system evaluation.
  - **Usage:** `python run_test.py [--question Q1] [--subject "DIGITAL ELECTRONICS"]`
  - Reads questions from `questions.txt`
  - Runs each through the full pipeline: route → retrieve → rerank → generate
  - Uses `parser.py` to parse question files and `reporter.py` for rich output
  - Outputs: rich table in terminal, JSON file with results, side-by-side text report
  - Supports single question or full batch execution

- **`parser.py`** — Question file parser with structured format support.
  - `Question` dataclass: `question_id`, `subject`, `subject_code`, `query`, `expected_answer`, `line_number`
  - `parse_questions_file(filepath)` — parses format with `SUBJECT:` headers, `Q1:` numbering, `EXPECTED ANSWER:` blocks
  - `get_subject_alias(subject_name)` — maps display names to internal codes (e.g., "DIGITAL ELECTRONICS" → "DIGITAL_ELECTRONICS")

- **`reporter.py`** — Rich report generation for test results.
  - `TestResult` dataclass with actual answer, metadata, timing info
  - Generates terminal tables, JSON output, and side-by-side text reports
  - Includes system metadata (model version, config hash, timestamp)
  - Report files saved as `report_YYYYMMDD_HHMMSS.txt` in timestamped subdirectories

- **`questions.txt`** — Primary question file for complete system tests.
  - Multi-subject format with SUBJECT headers and EXPECTED ANSWER blocks

- **`old_reports/`** — Historical test reports from previous runs, including both JSON and text formats for regression comparison.

---

### `ci/` — CI/CD Pipeline Tests

Tests designed to run in GitHub Actions CI workflows.

#### Files

- **`test_db.py`** — Database connectivity and schema test.
  - Verifies ChromaDB collections exist and are queryable
  - Checks collection item counts and metadata structure

- **`test_chat.py`** — Simple chat endpoint test for CI.
  - Verifies the RAG pipeline responds without error to a basic query

- **`test_router.py`** — Quick router sanity check for CI.
  - Confirms keyword router returns a non-None result for known-good queries

---

### `db/` — Database Utilities

Diagnostic and audit tools for ChromaDB state.

#### Files

- **`db_audit.py`** — Comprehensive ChromaDB audit.
  - Lists all collections with item counts
  - Shows metadata structure for sample items
  - Detects orphaned or inconsistent data

- **`db_dump.py`** — Database dump utility.
  - Exports ChromaDB collections to text/JSON for inspection
  - Used for debugging retrieval quality issues
  - Output: `db_dump_output.txt`

---

### `others/` — Miscellaneous Unit Tests

Individual component tests that don't fit into the main test categories.

#### Files

- **`test_parse.py`** — Parser test for question file format validation.
  - Tests `parser.py` edge cases (malformed headers, missing answers, etc.)

- **`test_query_expander.py`** — Query expansion test suite.
  - Tests abbreviation expansion (abbrev MAP entries)
  - Tests exam phrase normalization
  - Tests syllabus keyword injection
  - Validates that expanded queries are longer but more focused

- **`test_query.py`** — General query processing test.
  - Tests ChromaDB query execution with various filters

- **`test_models_registry.py`** — Models registry unit test.
  - Verifies lazy client initialization
  - Tests error handling for missing providers/keys

- **`chunk_test.py`** — Chunking test.
  - Tests text chunk boundaries and length constraints
  - Validates metadata preservation through chunking

- **`verify_config.py`** — Configuration verification script.
  - Loads CONFIG dict and validates all required keys exist
  - Checks API keys are non-empty
  - Used before pipeline runs to fail fast on misconfiguration

---

### `api/` — API Provider Smoke Tests

Standalone tests for individual LLM providers to verify API key validity and connectivity.

#### Files

- **`gemini_api_test.py`** — Simple smoke test for Gemini API.
  - Sends a test prompt, verifies response is non-empty
  - Used to debug API key issues before full pipeline runs

- **`groq_api_test.py`** — Simple smoke test for Groq API.
  - Same pattern as Gemini test, validates Groq connectivity

---

## Test Input Files (Non-Python)

| File | Description |
|---|---|
| `chat/questions.txt` | 80-query Cyber Security stress test (10 sections) |
| `chat/cyber_security_rag_test_questions.txt` | Duplicate of questions.txt |
| `complete_system/questions.txt` | Multi-subject question file with expected answers |
| `router/router_test_cases.json` | 8 router test cases with expected routes |
| `router/router_test_cases_backup.json` | Backup of test cases |
| `router/router_test_cases_debug.json` | Debug-version test cases |

## Test Output Artifacts

| File/Pattern | Description |
|---|---|
| `chat/results.jsonl` | JSONL results from chat sweeps |
| `chat/results_1.jsonl`, `results_2.jsonl` | Result sets from individual sweep runs |
| `router/logs/results_*.json` | Aggregated router run results |
| `router/logs/trace_*.json` | Per-query routing traces with full tier scoring detail |
| `router/bug_exploration_results.md` | HTML/Markdown test findings for routing bugs |
| `router/preservation_test_summary.md` | Summary of preservation test results |
| `db/db_dump_output.txt` | ChromaDB dump output |
| `complete_system/old_reports/` | Historical full-pipeline reports (JSON + text) |
| `complete_system/reports/` | Dynamically created subdirectories per run with `report_*.txt` and `results_*.json` |

## Test Execution Notes

- **Prerequisites:** ChromaDB must be populated, keyword map and unit embeddings generated before running retrieval or complete system tests
- **Router tests** run independently of the database (keyword-based only)
- **Complete system tests** require all three backends (routing, retrieval, generation) to be functional
- **CI tests** are lightweight and designed to catch breakages early (syntax, connectivity, basic routing)
- **Sweep tests** (`chat/sweep.py`) are ad-hoc quality checks run during development iterations
