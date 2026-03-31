"""
run_test.py
───────────
Complete system test runner for uniAI.

Feeds questions from questions.txt into the RAG pipeline and generates
comparison reports showing expected vs actual answers with full model metadata.

Usage:
    python run_test.py                    # Run all questions
    python run_test.py --question Q1    # Run specific question only
    python run_test.py --subject "DIGITAL ELECTRONICS"  # Filter by subject

Output:
    - Rich table in terminal
    - JSON file with full results
    - Side-by-side text report
"""

import sys
import os
import time
import argparse

# Add project root to path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from source_code.tests.complete_system.parser import parse_questions_file, Question
from source_code.tests.complete_system.reporter import (
    TestResult, generate_all_outputs, get_system_metadata
)


def run_single_question(question: Question) -> TestResult:
    """
    Run a single question through the RAG pipeline.

    Args:
        question: Question object with query and expected answer

    Returns:
        TestResult with actual answer and metadata
    """
    # Import here to avoid early loading
    from source_code.rag.rag_pipeline import answer_query
    from source_code.rag.unit_router import detect_unit
    from source_code.rag.router import detect_subject
    from source_code.rag.embedding_router import route as embedding_route
    from source_code.rag.hybrid_router import _llm_classify_subject_unit
    from source_code.config import CONFIG
    from source_code.tests.complete_system.reporter import RouterStageTrace

    print(f"  → Running {question.question_id}: {question.query[:50]}...")

    result = TestResult(
        question_id=question.question_id,
        subject_expected=question.subject,
        subject_code=question.subject_code,
        query=question.query,
        expected_answer=question.expected_answer
    )

    try:
        start_time = time.time()
        trace = RouterStageTrace()

        # ── Stage 1: Regex unit detection ──────────────────────────────────
        regex_unit = detect_unit(question.query)
        trace.regex_unit = regex_unit or ""

        # ── Stage 2: Keyword router ─────────────────────────────────────────
        kw_subj, kw_unit, kw_used_llm = detect_subject(question.query, debug=True)
        trace.keyword_subject  = kw_subj or ""
        trace.keyword_unit     = kw_unit or ""
        trace.keyword_used_llm = bool(kw_used_llm)
        trace.keyword_passed   = bool(kw_subj and not kw_used_llm)

        # ── Stage 3: Embedding router ───────────────────────────────────────
        emb_subj, emb_unit, emb_score = embedding_route(question.query)
        emb_threshold = CONFIG["rag"]["embedding_router_threshold"]
        trace.embedding_subject   = emb_subj or ""
        trace.embedding_unit      = emb_unit or ""
        trace.embedding_score     = float(emb_score)
        trace.embedding_threshold = float(emb_threshold)
        trace.embedding_passed    = bool(emb_subj)

        # ── Stage 4: LLM router (only run if both keyword and embedding failed) ──
        if not trace.keyword_passed and not trace.embedding_passed:
            llm_res = _llm_classify_subject_unit(question.query)
            trace.llm_subject = llm_res.subject or ""
            trace.llm_unit    = llm_res.unit or ""
            trace.llm_passed  = bool(llm_res.subject)

        # ── Determine winner (mirrors hybrid_router.route logic) ────────────
        if trace.keyword_passed:
            trace.winning_stage = "keyword"
        elif trace.embedding_passed:
            trace.winning_stage = "embedding"
        elif trace.llm_passed:
            trace.winning_stage = "llm"
        else:
            trace.winning_stage = "none"

        result.router_trace  = trace
        result.router_method = trace.winning_stage

        # ── Full pipeline call ──────────────────────────────────────────────
        response = answer_query(
            query=question.query,
            history=[],
            session_subject=None
        )

        execution_time = (time.time() - start_time) * 1000

        result.actual_answer    = response.get("answer", "")
        result.detected_subject = response.get("subject", "")
        result.detected_unit    = response.get("unit", "")
        result.mode             = response.get("mode", "")
        result.expanded_query   = response.get("expanded_query", "")
        result.sources          = response.get("sources", [])
        result.chunks           = response.get("chunks", [])
        result.execution_time_ms = execution_time

        if result.chunks:
            result.top_chunk_score = result.chunks[0].get("final_score", 0)

    except Exception as e:
        result.error = str(e)
        print(f"    [ERROR] Failed: {e}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Run complete system tests for uniAI RAG pipeline"
    )
    parser.add_argument(
        "--question", "-q",
        help="Run specific question ID (e.g., Q1, Q2)"
    )
    parser.add_argument(
        "--subject", "-s",
        help="Filter by subject name (e.g., 'DIGITAL ELECTRONICS')"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=os.path.dirname(__file__),
        help="Output directory for reports (default: same as script)"
    )
    parser.add_argument(
        "--no-rich",
        action="store_true",
        help="Skip rich table output (if rich not installed)"
    )

    args = parser.parse_args()

    # Resolve paths
    questions_path = os.path.join(os.path.dirname(__file__), "questions.txt")
    output_dir = args.output_dir if os.path.isabs(args.output_dir) else os.path.join(os.path.dirname(__file__), args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("UNIAI COMPLETE SYSTEM TEST")
    print("=" * 80)
    print()

    # Check if questions file exists
    if not os.path.exists(questions_path):
        print(f"[ERROR] Questions file not found: {questions_path}")
        sys.exit(1)

    # Parse questions
    print(f"[INFO] Parsing questions from: {questions_path}")
    questions = parse_questions_file(questions_path)
    print(f"[INFO] Parsed {len(questions)} questions")
    print()

    # Filter by question ID if specified
    if args.question:
        questions = [q for q in questions if q.question_id == args.question.upper()]
        if not questions:
            print(f"[ERROR] Question {args.question} not found")
            sys.exit(1)
        print(f"[INFO] Filtered to question: {args.question}")

    # Filter by subject if specified
    if args.subject:
        questions = [q for q in questions if args.subject.upper() in q.subject.upper()]
        if not questions:
            print(f"[ERROR] No questions found for subject: {args.subject}")
            sys.exit(1)
        print(f"[INFO] Filtered to subject: {args.subject}")

    # Show system config
    print("-" * 80)
    print("SYSTEM CONFIGURATION")
    print("-" * 80)
    metadata = get_system_metadata()
    print(f"Chat Model:      {metadata['chat_model']['active']}")
    print(f"Embedding Model: {metadata['embedding_model']['model']}")
    print(f"Reranker Model:  {metadata['reranker']['model']}")
    print(f"Similarity Threshold: {metadata['retrieval']['similarity_threshold']}")
    print(f"Router Threshold:     {metadata['router']['embedding_router_threshold']}")
    print()

    # Run questions
    print("-" * 80)
    print("RUNNING TESTS")
    print("-" * 80)

    results = []
    for i, question in enumerate(questions, 1):
        print(f"\n[{i}/{len(questions)}] {question.question_id} ({question.subject})")
        result = run_single_question(question)
        results.append(result)

        # Brief status
        if result.error:
            print(f"    [FAIL] Error: {result.error}")
        else:
            t = result.router_trace
            print(
                f"    [OK] Winner: {t.winning_stage} | "
                f"kw={'PASS' if t.keyword_passed else 'fail'} | "
                f"emb={'PASS' if t.embedding_passed else 'fail'}({t.embedding_score:.3f}) | "
                f"llm={'PASS' if t.llm_passed else 'fail'} | "
                f"Subject: {result.detected_subject or 'NONE'} | "
                f"Unit: {result.detected_unit or 'NONE'} | "
                f"Time: {result.execution_time_ms:.0f}ms"
            )

    # Generate outputs
    print()
    print("=" * 80)
    print("GENERATING REPORTS")
    print("=" * 80)
    print()

    outputs = generate_all_outputs(results, output_dir)

    print()
    print("=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to:")
    for format_type, path in outputs.items():
        print(f"  [{format_type.upper()}] {path}")
    print()

    # Summary
    total = len(results)
    success = sum(1 for r in results if not r.error)
    subject_matches = sum(1 for r in results
                          if r.detected_subject
                          and r.subject_expected.upper().replace(" ", "_") == r.detected_subject)

    print(f"Summary: {success}/{total} passed, {subject_matches}/{total} correct subject detection")


if __name__ == "__main__":
    main()
