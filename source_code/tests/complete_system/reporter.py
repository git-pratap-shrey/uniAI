"""
reporter.py
───────────
Generate comparison outputs in multiple formats:
- Rich table for terminal display
- JSON for programmatic comparison
- Side-by-side text report
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Any
from dataclasses import dataclass, asdict


def load_subject_aliases() -> dict:
    """Load subject aliases from JSON file and create reverse mapping."""
    alias_path = os.path.join(os.path.dirname(__file__), "../../data/subject_aliases.json")
    try:
        with open(alias_path, 'r') as f:
            aliases = json.load(f)
        # Create reverse mapping: alias -> canonical name
        reverse_map = {}
        for canonical, alias_list in aliases.items():
            canonical_normalized = canonical.upper().replace(" ", "_")
            reverse_map[canonical_normalized] = canonical_normalized
            for alias in alias_list:
                alias_normalized = alias.upper().replace(" ", "_")
                reverse_map[alias_normalized] = canonical_normalized
        return reverse_map
    except Exception as e:
        print(f"[WARNING] Failed to load subject aliases: {e}")
        return {}


_SUBJECT_ALIASES = load_subject_aliases()


def normalize_subject_name(name: str) -> str:
    """Normalize subject name using alias mapping."""
    if not name:
        return ""
    normalized = name.upper().replace(" ", "_")
    return _SUBJECT_ALIASES.get(normalized, normalized)


@dataclass
class RouterStageTrace:
    """Captures the raw output of every stage in the hybrid router."""
    # Stage 1 – Regex unit detection
    regex_unit: str = ""                  # e.g. "3" or "" if not found

    # Stage 2 – Keyword router
    keyword_subject: str = ""             # subject returned (or "")
    keyword_unit: str = ""                # unit returned (or "")
    keyword_used_llm: bool = False        # True when keyword stage fell back to LLM internally
    keyword_passed: bool = False          # True when keyword stage produced a subject

    # Stage 3 – Embedding router
    embedding_subject: str = ""           # subject returned (or "")
    embedding_unit: str = ""              # unit returned (or "")
    embedding_score: float = 0.0          # raw cosine similarity score
    embedding_threshold: float = 0.0      # threshold used
    embedding_passed: bool = False        # True when score exceeded threshold

    # Stage 4 – LLM router
    llm_subject: str = ""                 # subject returned (or "")
    llm_unit: str = ""                    # unit returned (or "")
    llm_passed: bool = False              # True when LLM returned a subject

    # Winner
    winning_stage: str = ""               # "keyword" | "embedding" | "llm" | "none"


@dataclass
class TestResult:
    """Stores the result of running a single question through the RAG pipeline."""
    question_id: str
    subject_expected: str
    subject_code: str
    query: str
    expected_answer: str

    # RAG Pipeline outputs
    actual_answer: str = ""
    detected_subject: str = ""
    detected_unit: str = ""
    mode: str = ""
    expanded_query: str = ""
    sources: List[Dict] = None
    chunks: List[Dict] = None

    # Router stage tracking
    router_method: str = ""               # "keyword" | "embedding" | "llm" | "none"
    router_trace: RouterStageTrace = None # full per-stage breakdown

    # Metadata
    execution_time_ms: float = 0.0
    top_chunk_score: float = 0.0
    error: str = ""

    def __post_init__(self):
        if self.sources is None:
            self.sources = []
        if self.chunks is None:
            self.chunks = []
        if self.router_trace is None:
            self.router_trace = RouterStageTrace()


def get_system_metadata() -> Dict[str, Any]:
    """
    Collect full system configuration metadata.
    Returns models, thresholds, and all hyperparameters.
    """
    import sys
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)

    from source_code.config import CONFIG
    from source_code.config.models import (
        MODEL_CONFIGS, ACTIVE_CHAT_MODEL, EMBEDDING_CONFIG,
        ROUTER_CONFIG, VISION_CONFIG
    )

    return {
        "timestamp": datetime.now().isoformat(),
        "chat_model": {
            "active": ACTIVE_CHAT_MODEL,
            "config": MODEL_CONFIGS.get(ACTIVE_CHAT_MODEL, {})
        },
        "embedding_model": EMBEDDING_CONFIG,
        "router_model": ROUTER_CONFIG,
        "vision_model": VISION_CONFIG,
        "reranker": {
            "model": CONFIG["rag"]["cross_encoder"]["model"],
            "min_score": CONFIG["rag"]["cross_encoder"]["min_score"],
            "candidates": CONFIG["rag"]["cross_encoder"]["candidates"],
            "pipeline_top_n": CONFIG["rag"]["cross_encoder"]["pipeline_top_n"]
        },
        "retrieval": {
            "similarity_threshold": CONFIG["rag"]["similarity_threshold"],
            "min_strong_sim": CONFIG["rag"]["min_strong_sim"],
            "notes_k": CONFIG["rag"]["notes_k"],
            "syllabus_k": CONFIG["rag"]["syllabus_k"],
            "history_limit": CONFIG["rag"]["history_limit"]
        },
        "router": {
            "keywords_min_score": CONFIG["rag"]["keywords"]["min_score"],
            "embedding_router_threshold": CONFIG["rag"]["embedding_router_threshold"]
        }
    }


def generate_rich_table(results: List[TestResult]) -> None:
    """
    Generate a rich terminal table with comparison data.
    """
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
        from rich import box
    except ImportError:
        print("[WARNING] rich library not installed. Skipping rich table output.")
        print("Install with: pip install rich")
        return

    console = Console()

    # Summary header
    total = len(results)
    success = sum(1 for r in results if not r.error)
    errors = sum(1 for r in results if r.error)
    syllabus_mode = sum(1 for r in results if r.mode == "syllabus")
    generic_mode = sum(1 for r in results if r.mode == "generic")

    console.print(Panel.fit(
        f"[bold green]✓ {success}[/] passed | [bold red]✗ {errors}[/] errors | "
        f"[bold blue]{syllabus_mode}[/] syllabus mode | [bold yellow]{generic_mode}[/] generic mode",
        title="Test Results Summary",
        border_style="blue"
    ))
    console.print()

    # Results table
    table = Table(
        title="RAG Pipeline Test Results",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta"
    )

    table.add_column("Q#", style="cyan", width=4)
    table.add_column("Subject", style="green", width=12)
    table.add_column("Detected", style="yellow", width=12)
    table.add_column("Unit", style="blue", width=5)
    table.add_column("Mode", style="magenta", width=8)
    table.add_column("Router", style="white", width=10)
    table.add_column("Query (truncated)", style="white", width=30)
    table.add_column("Score", style="cyan", width=6)
    table.add_column("Status", style="bold", width=8)

    for r in results:
        # Determine status
        if r.error:
            status = "[red]ERROR"
            score_str = "N/A"
        elif r.mode == "generic":
            status = "[yellow]GENERIC"
            score_str = f"{r.top_chunk_score:.2f}" if r.top_chunk_score else "N/A"
        elif r.detected_subject and normalize_subject_name(r.subject_expected) != normalize_subject_name(r.detected_subject):
            status = "[yellow]MISMATCH"
            score_str = f"{r.top_chunk_score:.2f}"
        else:
            status = "[green]OK"
            score_str = f"{r.top_chunk_score:.2f}"

        # Truncate query
        query_display = r.query[:37] + "..." if len(r.query) > 40 else r.query

        # Subject mismatch indicator
        expected_subj = r.subject_expected[:10]
        detected_subj = (r.detected_subject[:10] if r.detected_subject else "--")

        table.add_row(
            r.question_id,
            expected_subj,
            detected_subj,
            r.detected_unit or "--",
            r.mode or "--",
            query_display,
            score_str,
            status
        )

    console.print(table)
    console.print()

    # Router stage breakdown table
    router_table = Table(
        title="Router Stage Breakdown",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan"
    )
    router_table.add_column("Q#",       style="cyan",    width=4)
    router_table.add_column("Regex\nUnit", style="white", width=6)
    router_table.add_column("Keyword\nSubject", style="green", width=14)
    router_table.add_column("Keyword\nUnit",    style="green", width=8)
    router_table.add_column("Keyword\nPass?",   style="green", width=8)
    router_table.add_column("Emb\nSubject",     style="yellow", width=14)
    router_table.add_column("Emb\nScore",       style="yellow", width=7)
    router_table.add_column("Emb\nThresh",      style="yellow", width=7)
    router_table.add_column("Emb\nPass?",       style="yellow", width=7)
    router_table.add_column("LLM\nSubject",     style="magenta", width=14)
    router_table.add_column("LLM\nPass?",       style="magenta", width=7)
    router_table.add_column("Winner",           style="bold white", width=10)

    for r in results:
        t = r.router_trace
        kw_pass  = "[green]YES" if t.keyword_passed  else "[red]NO"
        emb_pass = "[green]YES" if t.embedding_passed else "[red]NO"
        llm_pass = "[green]YES" if t.llm_passed       else "[red]NO"

        router_table.add_row(
            r.question_id,
            t.regex_unit or "--",
            t.keyword_subject[:12] or "--",
            t.keyword_unit or "--",
            kw_pass,
            t.embedding_subject[:12] or "--",
            f"{t.embedding_score:.3f}",
            f"{t.embedding_threshold:.3f}",
            emb_pass,
            t.llm_subject[:12] or "--",
            llm_pass,
            t.winning_stage or "--",
        )

    console.print(router_table)
    console.print()

    # Detailed answers table
    detail_table = Table(
        title="Answer Comparison",
        box=box.SIMPLE,
        show_header=True,
        header_style="bold cyan"
    )

    detail_table.add_column("Q#", style="cyan", width=4)
    detail_table.add_column("Expected Answer", style="green", width=50)
    detail_table.add_column("Actual Answer", style="yellow", width=50)

    for r in results:
        exp = r.expected_answer[:47] + "..." if len(r.expected_answer) > 50 else r.expected_answer
        act = r.actual_answer[:47] + "..." if len(r.actual_answer) > 50 else r.actual_answer
        detail_table.add_row(r.question_id, exp, act)

    console.print(detail_table)
    console.print()


def generate_json_output(results: List[TestResult], metadata: Dict[str, Any], output_path: str) -> None:
    """
    Generate JSON output with full results and metadata.
    """
    output = {
        "metadata": metadata,
        "results": []
    }

    for r in results:
        result_dict = {
            "question_id": r.question_id,
            "subject": {
                "expected": r.subject_expected,
                "expected_code": r.subject_code,
                "detected": r.detected_subject
            },
            "query": r.query,
            "expanded_query": r.expanded_query,
            "expected_answer": r.expected_answer,
            "actual_answer": r.actual_answer,
            "unit_detected": r.detected_unit,
            "mode": r.mode,
            "top_chunk_score": r.top_chunk_score,
            "execution_time_ms": r.execution_time_ms,
            "error": r.error,
            "sources": r.sources,
            "chunks_retrieved": len(r.chunks),
            "router_trace": {
                "regex_unit":           r.router_trace.regex_unit,
                "keyword_subject":      r.router_trace.keyword_subject,
                "keyword_unit":         r.router_trace.keyword_unit,
                "keyword_used_llm":     r.router_trace.keyword_used_llm,
                "keyword_passed":       r.router_trace.keyword_passed,
                "embedding_subject":    r.router_trace.embedding_subject,
                "embedding_unit":       r.router_trace.embedding_unit,
                "embedding_score":      round(r.router_trace.embedding_score, 4),
                "embedding_threshold":  round(r.router_trace.embedding_threshold, 4),
                "embedding_passed":     r.router_trace.embedding_passed,
                "llm_subject":          r.router_trace.llm_subject,
                "llm_unit":             r.router_trace.llm_unit,
                "llm_passed":           r.router_trace.llm_passed,
                "winning_stage":        r.router_trace.winning_stage,
            }
        }
        output["results"].append(result_dict)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"[INFO] JSON results saved to: {output_path}")


def generate_text_report(results: List[TestResult], metadata: Dict[str, Any], output_path: str) -> None:
    """
    Generate a side-by-side text report.
    """
    lines = []
    lines.append("=" * 120)
    lines.append("UNIAI COMPLETE SYSTEM TEST REPORT")
    lines.append("=" * 120)
    lines.append(f"Generated: {metadata['timestamp']}")
    lines.append("")

    # System config section
    lines.append("-" * 120)
    lines.append("SYSTEM CONFIGURATION")
    lines.append("-" * 120)
    lines.append(f"Chat Model:      {metadata['chat_model']['active']}")
    lines.append(f"  → Config:     {metadata['chat_model']['config']}")
    lines.append(f"Embedding Model: {metadata['embedding_model']['model']}")
    lines.append(f"Reranker Model:  {metadata['reranker']['model']}")
    lines.append(f"  → Min Score:  {metadata['reranker']['min_score']}")
    lines.append(f"Similarity Threshold: {metadata['retrieval']['similarity_threshold']}")
    lines.append(f"Router Threshold:     {metadata['router']['embedding_router_threshold']}")
    lines.append("")

    # Results section
    lines.append("-" * 120)
    lines.append("TEST RESULTS (Side-by-Side Comparison)")
    lines.append("-" * 120)
    lines.append("")

    for r in results:
        lines.append(f"[{r.question_id}] {r.subject_expected} ({r.subject_code})")
        lines.append(f"Query: {r.query}")
        lines.append(f"Expanded: {r.expanded_query}")
        lines.append(f"Detected Subject: {r.detected_subject or 'NONE'} | Unit: {r.detected_unit or 'NONE'} | Mode: {r.mode or 'NONE'}")
        if r.error:
            lines.append(f"ERROR: {r.error}")
        lines.append("")

        # Router trace
        t = r.router_trace
        lines.append("  ROUTER TRACE:")
        lines.append(f"    [1] Regex      → unit={t.regex_unit or 'none'}")
        kw_status = "PASS" if t.keyword_passed else "FAIL"
        lines.append(f"    [2] Keyword    → [{kw_status}] subject={t.keyword_subject or 'none'} | unit={t.keyword_unit or 'none'} | used_llm={t.keyword_used_llm}")
        emb_status = "PASS" if t.embedding_passed else "FAIL"
        lines.append(f"    [3] Embedding  → [{emb_status}] subject={t.embedding_subject or 'none'} | unit={t.embedding_unit or 'none'} | score={t.embedding_score:.4f} (threshold={t.embedding_threshold:.4f})")
        llm_status = "PASS" if t.llm_passed else "FAIL"
        lines.append(f"    [4] LLM        → [{llm_status}] subject={t.llm_subject or 'none'} | unit={t.llm_unit or 'none'}")
        lines.append(f"    [→] Winner     → {t.winning_stage or 'none'}")
        lines.append("")

        # Side-by-side expected vs actual
        lines.append("-" * 60 + " EXPECTED " + "-" * 50)
        lines.append(_wrap_text(r.expected_answer, 118))
        lines.append("")
        lines.append("-" * 60 + " ACTUAL " + "-" * 52)
        lines.append(_wrap_text(r.actual_answer, 118))
        lines.append("")
        lines.append(f"Top Chunk Score: {r.top_chunk_score:.3f} | Execution Time: {r.execution_time_ms:.1f}ms")
        lines.append("=" * 120)
        lines.append("")

    # Summary stats
    lines.append("")
    lines.append("-" * 120)
    lines.append("SUMMARY STATISTICS")
    lines.append("-" * 120)
    total = len(results)
    success = sum(1 for r in results if not r.error)
    subject_matches = sum(1 for r in results
                          if r.detected_subject
                          and normalize_subject_name(r.subject_expected) == normalize_subject_name(r.detected_subject))
    lines.append(f"Total Questions: {total}")
    lines.append(f"Successful: {success} ({100*success/total:.1f}%)")
    lines.append(f"Subject Detection Accuracy: {subject_matches}/{total} ({100*subject_matches/total:.1f}%)")
    lines.append("=" * 120)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f"[INFO] Text report saved to: {output_path}")


def _wrap_text(text: str, width: int) -> str:
    """Simple text wrapper for report formatting."""
    if not text:
        return ""
    words = text.split()
    lines = []
    current_line = ""

    for word in words:
        if len(current_line) + len(word) + 1 <= width:
            current_line += (" " if current_line else "") + word
        else:
            lines.append(current_line)
            current_line = word

    if current_line:
        lines.append(current_line)

    return '\n'.join(lines)


def generate_all_outputs(results: List[TestResult], output_dir: str) -> Dict[str, str]:
    """
    Generate all three output formats.

    Returns:
        Dict with paths to generated files.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metadata = get_system_metadata()

    outputs = {}

    # Rich table (terminal only)
    generate_rich_table(results)

    # JSON
    json_path = os.path.join(output_dir, f"results_{timestamp}.json")
    generate_json_output(results, metadata, json_path)
    outputs['json'] = json_path

    # Text report
    report_path = os.path.join(output_dir, f"report_{timestamp}.txt")
    generate_text_report(results, metadata, report_path)
    outputs['report'] = report_path

    return outputs


if __name__ == "__main__":
    # Test with dummy data
    test_results = [
        TestResult(
            question_id="Q1",
            subject_expected="DIGITAL ELECTRONICS",
            subject_code="BEC301",
            query="What is a gate?",
            expected_answer="A logic gate is...",
            actual_answer="A logic gate is a device...",
            detected_subject="DIGITAL_ELECTRONICS",
            detected_unit="1",
            mode="syllabus",
            top_chunk_score=0.85
        )
    ]

    generate_all_outputs(test_results, ".")
