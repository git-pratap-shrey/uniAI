"""
run_router_tests.py
───────────────────
Batch test runner for the router with debug tracing.
Loops through test cases and saves detailed debug logs.
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path

# Add project root to path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../.."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from source_code.config import CONFIG
from source_code import models
from source_code.tests.router.router_debug import (
    build_debug_trace,
    parse_router_output,
    format_router_output,
)
from source_code.rag.hybrid_router import route as hybrid_route
from source_code.rag.router import detect_subject, _keyword_map
from source_code.rag.unit_router import detect_unit
from source_code.rag.embedding_router import route as embedding_route


def run_full_router_with_debug(query: str) -> dict:
    """
    Run the full router pipeline with debug tracing.

    Args:
        query: The user query string.

    Returns:
        Debug trace dict with all routing stages.
    """
    trace = build_debug_trace()
    trace["query"] = query

    # Log model info at the start
    trace["model_info"] = {
        "router_model": CONFIG["providers"].get("router"),
        "router_model_name": CONFIG["rag"].get("router_model", "default"),
        "temperature": CONFIG["rag"].get("router_temperature", 0.0),
        "max_tokens": CONFIG["rag"].get("router_num_predict", 10),
    }

    # -----------------------------
    # Stage 1: Regex (explicit unit)
    # -----------------------------
    explicit_unit = detect_unit(query)
    trace["regex"] = {
        "output": format_router_output(None, explicit_unit),
        "explicit_unit": explicit_unit,
    }

    # -----------------------------
    # Stage 2: Keyword Router
    # -----------------------------
    keyword_subject = None
    keyword_unit = None
    keyword_used_llm = False

    if _keyword_map:
        query_lower = query.lower()
        from source_code.rag.router import _score_subject

        scores = {
            subject: _score_subject(query_lower, entry)
            for subject, entry in _keyword_map.items()
        }
        max_score = max(scores.values()) if scores else 0

        if max_score >= CONFIG["rag"]["keywords"]["min_score"]:
            top_subjects = [s for s, v in scores.items() if v == max_score]
            if len(top_subjects) == 1:
                keyword_subject = top_subjects[0]
                from source_code.rag.unit_router import score_units

                unit_result = score_units(query_lower, _keyword_map[keyword_subject])
                keyword_unit = unit_result[0] if unit_result else None
                keyword_used_llm = False

    trace["keyword"] = {
        "matched_subject": keyword_subject,
        "matched_unit": keyword_unit,
        "used_llm": keyword_used_llm,
        "output": format_router_output(keyword_subject, keyword_unit),
    }

    # -----------------------------
    # Stage 3: Embedding Router
    # -----------------------------
    emb_subject = None
    emb_unit = None
    emb_score = 0.0

    try:
        emb_subject, emb_unit, emb_score = embedding_route(query)
    except Exception as e:
        trace["embedding"]["error"] = str(e)

    trace["embedding"] = {
        "matched_subject": emb_subject,
        "matched_unit": emb_unit,
        "score": emb_score,
        "threshold": CONFIG["rag"]["embedding_router_threshold"],
        "output": format_router_output(emb_subject, emb_unit),
    }

    # -----------------------------
    # Stage 4: LLM Router
    # -----------------------------
    llm_subject = None
    llm_unit = None
    llm_raw = None

    if _keyword_map:
        # Build subjects_units_list
        subject_units = []
        for subject, entry in _keyword_map.items():
            if isinstance(entry, list):
                subject_units.append(subject)
                continue
            units = set()
            for col_val in entry.values():
                if isinstance(col_val, dict):
                    for u in col_val.keys():
                        if u not in ("unknown", "core"):
                            units.add(u)

            if units:
                for u in sorted(units):
                    subject_units.append(f"{subject}_{u}")
            else:
                subject_units.append(subject)

        subjects_units_str = ", ".join(subject_units)

        from source_code.prompts import subject_unit_router

        prompt = subject_unit_router(query=query, subjects_units_list=subjects_units_str)

        trace["llm"] = {
            "model": CONFIG["rag"].get("router_model", CONFIG["providers"].get("router")),
            "temperature": CONFIG["rag"].get("router_temperature", 0.0),
            "max_tokens": CONFIG["rag"].get("router_num_predict", 10),
            "prompt": prompt,
        }

        try:
            response_text = models.chat(
                prompt=f"{prompt} /no_think",
                system_prompt="You are a helpful assistant. You must respond directly without internal reasoning or <think> tags.",
                model=CONFIG["rag"].get("router_model", CONFIG["providers"].get("router")),
                provider=CONFIG["providers"].get("router", "ollama"),
                temperature=CONFIG["rag"].get("router_temperature", 0.0),
                num_predict=CONFIG["rag"].get("router_num_predict", 10),
            )

            llm_raw = response_text.strip()
            trace["llm"]["raw_output"] = llm_raw

            # Parse the output
            llm_subject, llm_unit = parse_router_output(llm_raw)

            # Try to match with known subjects
            llm_choice = llm_raw.strip().upper().replace(" ", "_")
            for su in subject_units:
                if su.upper() == llm_choice:
                    parts = su.rsplit("_", 1)
                    if len(parts) == 2 and parts[1].isdigit():
                        llm_subject = parts[0]
                        llm_unit = parts[1]
                    else:
                        llm_subject = su
                        llm_unit = None
                    break

        except Exception as e:
            trace["llm"]["error"] = str(e)

    trace["llm"]["parsed"] = {
        "subject": llm_subject,
        "unit": llm_unit,
    }
    trace["llm"]["output"] = format_router_output(llm_subject, llm_unit)

    # -----------------------------
    # Final Decision Logic
    # -----------------------------
    if llm_subject and llm_subject != "none":
        final_subject = llm_subject
        final_unit = llm_unit or explicit_unit
        final_method = "llm"
    elif emb_score > 0.7:
        final_subject = emb_subject
        final_unit = explicit_unit or emb_unit
        final_method = "embedding"
    elif keyword_subject:
        final_subject = keyword_subject
        final_unit = explicit_unit or keyword_unit
        final_method = "keyword"
    else:
        final_subject = None
        final_unit = explicit_unit
        final_method = "none"

    trace["final_decision"] = {
        "subject": final_subject,
        "unit": final_unit,
        "method": final_method,
        "output": format_router_output(final_subject, final_unit),
    }

    return trace


def main():
    """Run the batch router tests."""
    test_file = os.path.join(CURRENT_DIR, "router_test_cases_debug.json")

    if not os.path.exists(test_file):
        print(f"Test file not found: {test_file}")
        return

    with open(test_file, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    # Create logs directory
    logs_dir = os.path.join(CURRENT_DIR, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = []

    print("=" * 60)
    print("ROUTER DEBUG TEST RUN")
    print("=" * 60)
    print(f"Model: {CONFIG['providers'].get('router')}")
    print(f"Temperature: {CONFIG['rag'].get('router_temperature', 0.0)}")
    print(f"Max Tokens: {CONFIG['rag'].get('router_num_predict', 10)}")
    print("=" * 60)

    for i, case in enumerate(test_cases):
        query = case.get("query")
        print(f"\n[{i + 1}] Query: {query}")

        trace = run_full_router_with_debug(query)
        results.append(trace)

        # Print key results
        final = trace.get("final_decision", {})
        print(f"    → Final: {final.get('output')} (method: {final.get('method')})")
        print(f"    → LLM Raw: {trace.get('llm', {}).get('raw_output', 'N/A')}")

        # Save individual trace
        trace_file = os.path.join(logs_dir, f"trace_{timestamp}_{i + 1}.json")
        with open(trace_file, "w", encoding="utf-8") as f:
            json.dump(trace, f, indent=2)

    # Save full results
    results_file = os.path.join(logs_dir, f"results_{timestamp}.json")
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Results saved to: {results_file}")
    print("=" * 60)

    # Summary
    method_counts = {}
    for trace in results:
        method = trace.get("final_decision", {}).get("method", "none")
        method_counts[method] = method_counts.get(method, 0) + 1

    print("\nMETHOD USAGE:")
    for method, count in method_counts.items():
        print(f"  {method}: {count}/{len(test_cases)}")


if __name__ == "__main__":
    main()