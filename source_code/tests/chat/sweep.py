"""
sweep.py
────────
Automated threshold sweep for cross-encoder MIN_CROSS_SCORE calibration.

Runs all questions from questions.txt at multiple threshold values and
produces per-threshold JSONL results + a comparison summary table.

Usage:
    cd /home/anon/PROJECTS/uniAI/source_code
    python tests/chat/sweep.py
"""

import os
import sys
import json
import re
import time
from collections import Counter

# ------------------------------------------------------------
# Path Setup
# ------------------------------------------------------------

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import config
from rag.rag_pipeline import answer_query

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

BASE_DIR = os.path.dirname(__file__)
INPUT_FILE = os.path.join(BASE_DIR, "questions.txt")

SESSION_SUBJECT = "CYBER_SECURITY"

# Thresholds to sweep
THRESHOLDS = [0.60, 0.65, 0.70]


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def load_questions(filename):
    """Load numbered questions from file, returning (section, question) tuples."""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Input file not found: {filename}")

    with open(filename, "r", encoding="utf-8") as f:
        lines = [line.rstrip() for line in f]

    questions = []
    current_section = None

    for line in lines:
        stripped = line.strip()

        if stripped.lower().startswith("section"):
            current_section = stripped
            continue

        if not re.match(r"^\d+\.\s+", stripped):
            continue

        question = re.sub(r"^\d+\.\s+", "", stripped)
        questions.append((current_section, question))

    return questions


def write_jsonl(filename, records):
    with open(filename, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ------------------------------------------------------------
# Single-threshold evaluation
# ------------------------------------------------------------

def run_single_threshold(questions, threshold):
    """Run all questions at a given threshold and return results."""
    # Override the config threshold
    config.MIN_CROSS_SCORE = threshold

    results = []
    history = []
    current_section = None

    for section, question in questions:
        # Reset history on new section
        if section != current_section:
            history = []
            current_section = section

        result = answer_query(
            query=question,
            history=history,
            session_subject=SESSION_SUBJECT,
        )

        # Update history
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": result["answer"]})

        top_similarity = None
        top_final_score = None
        top_cross_raw = None

        if result["chunks"]:
            top = result["chunks"][0]
            top_similarity = top.get("similarity")
            top_final_score = top.get("final_score")
            top_cross_raw = top.get("cross_score_raw")

        record = {
            "section": section,
            "question": question,
            "mode": result["mode"],
            "subject": result["subject"],
            "unit": result["unit"],
            "num_chunks": len(result["chunks"]),
            "top_similarity": top_similarity,
            "top_final_score": top_final_score,
            "top_cross_raw": top_cross_raw,
        }

        results.append(record)

    return results


# ------------------------------------------------------------
# Summary statistics
# ------------------------------------------------------------

def compute_stats(results):
    """Compute summary statistics from results."""
    total = len(results)
    mode_counter = Counter(r["mode"] for r in results)
    zero_chunk = sum(1 for r in results if r["num_chunks"] == 0)
    scores = [r["top_final_score"] for r in results if r["top_final_score"] is not None]

    return {
        "total": total,
        "syllabus": mode_counter.get("syllabus", 0),
        "generic": mode_counter.get("generic", 0),
        "zero_chunk": zero_chunk,
        "avg_top_score": round(sum(scores) / len(scores), 4) if scores else 0,
        "fallback_pct": round(zero_chunk / total * 100, 1) if total else 0,
        "syllabus_pct": round(mode_counter.get("syllabus", 0) / total * 100, 1) if total else 0,
    }


# ------------------------------------------------------------
# Main sweep
# ------------------------------------------------------------

def sweep():
    questions = load_questions(INPUT_FILE)
    print(f"Loaded {len(questions)} questions from {INPUT_FILE}")

    all_stats = {}

    for threshold in THRESHOLDS:
        print(f"\n{'='*60}")
        print(f"  Threshold: {threshold}")
        print(f"{'='*60}")

        start = time.time()
        results = run_single_threshold(questions, threshold)
        elapsed = time.time() - start

        # Write per-threshold results
        out_file = os.path.join(BASE_DIR, f"sweep_{threshold:.2f}.jsonl")
        write_jsonl(out_file, results)
        print(f"  Results → {out_file} ({elapsed:.1f}s)")

        stats = compute_stats(results)
        stats["elapsed_s"] = round(elapsed, 1)
        all_stats[threshold] = stats

    # ---- Summary table ----
    print(f"\n{'='*70}")
    print("  THRESHOLD SWEEP SUMMARY")
    print(f"{'='*70}")
    print(f"{'Threshold':>10} | {'Syllabus':>8} | {'Generic':>8} | {'Fallback':>8} | {'Avg Score':>10} | {'Time':>6}")
    print(f"{'-'*10}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*10}-+-{'-'*6}")

    for t in THRESHOLDS:
        s = all_stats[t]
        print(
            f"{t:>10.2f} | "
            f"{s['syllabus']:>5} ({s['syllabus_pct']:>4.1f}%) | "
            f"{s['generic']:>5} | "
            f"{s['zero_chunk']:>5} ({s['fallback_pct']:>4.1f}%) | "
            f"{s['avg_top_score']:>10.4f} | "
            f"{s['elapsed_s']:>5.1f}s"
        )

    print(f"{'='*70}")

    # Write summary JSON
    summary_file = os.path.join(BASE_DIR, "sweep_summary.json")
    with open(summary_file, "w") as f:
        json.dump({str(k): v for k, v in all_stats.items()}, f, indent=2)
    print(f"\nSummary JSON → {summary_file}")


if __name__ == "__main__":
    sweep()
