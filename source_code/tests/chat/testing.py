import os
import sys
import json
import re
from collections import Counter

# ------------------------------------------------------------
# Path Setup (robust)
# ------------------------------------------------------------

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from rag.rag_pipeline import answer_query


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

BASE_DIR = os.path.dirname(__file__)
INPUT_FILE = os.path.join(BASE_DIR, "questions.txt")
OUTPUT_FILE = os.path.join(BASE_DIR, "results.jsonl")

# Set to None if you want subject auto-detection
SESSION_SUBJECT = "CYBER_SECURITY"


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def load_lines(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Input file not found: {filename}")
    with open(filename, "r", encoding="utf-8") as f:
        return [line.rstrip() for line in f]


def write_jsonl(filename, records):
    with open(filename, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------

def evaluate():
    lines = load_lines(INPUT_FILE)
    results = []

    history = []
    current_section = None

    mode_counter = Counter()
    zero_chunk_count = 0

    for line in lines:

        stripped = line.strip()

        # ---- Detect new section ----
        if stripped.lower().startswith("section"):
            print(f"\n--- New Session: {stripped} ---")
            history = []
            current_section = stripped
            continue

        # ---- Only process numbered questions ----
        if not re.match(r"^\d+\.\s+", stripped):
            continue

        question = re.sub(r"^\d+\.\s+", "", stripped)
        print(f"Processing: {question}")

        result = answer_query(
            query=question,
            history=history,
            session_subject=SESSION_SUBJECT
        )

        # Update history (simulated conversation)
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": result["answer"]})

        mode_counter[result["mode"]] += 1

        if not result["chunks"]:
            zero_chunk_count += 1

        top_similarity = None
        top_final_score = None

        if result["chunks"]:
            top_chunk = result["chunks"][0]
            top_similarity = top_chunk.get("similarity")
            top_final_score = top_chunk.get("final_score")

        record = {
            "section": current_section,
            "question": question,
            "mode": result["mode"],
            "subject": result["subject"],
            "unit": result["unit"],
            "num_chunks": len(result["chunks"]),
            "top_similarity": top_similarity,
            "top_final_score": top_final_score,
            "answer": result["answer"],
        }

        results.append(record)

    write_jsonl(OUTPUT_FILE, results)

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------

    print("\n================ Evaluation Summary ================")
    print(f"Total Questions: {len(results)}")
    print(f"Mode Distribution: {dict(mode_counter)}")
    print(f"Zero-Chunk Responses: {zero_chunk_count}")
    print(f"Results written to: {OUTPUT_FILE}")
    print("====================================================")


# ------------------------------------------------------------
# Run
# ------------------------------------------------------------

if __name__ == "__main__":
    evaluate()