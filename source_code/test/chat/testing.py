import os
import sys
import json
import re

# ------------------------------------------------------------
# Path Setup
# ------------------------------------------------------------

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from rag.rag_pipeline import answer_query


# ------------------------------------------------------------
# Hardcoded Files
# ------------------------------------------------------------

BASE_DIR = os.path.dirname(__file__)
INPUT_FILE = os.path.join(BASE_DIR, "questions.txt")
OUTPUT_FILE = os.path.join(BASE_DIR, "results.jsonl")

SESSION_SUBJECT = "CYBER_SECURITY"   # set to None if you don't want locking


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def load_lines(filename):
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

    for line in lines:

        stripped = line.strip()

        # ---- Detect new section ----
        if stripped.lower().startswith("section"):
            print(f"\n--- New Session: {stripped} ---")
            history = []  # reset session history
            current_section = stripped
            continue

        # ---- Only process numbered questions ----
        if not re.match(r"^\d+\.\s+", stripped):
            continue

        # Extract question text after "1. "
        question = re.sub(r"^\d+\.\s+", "", stripped)

        print(f"Processing: {question}")

        result = answer_query(
            query=question,
            history=history,
            session_subject=SESSION_SUBJECT
        )

        # Update session history
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": result["answer"]})

        top_similarity = None
        top_final_score = None

        if result["chunks"]:
            top_similarity = result["chunks"][0].get("similarity")
            top_final_score = result["chunks"][0].get("final_score")

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
    print(f"\nEvaluation complete. Results written to {OUTPUT_FILE}")


# ------------------------------------------------------------
# Run
# ------------------------------------------------------------

if __name__ == "__main__":
    evaluate()