import os
import sys
import json

# -------------------------------------------------
# Fix import path (because test/ is inside source_code/)
# -------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../"))

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from rag.router import detect_subject


# -------------------------------------------------
# Real router wrapper
# -------------------------------------------------
def real_router(question):
    """
    Returns:
        route (str): "SYLLABUS" or "GENERIC"
        used_llm (bool): whether LLM fallback was triggered
    """
    subject, used_llm = detect_subject(question, debug=True)
    route = "SYLLABUS" if subject else "GENERIC"
    return route, used_llm


# -------------------------------------------------
# Evaluation Function
# -------------------------------------------------
def evaluate_router(router_function, test_file=None):

    if test_file is None:
        test_file = os.path.join(CURRENT_DIR, "router_test_cases.json")

    with open(test_file, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    total = len(test_cases)
    correct = 0

    category_stats = {
        "SYLLABUS": {"total": 0, "correct": 0},
        "GENERIC": {"total": 0, "correct": 0},
    }

    llm_fallback_count = 0

    for case in test_cases:
        question = case["question"]
        expected = case["expected_route"]

        predicted, used_llm = router_function(question)

        if used_llm:
            llm_fallback_count += 1

        category_stats[expected]["total"] += 1

        if predicted == expected:
            correct += 1
            category_stats[expected]["correct"] += 1
        else:
            print(f"Mismatch: {question}")
            print(f"Expected: {expected}, Got: {predicted}")
            print("-" * 40)

    accuracy = (correct / total) * 100

    print("\n========== OVERALL ==========")
    print(f"Accuracy: {accuracy:.2f}% ({correct}/{total})")

    print("\n========== PER CATEGORY ==========")
    for category, stats in category_stats.items():
        if stats["total"] > 0:
            cat_acc = (stats["correct"] / stats["total"]) * 100
            print(
                f"{category}: {cat_acc:.2f}% "
                f"({stats['correct']}/{stats['total']})"
            )

    print("\n========== LLM FALLBACK ==========")
    print(
        f"LLM used in {llm_fallback_count}/{total} cases "
        f"({(llm_fallback_count / total) * 100:.2f}%)"
    )

    return accuracy


# -------------------------------------------------
# Entry Point
# -------------------------------------------------
if __name__ == "__main__":
    evaluate_router(real_router)