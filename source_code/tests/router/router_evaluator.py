import os
import sys
import json

# -------------------------------------------------
# Fix import path
# -------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../"))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rag.hybrid_router import route

# -------------------------------------------------
# Evaluation
# -------------------------------------------------
def evaluate_router(test_file=None):

    if test_file is None:
        test_file = os.path.join(CURRENT_DIR, "router_test_cases.json")

    with open(test_file, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    total = len(test_cases)
    
    overall_correct = 0
    subject_correct = 0
    unit_correct = 0
    unit_total = 0
    
    methods = {"keyword": 0, "embedding": 0, "llm": 0, "none": 0}

    for case in test_cases:
        question = case.get("question")
        expected_route = case.get("expected_route")
        
        expected_subject = case.get("expected_subject")
        expected_unit = case.get("expected_unit")

        res = route(question)
        
        predicted_route = "SYLLABUS" if res.subject else "GENERIC"
        
        if expected_route and predicted_route == expected_route:
            overall_correct += 1
            
        methods[res.method] = methods.get(res.method, 0) + 1
            
        if expected_subject:
            if res.subject == expected_subject:
                subject_correct += 1
                
        if expected_unit and expected_subject:
            unit_total += 1
            if res.unit == str(expected_unit) and res.subject == expected_subject:
                unit_correct += 1
            else:
                print(f"\nMismatch (Subject/Unit):")
                print(f"Question: {question}")
                print(f"Expected: {expected_subject}_{expected_unit}, Got: {res.subject}_{res.unit} (Method: {res.method})")
                print("-" * 40)

    print("\n========== OVERALL ROUTE ==========")
    if total > 0:
        print(f"Accuracy: {(overall_correct / total) * 100:.2f}% ({overall_correct}/{total})")

    subject_cases_count = sum(1 for c in test_cases if "expected_subject" in c)
    if subject_cases_count > 0:
        print("\n========== SUBJECT ACCURACY ==========")
        print(f"Accuracy: {(subject_correct / subject_cases_count) * 100:.2f}% ({subject_correct}/{subject_cases_count})")
        
    if unit_total > 0:
        print("\n========== UNIT ACCURACY ==========")
        print(f"Accuracy: {(unit_correct / unit_total) * 100:.2f}% ({unit_correct}/{unit_total})")

    print("\n========== ROUTING METHODS ==========")
    for m, count in methods.items():
        if count > 0:
            print(f"{m}: {count}/{total} ({(count / total) * 100:.2f}%)")

if __name__ == "__main__":
    evaluate_router()