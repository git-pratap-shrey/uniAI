"""
Debug script for UHV Q5 routing issue.
"""

import os
import sys

# Add project root to sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from source_code.rag.router import detect_subject
from source_code.rag.unit_router import score_units
import json

# Load keyword map
with open("source_code/data/subject_keywords.json", "r") as f:
    keyword_map = json.load(f)

# UHV Q5
query = "Define 'Prosperity'. How is it different from 'Wealth'?"
print(f"Query: {query}\n")

# Test subject detection
subject, unit, used_llm = detect_subject(query, debug=True)
print(f"Detected Subject: {subject}")
print(f"Detected Unit: {unit}")
print(f"Used LLM: {used_llm}\n")

# Manual unit scoring
query_lower = query.lower()
uhv_entry = keyword_map.get("UHV", {})

print("Manual Unit Scoring:")
print("=" * 60)

# Check Unit 1 keywords
unit1_notes_keywords = uhv_entry.get("notes", {}).get("1", [])
unit1_syllabus_keywords = uhv_entry.get("syllabus", {}).get("1", [])
print(f"\nUnit 1 notes keywords: {unit1_notes_keywords}")
unit1_notes_matches = [kw for kw in unit1_notes_keywords if kw in query_lower]
print(f"Unit 1 notes matches: {unit1_notes_matches}")
print(f"\nUnit 1 syllabus keywords: {unit1_syllabus_keywords}")
unit1_syllabus_matches = [kw for kw in unit1_syllabus_keywords if kw in query_lower]
print(f"Unit 1 syllabus matches: {unit1_syllabus_matches}")

# Check Unit 2 keywords
unit2_notes_keywords = uhv_entry.get("notes", {}).get("2", [])
unit2_syllabus_keywords = uhv_entry.get("syllabus", {}).get("2", [])
print(f"\nUnit 2 notes keywords: {unit2_notes_keywords}")
unit2_notes_matches = [kw for kw in unit2_notes_keywords if kw in query_lower]
print(f"Unit 2 notes matches: {unit2_notes_matches}")
print(f"\nUnit 2 syllabus keywords: {unit2_syllabus_keywords}")
unit2_syllabus_matches = [kw for kw in unit2_syllabus_keywords if kw in query_lower]
print(f"Unit 2 syllabus matches: {unit2_syllabus_matches}")

# Score units
unit_result = score_units(query_lower, uhv_entry)
if unit_result:
    best_unit, best_score = unit_result
    print(f"\nBest unit from score_units: {best_unit} (score: {best_score})")
else:
    print("\nNo unit scored")
