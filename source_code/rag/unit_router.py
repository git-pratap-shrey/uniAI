"""
unit_router.py
──────────────
Responsible for identifying which unit of a subject the user is asking about.
It uses a two-stage approach:
1. Regex matching for explicit mentions (e.g., "unit 3").
2. Weighted keyword scoring against the subject's internal structure.

This module is a key part of the RAG pipeline's 'Routing' stage, narrowing
down retrieval to relevant syllabus sections.
"""

import re

# Scoring weights (must match router.py's unit-level weights)
_WEIGHTS = {
    ("notes",    "unit"):    4,
    ("syllabus", "unit"):    3,
}

def detect_unit(query: str) -> str | None:
    """
    Extract the unit number from the query using regular expressions.

    Args:
        query: The raw user query string.

    Returns:
        A numeric string representing the unit (e.g., "1", "3", "5")
        if a match is found, otherwise None.
    """
    match = re.search(r"\bunit[\s\-]*([1-9]\d*)\b", query.lower())
    if match:
        return match.group(1)
    return None

def score_units(query_lower: str, subject_entry) -> tuple[str, float] | None:
    """
    Score all units within a subject based on keyword frequency in the query.

    Args:
        query_lower: The pre-normalized (lowercase) user query.
        subject_entry: The subject's entry from the master keyword map.

    Returns:
        A tuple of (best_unit_label, total_score) or None if no units match.
    """
    if isinstance(subject_entry, list):
        return None  # Legacy format lacks units
        
    scores = {}
    
    for collection, collection_val in subject_entry.items():
        if collection == "pyq" or not isinstance(collection_val, dict):
            continue
            
        for unit_label, kws in collection_val.items():
            if unit_label in ("unknown", "core"):
                continue
                
            w = _WEIGHTS.get((collection, "unit"), 1)
            
            if isinstance(kws, list):
                match_count = sum(1 for kw in kws if kw in query_lower)
                if match_count > 0:
                    scores[unit_label] = scores.get(unit_label, 0.0) + (match_count * w)
                    
    if not scores:
        return None
        
    best_unit, best_score = max(scores.items(), key=lambda x: x[1])
    return best_unit, best_score

def format_unit_filter(unit: str) -> str:
    """
    Convert a detected unit identifier into the format expected by ChromaDB.

    Currently, this is a pass-through as both storage and routing
    standardized on plain numeric strings ("1", "2", etc.).
    """
    return unit
