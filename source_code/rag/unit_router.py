"""
unit_router.py
────────────────
Infers unit number from a query string, either via explicit regex
or by scoring keyword matches against a subject's dictionary.
"""

import re

# Scoring weights (must match router.py's unit-level weights)
_WEIGHTS = {
    ("notes",    "unit"):    4,
    ("syllabus", "unit"):    3,
}

def detect_unit(query: str) -> str | None:
    """
    Extract the unit number from the query.

    Returns a string like "1", "3", "5" or None.
    The returned value matches the format stored in ChromaDB metadata.
    """
    match = re.search(r"\bunit[\s\-]*([1-9]\d*)\b", query.lower())
    if match:
        return match.group(1)
    return None

def score_units(query_lower: str, subject_entry) -> tuple[str, float] | None:
    """
    Score units within a subject using keyword matching.
    Returns (best_unit_str, score) or None.
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
    Convert a unit number to the ChromaDB metadata format.
    """
    return unit
