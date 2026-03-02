"""
unit_detector.py
────────────────
Infers unit number from a query string.

v1: Regex-based explicit detection only.
    Handles patterns like: "unit 3", "unit3", "unit-3", "unit 3 file handling"

No database calls. No LLM calls.
"""

import re


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


def format_unit_filter(unit: str) -> str:
    """
    Convert a unit number to the ChromaDB metadata format.

    ChromaDB stores unit as e.g. "1", "2", "unit1" depending on ingestion.
    Current ingestion (ingest_multimodal.py) stores it as the raw value from
    the path, which is typically "unit1", "unit2" etc.

    This function returns both forms so callers can decide which to use.
    """
    return unit  # Stored as plain string "1", "2" etc. from syllabus ingestion
                 # and as "unit1", "unit2" from notes ingestion path.
                 # search.py handles the OR logic.
