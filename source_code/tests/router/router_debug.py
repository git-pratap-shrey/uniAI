"""
router_debug.py
──────────────
Debug tracing utilities for the router pipeline.
Provides structured debug output for each routing stage.
"""

import re
from typing import Any


def build_debug_trace() -> dict[str, Any]:
    """
    Create an empty debug trace structure.

    Returns:
        Dict with all routing stages initialized.
    """
    return {
        "query": None,
        "regex": {},
        "keyword": {},
        "embedding": {},
        "llm": {},
        "final_decision": None
    }


def parse_router_output(text: str) -> tuple[str, str]:
    """
    Parse router output to extract subject and unit.

    Args:
        text: The raw output text from a router stage.

    Returns:
        Tuple of (subject, unit) strings.
    """
    subject_match = re.search(r"subject\s*=\s*([A-Z_]+)", text, re.IGNORECASE)
    unit_match = re.search(r"unit\s*=\s*([0-9]+)", text)

    subject = subject_match.group(1).upper() if subject_match else "none"
    unit = unit_match.group(1) if unit_match else "none"

    return subject, unit


def format_router_output(subject: str | None, unit: str | None) -> str:
    """
    Format subject and unit into standardized output format.

    Args:
        subject: The subject name (or None).
        unit: The unit number (or None).

    Returns:
        Formatted string: "subject=<SUBJECT> | unit=<UNIT>"
    """
    subj = subject.upper().replace(" ", "_") if subject else "none"
    u = unit if unit else "none"
    return f"subject={subj} | unit={u}"