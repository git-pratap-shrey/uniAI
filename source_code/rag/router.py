"""
router.py
─────────
Specialized subject detector that uses a weighted keyword scoring
system against a nested keyword map.

Scoring Strategy:
1. Exact keyword match across multiple collections (notes, syllabus, pyq).
2. Per-collection weights to prioritize unit-specific topics over generic ones.
3. Fallback to LLM if no candidate subject reaches the minimum score threshold.

The scoring system ensures that high-signal sources like PYQs contribute
more to the subject detection than generic reference lists.
"""

import json
import os
import sys
import re
from source_code.config import CONFIG
from source_code import models

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from prompts import subject_router
from rag import unit_router


# -------------------------------------------------
# Exam-style question pattern tokens
# These help the router recognise student-style questions
# even when they don't contain subject keywords literally.
# -------------------------------------------------
QUESTION_TOKENS = {
    "what", "define", "explain", "list", "describe",
    "difference between", "advantages", "disadvantages",
    "types of", "working of", "give example", "compare",
    "discuss", "state", "how", "why", "write short note",
}

# Scoring weights by (collection, unit_label)
# "unknown" → 0, "core" → collection-level weight, numbered units → unit weight
_WEIGHTS = {
    ("notes",    "unit"):    4,
    ("notes",    "core"):    2,
    ("syllabus", "unit"):    3,
    ("syllabus", "core"):    2,
    ("pyq",      "flat"):    5,   # PYQ is a flat list
    ("any",      "unknown"): 0,   # suppress noise
}


# -------------------------------------------------
# Load keyword map once
# -------------------------------------------------

KEYWORDS_FILE = config.KEYWORDS_FILE_PATH

_keyword_map: dict = {}

if os.path.exists(KEYWORDS_FILE):
    try:
        with open(KEYWORDS_FILE, "r", encoding="utf-8") as f:
            _keyword_map = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"[router] WARNING: Could not load keyword map: {e}")
else:
    print(f"[router] WARNING: Keyword map not found at {KEYWORDS_FILE}")
    print("         Run generate_keyword_map.py first.")


def _flatten_keywords(entry) -> list[str]:
    """
    Consolidate a nested subject's keyword dictionary into a single flat list.

    This ensures backwards compatibility with legacy flat list formats
    and allows for simplified, fast keyword scanning.

    Args:
        entry: The subject's keyword dictionary (nested or flat).

    Returns:
        A flat list of all keywords associated with the subject.
    """
    if isinstance(entry, list):
        return entry  # legacy flat format
    flat = []
    for collection_val in entry.values():
        if isinstance(collection_val, list):
            flat.extend(collection_val)
        elif isinstance(collection_val, dict):
            for unit_kws in collection_val.values():
                if isinstance(unit_kws, list):
                    flat.extend(unit_kws)
    return flat


def _score_subject(query_lower: str, entry) -> float:
    """
    Calculate the total weighted score for a subject based on query keyword matches.

    Aggregates scores from notes (core/unit), syllabus (core/unit), and pyq flat lists
    using the weights defined in _WEIGHTS.

    Args:
        query_lower: Normalized user query.
        entry: The subject's keyword entry.

    Returns:
        A numeric score.
    """
    # Legacy flat format: treat as notes.unit weight
    if isinstance(entry, list):
        return sum(_WEIGHTS[("notes", "unit")] for kw in entry if kw in query_lower)

    score = 0.0
    for collection, collection_val in entry.items():

        if collection == "pyq":
            if isinstance(collection_val, list):
                w = _WEIGHTS[("pyq", "flat")]
                score += sum(w for kw in collection_val if kw in query_lower)
            continue

        if not isinstance(collection_val, dict):
            continue

        for unit_label, kws in collection_val.items():
            if unit_label == "unknown":
                continue   # weight = 0
            elif unit_label == "core":
                w = _WEIGHTS.get((collection, "core"), 1)
            else:
                w = _WEIGHTS.get((collection, "unit"), 1)

            if isinstance(kws, list):
                score += sum(w for kw in kws if kw in query_lower)

    return score


# -------------------------------------------------
# Public API
# -------------------------------------------------

def detect_subject(query: str, debug: bool = False):
    """
    Analyze a query to identify which subject it belongs to.

    This function coordinates the keyword scoring process and
    decides whether to trigger the LLM fallback.

    Args:
        query: User input query.
        debug: If True, returns additional info (unit, whether LLM was used).

    Returns:
        If debug is False: (subject_name, best_unit)
        If debug is True:  (subject_name, best_unit, used_llm_flag)
    """

    if not _keyword_map:
        return (None, None, False) if debug else (None, None)

    query_lower = query.lower()

    # Strip common exam question prefixes so we score on the actual topic
    for token in sorted(QUESTION_TOKENS, key=len, reverse=True):
        query_lower = re.sub(rf'\b{re.escape(token)}\b\s*', '', query_lower).strip()

    scores = {subject: _score_subject(query_lower, entry)
              for subject, entry in _keyword_map.items()}

    max_score = max(scores.values()) if scores else 0

    if max_score >= config.KEYWORD_MIN_SCORE:
        top_subjects = [s for s, v in scores.items() if v == max_score]
        if len(top_subjects) == 1:
            result = top_subjects[0]
            from rag.unit_router import score_units
            
            unit_result = score_units(query_lower, _keyword_map[result])
            best_unit = unit_result[0] if unit_result else None
            
            return (result, best_unit, False) if debug else (result, best_unit)

    # -------------------------
    # Fallback to LLM
    # -------------------------
    llm_result = _llm_classify(query)
    return (llm_result, None, True) if debug else (llm_result, None)


# -------------------------------------------------
# LLM Classification
# -------------------------------------------------

def _llm_classify(query: str) -> str | None:
    """
    Perform semantic subject classification using the centralized models registry.

    Args:
        query: User input query.

    Returns:
        The matched subject name or None.
    """
    if not _keyword_map:
        return None

    subjects_list = ", ".join(_keyword_map.keys())
    prompt = subject_router(query=query, subjects_list=subjects_list)

    try:
        response_text = models.chat(
            prompt=f"{prompt} /no_think",
            system_prompt="You are a helpful assistant. You must respond directly without internal reasoning or <think> tags.",
            model=CONFIG["rag"]["router_model"] if "router_model" in CONFIG["rag"] else CONFIG["providers"].get("router"),
            provider=CONFIG["providers"].get("router", "ollama"),
            temperature=CONFIG["rag"].get("router_temperature", 0.0),
            num_predict=CONFIG["rag"].get("router_num_predict", 10),
        )

        llm_choice = response_text.strip()
        print(f"[LLM RAW OUTPUT]: '{llm_choice}'")
        llm_choice_normalized = llm_choice.strip().upper().replace(" ", "_")

        for subject in _keyword_map:
            if subject.upper() == llm_choice_normalized:
                return subject

    except Exception as e:
        print(f"[router] LLM classification failed: {e}")

    return None


# -------------------------------------------------
# Utility
# -------------------------------------------------

def list_subjects() -> list[str]:
    """Return all known subjects."""
    return list(_keyword_map.keys())