"""
router.py
----------
Detects which subject a query belongs to.

Strategy:
1. Score query against keyword map (fast, deterministic)
2. If ambiguous or no match → LLM fallback (slower, semantic)
3. Returns subject name or None

Optional debug mode:
- Returns (subject, used_llm_bool)
"""

import json
import os
import sys
import config
import ollama

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from prompts import subject_router


# -------------------------------------------------
# Load keyword map once
# -------------------------------------------------

KEYWORDS_FILE = os.path.join(ROOT_DIR, "data", "subject_keywords.json")

_keyword_map: dict[str, list[str]] = {}

if os.path.exists(KEYWORDS_FILE):
    try:
        with open(KEYWORDS_FILE, "r", encoding="utf-8") as f:
            _keyword_map = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"[router] WARNING: Could not load keyword map: {e}")
else:
    print(f"[router] WARNING: Keyword map not found at {KEYWORDS_FILE}")
    print("         Run generate_keyword_map.py first.")


# -------------------------------------------------
# Public API
# -------------------------------------------------

def detect_subject(query: str, debug: bool = False):
    """
    Detect subject of a query.

    Returns:
        subject (str | None)
        If debug=True → returns (subject, used_llm_bool)
    """

    if not _keyword_map:
        return (None, False) if debug else None

    query_lower = query.lower()
    scores = {subject: 0 for subject in _keyword_map}

    # -------------------------
    # Keyword Scoring
    # -------------------------
    for subject, keywords in _keyword_map.items():
        for kw in keywords:
            if kw in query_lower:
                scores[subject] += 1

    max_score = max(scores.values())

    if max_score > 0:
        top_subjects = [s for s, v in scores.items() if v == max_score]

        if len(top_subjects) == 1:
            result = top_subjects[0]
            return (result, False) if debug else result

    # -------------------------
    # Fallback to LLM
    # -------------------------
    llm_result = _llm_classify(query)

    return (llm_result, True) if debug else llm_result


# -------------------------------------------------
# LLM Classification
# -------------------------------------------------

def _llm_classify(query: str) -> str | None:
    """Ask local LLM to classify subject."""

    if not _keyword_map:
        return None

    subjects_list = ", ".join(_keyword_map.keys())

    prompt = subject_router(query=query, subjects_list=subjects_list)

    try:
        client = ollama.Client(host=config.OLLAMA_LOCAL_URL)

        response = client.chat(
            model=config.MODEL_ROUTER,
            messages=[
                # Force the model to bypass reasoning at the system level
                {"role": "system", "content": "You are a helpful assistant. You must respond directly without internal reasoning or <think> tags."},
                {"role": "user", "content": f"{prompt} /no_think"}
            ],
            think=False,
            options={
                "temperature": 0,
                "num_predict": 10,
            },
        )

        llm_choice = response["message"]["content"].strip()
        print(f"[LLM RAW OUTPUT]: '{llm_choice}'")  # add this

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