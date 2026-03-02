"""
router.py
─────────
Detects which subject a query belongs to.

Strategy:
  1. Score query against keyword map (fast, deterministic)
  2. If ambiguous or no match → LLM fallback (slow, accurate)
  3. Returns None if subject cannot be resolved

No database calls. No retrieval logic.
"""

import json
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import config
import ollama

# ---------------------------------------------------------------------------
# Load keyword map once at module import
# ---------------------------------------------------------------------------

KEYWORDS_FILE = os.path.join(ROOT_DIR, "data", "subject_keywords.json")

_keyword_map: dict[str, list[str]] = {}

if os.path.exists(KEYWORDS_FILE):
    try:
        with open(KEYWORDS_FILE, "r", encoding="utf-8") as f:
            _keyword_map = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"[router] WARNING: Could not load keyword map: {e}")
else:
    print(f"[router] WARNING: Keyword map not found at {KEYWORDS_FILE}.")
    print("         Run source_code/pipeline/generate_keyword_map.py first.")

# ---------------------------------------------------------------------------

def detect_subject(query: str) -> str | None:
    """
    Detect the subject a query belongs to.

    Returns the subject name (e.g. 'PYTHON', 'COA') or None.
    """
    if not _keyword_map:
        return None

    query_lower = query.lower()
    scores = {subject: 0 for subject in _keyword_map}

    for subject, keywords in _keyword_map.items():
        for kw in keywords:
            if kw in query_lower:
                scores[subject] += 1

    max_score = max(scores.values())

    if max_score > 0:
        top = [s for s, v in scores.items() if v == max_score]
        if len(top) == 1:
            return top[0]  # clear winner

    # Ambiguous or zero score — ask the LLM
    return _llm_classify(query)


def _llm_classify(query: str) -> str | None:
    """Fallback: ask a fast local model to classify the query."""
    subjects_list = ", ".join(_keyword_map.keys())

    prompt = (
        f"You are a routing agent for a university study assistant.\n"
        f"Known subjects: {subjects_list}\n\n"
        f'User query: "{query}"\n\n'
        f"Which subject is this query about?\n"
        f"Reply ONLY with the exact subject name from the list above. "
        f"If none match, reply NONE."
    )

    try:
        client = ollama.Client(host=config.OLLAMA_LOCAL_URL)
        response = client.chat(
            model=config.MODEL_ROUTER,
            messages=[{"role": "user", "content": prompt}]
        )
        llm_choice = response["message"]["content"].strip()

        for subject in _keyword_map:
            if subject.lower() in llm_choice.lower():
                return subject

    except Exception as e:
        print(f"[router] LLM classification failed: {e}")

    return None


def list_subjects() -> list[str]:
    """Return all known subjects."""
    return list(_keyword_map.keys())
