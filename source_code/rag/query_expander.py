"""
query_expander.py
─────────────────
Enhances user queries to bridge the vocabulary gap between student
phrasing and formal academic materials.

Expansion Strategy:
1. **Syllabus Normalization**: Removes common exam phrases (e.g., "Write a short note on").
2. **Abbreviation Expansion**: Replaces technical shorthand (e.g., "CIA" → "Confidentiality Integrity Availability").
3. **Keyword Injection**: Appends high-value keywords from the target subject/unit to anchor the search.

This module ensures that even a loosely phrased student query can find
precisely relevant technical content in the database.
"""

import json
import os
import sys
import re

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import config

ALIASES_FILE  = config.ALIASES_FILE_PATH
KEYWORDS_FILE = config.KEYWORDS_FILE_PATH

# ---------------------------------------------------------------------------
# Layer 1 — Exam phrasing normalization
# Strip question-format words so embedding focuses on the concept.
# Mirrors the stripping done in router.py for keyword scoring.
# ---------------------------------------------------------------------------

_EXAM_PHRASING = re.compile(
    r'\b(short note on|write a note on|define|explain|describe|'
    r'discuss|differentiate between|difference between|compare|'
    r'advantages of|disadvantages of|what is|what are|'
    r'how does|how do|list|enumerate|give examples? of|'
    r'\d+\s*marks?)\b',
    re.IGNORECASE
)

def normalize_exam_phrasing(query: str) -> str:
    """
    Remove common academic/exam prefixes that don't add semantic value.

    Args:
        query: User input query.

    Returns:
        The query string without leading/trailing exam-style phrasing.
    """
    cleaned = _EXAM_PHRASING.sub(' ', query)
    return re.sub(r'\s+', ' ', cleaned).strip()


# ---------------------------------------------------------------------------
# Layer 2 — Abbreviation/alias expansion
# Expands known abbreviations and subject-specific shorthand.
# ---------------------------------------------------------------------------

# Hardcoded map for abbreviations that are too short or ambiguous
# for reliable detection from subject_aliases.json
# Keys must be ≥ 3 chars or use explicit context guards
ABBREV_MAP: dict[str, list[str]] = {
    "k-map":         ["karnaugh map"],
    "k map":         ["karnaugh map"],
    "tabular method":["quine mccluskey", "boolean minimization"],
    "qm method":     ["quine mccluskey"],
    "pos":           ["product of sums"],
    "sop":           ["sum of products"],
    "ddos":          ["distributed denial of service"],
    "dos attack":    ["denial of service"],
    "mitm":          ["man in the middle attack"],
    "sql injection": ["web attack", "database attack"],
    "cia triad":     ["confidentiality integrity availability"],
    "dpdp":          ["digital personal data protection"],
    "ipr":           ["intellectual property rights"],
    "mux":           ["multiplexer"],
    "demux":         ["demultiplexer"],
    "ff":            ["flip flop"],
    "alu":           ["arithmetic logic unit"],
    "isa":           ["instruction set architecture"],
    "risc":          ["reduced instruction set computer"],
    "cisc":          ["complex instruction set computer"],
}

# Load subject aliases (must be ≥ 3 chars to avoid substring false positives)
_subject_aliases: dict[str, list[str]] = {}
if os.path.exists(ALIASES_FILE):
    try:
        with open(ALIASES_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)
        # Only load aliases that are long enough to be safe
        for subject, aliases in raw.items():
            safe = [a for a in aliases if len(a) >= 3]
            if safe:
                _subject_aliases[subject] = safe
    except (json.JSONDecodeError, IOError) as e:
        print(f"[query_expander] Could not load aliases: {e}")


def expand_abbreviations(query: str) -> tuple[str, set[str]]:
    """
    Identify and expand known abbreviations in the query.

    Args:
        query: The raw or normalized query.

    Returns:
        A tuple of (original_query, set_of_expansion_terms).
    """
    q = query.lower()
    expansions: set[str] = set()

    # Hardcoded abbreviations — use word boundaries
    for abbrev, terms in ABBREV_MAP.items():
        if re.search(rf'\b{re.escape(abbrev)}\b', q):
            expansions.update(terms)

    # Subject aliases — only match whole words, min 3 chars
    for subject, aliases in _subject_aliases.items():
        for alias in aliases:
            if len(alias) >= 3 and re.search(rf'\b{re.escape(alias)}\b', q):
                expansions.add(subject.replace("_", " ").lower())
                break

    return query, expansions


# ---------------------------------------------------------------------------
# Layer 3 — Syllabus keyword injection
# If subject + unit are known, append the unit's top keywords
# so the embedding is anchored to syllabus vocabulary.
# ---------------------------------------------------------------------------

_keyword_map: dict = {}
if os.path.exists(KEYWORDS_FILE):
    try:
        with open(KEYWORDS_FILE, "r", encoding="utf-8") as f:
            _keyword_map = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"[query_expander] Could not load keyword map: {e}")


def get_unit_keywords(subject: str, unit: str | None, top_n: int = config.QUERY_EXPANDER_MAX_KEYWORDS) -> list[str]:
    """
    Retrieve the most descriptive keywords for a specific subject/unit.

    Args:
        subject: The detected subject name.
        unit:    The detected unit number.
        top_n:   Maximum number of keywords to retrieve.

    Returns:
        A list of academic keywords associated with the unit.
    """
    if not subject or subject not in _keyword_map:
        return []

    entry = _keyword_map[subject]
    if isinstance(entry, list):
        # Legacy flat format
        return entry[:top_n]

    keywords: list[str] = []

    # Prefer unit-specific keywords
    if unit:
        for collection in ("notes", "syllabus"):
            col_val = entry.get(collection, {})
            if isinstance(col_val, dict) and unit in col_val:
                keywords.extend(col_val[unit])

    # Always include core keywords as anchors
    for collection in ("notes", "syllabus"):
        col_val = entry.get(collection, {})
        if isinstance(col_val, dict):
            keywords.extend(col_val.get("core", []))

    # Deduplicate, preserve order
    seen = set()
    deduped = []
    for kw in keywords:
        if kw not in seen:
            seen.add(kw)
            deduped.append(kw)

    return deduped[:top_n]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def expand_query(
    user_query: str,
    subject: str | None = None,
    unit: str | None = None,
) -> str:
    """
    The main entry point for query enrichment.

    Args:
        user_query: The student's raw input.
        subject:    Detected subject for keyword injection.
        unit:       Detected unit for keyword injection.

    Returns:
        An expanded, high-signal query string ready for vector embedding.
    """
    # Layer 1: Normalize exam phrasing
    normalized = normalize_exam_phrasing(user_query)
    if not normalized:
        normalized = user_query  # don't erase the whole query

    # Layer 2: Abbreviation expansion
    _, abbrev_terms = expand_abbreviations(normalized)

    # Layer 3: Syllabus keyword injection
    syllabus_terms = get_unit_keywords(subject, unit) if subject else []

    # Combine: original concept (normalized) + abbreviation expansions + syllabus anchors
    all_additions = list(abbrev_terms) + syllabus_terms

    if all_additions:
        return normalized + " " + " ".join(all_additions)

    return normalized