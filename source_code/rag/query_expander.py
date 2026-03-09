"""
query_expander.py
─────────────────
Enriches the raw query before embedding to bridge vocabulary gaps
between student phrasing and note/syllabus terminology.

Three expansion layers:
  1. Exam phrasing normalization  — strips question format tokens
  2. Abbreviation expansion       — driven by subject_aliases.json + hardcoded map
  3. Syllabus keyword injection   — appends unit keywords from subject_keywords.json
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
    Returns (query_unchanged, set_of_expansion_terms).
    Expansions are collected separately so the caller
    can append them without duplicating the original query.
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
    Retrieve the top N keywords for a subject/unit from subject_keywords.json.
    Falls back to core keywords if unit is unknown.
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
    Expand a raw user query for better semantic retrieval.

    Args:
        user_query: Raw student question.
        subject:    Detected subject (e.g. "CYBER_SECURITY"). Optional.
        unit:       Detected unit number string (e.g. "3"). Optional.

    Returns:
        Enriched query string for embedding.
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