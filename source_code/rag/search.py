"""
search.py
─────────
Retrieves candidate chunks from ChromaDB across three isolated collections:

  multimodal_notes    — lecture notes, handwritten notes, printed slides
  multimodal_syllabus — syllabus unit topics, course outcomes, book lists
  multimodal_pyq      — past year question papers

Public API
----------
  retrieve_notes(query, subject, unit, k, threshold)   → list[Chunk]
  retrieve_syllabus(query, subject, unit, k, threshold) → list[Chunk]
  retrieve_pyq(query, subject, unit, k, threshold)      → list[Chunk]

Each function returns a list of Chunk dicts:
  {
    "text":       str,
    "metadata":   dict,   # raw ChromaDB metadata
    "distance":   float,  # cosine distance (lower = more similar)
    "similarity": float,  # 1 - distance (higher = more similar)
    "collection": str,    # which collection this came from
  }

Unit normalisation
------------------
Both ingestion pipelines now write plain numeric strings ("1", "2" …).
Older ingest runs may have written "unit1". The _unit_filter() helper
builds a ChromaDB $or clause that matches both forms so old data works
transparently.
"""

import os
import re
import sys
from typing import TypedDict

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import chromadb
from source_code.config import CONFIG
from pipeline.embeddings.local_embedding import embed

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class Chunk(TypedDict):
    text:       str
    metadata:   dict
    distance:   float
    similarity: float
    collection: str


# ---------------------------------------------------------------------------
# ChromaDB — one persistent client, lazy-loaded collections
# ---------------------------------------------------------------------------

_client = chromadb.PersistentClient(path=CONFIG["paths"]["chroma"])

# Collection handles, populated on first access
_collections: dict[str, chromadb.Collection] = {}

# Mapping of collection aliases to their actual names in ChromaDB.
# - notes:    Contains PDF/Slide chunks (lecture content).
# - syllabus: Contains Unit descriptions and course outcomes.
# - pyq:      Contains short, standalone exam questions.
_COLLECTION_NAMES = {
    "notes":    CONFIG["paths"]["collections"]["notes"],          # multimodal_notes
    "syllabus": CONFIG["paths"]["collections"]["syllabus"],  # multimodal_syllabus
    "pyq":      CONFIG["paths"]["collections"]["pyq"],       # multimodal_pyq
}


def _get(alias: str) -> chromadb.Collection:
    """
    Retrieve a ChromaDB collection object by its internal alias.

    This uses a lazy-loading pattern to minimize initialization overhead.

    Args:
        alias: One of "notes", "syllabus", or "pyq".

    Returns:
        The corresponding chromadb.Collection object.

    Raises:
        RuntimeError: If the collection does not exist in the database.
    """
    if alias not in _collections:
        name = _COLLECTION_NAMES[alias]
        try:
            _collections[alias] = _client.get_collection(name)
        except Exception as exc:
            raise RuntimeError(
                f"ChromaDB collection '{name}' not found. "
                f"Run the corresponding ingest script first.\n"
                f"Original error: {exc}"
            )
    return _collections[alias]


def collection_exists(alias: str) -> bool:
    """
    Check if a specific search collection is available in ChromaDB.

    Args:
        alias: The collection alias to check.

    Returns:
        True if available, False otherwise.
    """
    try:
        _get(alias)
        return True
    except RuntimeError:
        return False


# ---------------------------------------------------------------------------
# Unit normalisation
# ---------------------------------------------------------------------------

def normalize_unit(raw: str | int | None) -> str | None:
    """
    Standardize unit identifiers into a clean numeric string.

    This handles various input formats like "unit 3", "Unit3", or just 3.
    The resulting string ("1", "2", "3", etc.) is used for filtering.

    Args:
        raw: The raw unit input.

    Returns:
        A numeric string or None if no number is found.
    """
    if raw is None:
        return None
    s = str(raw).strip().lower()
    m = re.search(r"\d+", s)
    if m:
        return str(int(m.group()))   # strips leading zeros
    return None


def _unit_filter(unit: str) -> dict:
    """
    Build a flexible ChromaDB filter for unit numbers.

    This creates an $or clause to support both the current ("1")
    and legacy ("unit1") storage formats.

    Args:
        unit: The normalized numeric unit string.
    """
    return {
        "$or": [
            {"unit": unit},
            {"unit": f"unit{unit}"},
        ]
    }


# ---------------------------------------------------------------------------
# Where-clause builder
# ---------------------------------------------------------------------------

def _build_where(
    subject: str | None = None,
    unit: str | None = None,
    extra: list[dict] | None = None,
) -> dict | None:
    """
    Construct a nested ChromaDB 'where' clause for metadata filtering.

    Args:
        subject: Optional subject name to filter by.
        unit:    Optional unit number to filter by.
        extra:   A list of additional ChromaDB filter dictionaries.

    Returns:
        A combined ChromaDB filter dict, or None if no filters are provided.
    """
    filters: list[dict] = []

    if subject:
        filters.append({"subject": subject.upper()})

    if unit:
        n = normalize_unit(unit)
        if n:
            filters.append(_unit_filter(n))

    if extra:
        filters.extend(extra)

    if not filters:
        return None
    if len(filters) == 1:
        return filters[0]
    return {"$and": filters}


# ---------------------------------------------------------------------------
# Core retrieval helper
# ---------------------------------------------------------------------------

def _query_collection(
    alias: str,
    query: str,
    where: dict | None,
    k: int,
    threshold: float,
) -> list[Chunk]:
    """
    Internal helper to execute a vector similarity search.

    This function handles query embedding, searching the specific collection,
    and post-filtering results based on distance.

    Args:
        alias:     Collection alias ("notes", "syllabus", "pyq").
        query:     The user's query text.
        where:     The pre-constructed metadata filter.
        k:         Number of results to fetch.
        threshold: Minimum similarity score (0.0 to 1.0) to keep a result.

    Returns:
        A list of Chunk objects sorted by similarity.
    """
    collection = _get(alias)
    query_vector = embed([query])[0]

    params: dict = {
        "query_embeddings": [query_vector],
        "n_results": k,
        "include": ["documents", "metadatas", "distances"],
    }
    if where is not None:
        params["where"] = where

    try:
        results = collection.query(**params)
    except Exception as exc:
        print(f"[search] {alias} query failed: {exc}")
        return []

    if not results or not results.get("documents"):
        return []

    docs  = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    # cosine space: distance = 1 − similarity → keep if distance ≤ max_distance
    max_dist = 1.0 - threshold

    return [
        Chunk(
            text=doc,
            metadata=meta,
            distance=round(dist, 6),
            similarity=round(1.0 - dist, 4),
            collection=alias,
        )
        for doc, meta, dist in zip(docs, metas, dists)
        if dist <= max_dist
    ]


# ---------------------------------------------------------------------------
# Public retrieval functions
# ---------------------------------------------------------------------------

def retrieve_notes(
    query: str,
    subject: str | None = None,
    unit: str | None = None,
    k: int = None,
    threshold: float | None = None,
) -> list[Chunk]:
    """
    Fetch relevant lecture note chunks while explicitly excluding syllabus metadata.

    Args:
        query:     Search text.
        subject:   Subject filter.
        unit:      Unit filter.
        k:         Max results (overrides config if provided).
        threshold: Score threshold (overrides config if provided).

    Returns:
        List of candidate chunks from lecture notes.
    """
    if k is None:
        k = CONFIG["rag"]["notes_k_default"]
    if threshold is None:
        threshold = CONFIG["rag"]["similarity_threshold"]

    where = _build_where(
        subject=subject,
        unit=unit,
        extra=[{"document_type": {"$ne": "syllabus"}}],
    )
    return _query_collection("notes", query, where, k, threshold)


def retrieve_syllabus(
    query: str,
    subject: str | None = None,
    unit: str | None = None,
    k: int = None,
    threshold: float | None = None,
) -> list[Chunk]:
    """
    Retrieve syllabus topics and learning outcomes.

    This searches the dedicated syllabus collection, which is more
    structured and higher-density than lecture notes.

    Args:
        query:     Search text.
        subject:   Subject filter.
        unit:      Unit filter.
        k:         Max results.
        threshold: Score threshold.

    Returns:
        List of syllabus-specific chunks.
    """
    if k is None:
        k = CONFIG["rag"]["syllabus_k_default"]
    if threshold is None:
        threshold = CONFIG["rag"]["similarity_threshold"]

    where = _build_where(subject=subject, unit=unit)
    return _query_collection("syllabus", query, where, k, threshold)


def retrieve_pyq(
    query: str,
    subject: str | None = None,
    unit: str | None = None,
    k: int = None,
    threshold: float = None,
    marks: int | None = None,
    year: int | None = None,
) -> list[Chunk]:
    """
    Retrieve historical exam questions from the PYQ collection.

    This collection uses a higher similarity threshold by default because
    questions are short and generic matches are common but often irrelevant.

    Args:
        query:     Search text.
        subject:   Subject filter.
        unit:      Unit filter.
        k:         Max results.
        threshold: Score threshold.
        marks:     Filter for question mark value (e.g., 2, 5, 10).
        year:      Filter for a specific exam year.

    Returns:
        List of matching past-year questions.
    """
    if k is None:
        k = CONFIG["rag"]["pyq_k_default"]
    if threshold is None:
        threshold = CONFIG["rag"]["pyq_threshold"]

    extra: list[dict] = []
    if marks is not None:
        extra.append({"marks": marks})
    if year is not None:
        extra.append({"year": year})

    where = _build_where(subject=subject, unit=unit, extra=extra or None)
    return _query_collection("pyq", query, where, k, threshold)


def retrieve_all(
    query: str,
    subject: str | None = None,
    unit: str | None = None,
    notes_k: int = None,
    syllabus_k: int = None,
    threshold: float | None = None,
) -> list[Chunk]:
    """
    Retrieve from both notes and syllabus and return them interleaved.

    This is primarily used for unit overview requests where the user wants
    to see both what is officially in the syllabus and the detailed notes for it.

    Args:
        query:      Search text.
        subject:    Subject filter.
        unit:       Unit filter.
        notes_k:    Max note results.
        syllabus_k: Max syllabus results.
        threshold:  Score threshold.

    Returns:
        A combined list of chunks.
    """
    if notes_k is None:
        notes_k = CONFIG["rag"]["all_notes_k"]
    if syllabus_k is None:
        syllabus_k = CONFIG["rag"]["all_syllabus_k"]

    notes = retrieve_notes(query, subject=subject, unit=unit, k=notes_k, threshold=threshold)
    syllabus = retrieve_syllabus(query, subject=subject, unit=unit, k=syllabus_k, threshold=threshold)
    return notes + syllabus