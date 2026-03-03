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
import config
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

_client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)

# Collection handles, populated on first access
_collections: dict[str, chromadb.Collection] = {}

_COLLECTION_NAMES = {
    "notes":    config.CHROMA_COLLECTION_NAME,          # multimodal_notes
    "syllabus": config.CHROMA_SYLLABUS_COLLECTION_NAME,  # multimodal_syllabus
    "pyq":      config.CHROMA_PYQ_COLLECTION_NAME,       # multimodal_pyq
}


def _get(alias: str) -> chromadb.Collection:
    """Return (and cache) a ChromaDB collection by alias."""
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
    """Check whether a collection is present without raising."""
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
    Return a plain numeric string ("1", "3" …) or None.

    Handles:  1, "1", "unit1", "Unit 1", "unit-1", "unit 03"
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
    Build a ChromaDB $or clause that matches both storage formats.

    Notes ingest (current):   unit = "1"
    Notes ingest (old):       unit = "unit1"
    Syllabus ingest:          unit = "1"
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
    Compose a ChromaDB where clause from optional subject, unit, and extra filters.

    Returns None if no filters are needed (avoids passing empty where= to Chroma).
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
    Embed the query, run a ChromaDB similarity search, and filter by threshold.
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
    Retrieve lecture note chunks from multimodal_notes.

    Excludes syllabus chunks (they have their own collection).
    """
    if k is None:
        k = config.SEARCH_NOTES_K_DEFAULT
    if threshold is None:
        threshold = config.SIMILARITY_THRESHOLD

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
    Retrieve syllabus chunks from multimodal_syllabus.

    This is the correct collection — NOT a filter inside multimodal_notes.
    chunk_type values:  unit_1…unit_5, course_outcomes, books_references
    """
    if k is None:
        k = config.SEARCH_SYLLABUS_K_DEFAULT
    if threshold is None:
        threshold = config.SIMILARITY_THRESHOLD

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
    Retrieve past year question chunks from multimodal_pyq.

    Uses a higher default threshold (0.60) because PYQ questions are short
    and low-similarity hits are almost always noise.

    Optional filters:
      marks — e.g. 10 for ten-mark questions
      year  — e.g. 2023
    """
    if k is None:
        k = config.SEARCH_PYQ_K_DEFAULT
    if threshold is None:
        threshold = config.SEARCH_PYQ_THRESHOLD

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
    Retrieve from notes + syllabus and return them interleaved (notes first).

    Useful for unit-overview queries where you want both conceptual content
    and the official topic list.
    """
    if notes_k is None:
        notes_k = config.SEARCH_ALL_NOTES_K
    if syllabus_k is None:
        syllabus_k = config.SEARCH_ALL_SYLLABUS_K

    notes = retrieve_notes(query, subject=subject, unit=unit, k=notes_k, threshold=threshold)
    syllabus = retrieve_syllabus(query, subject=subject, unit=unit, k=syllabus_k, threshold=threshold)
    return notes + syllabus