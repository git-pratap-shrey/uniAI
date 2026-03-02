"""
search.py
─────────
Retrieves candidate chunks from ChromaDB.

Handles two document types in one collection:
  - notes / handwritten_notes / printed_notes / question_paper
  - syllabus

Both share the same metadata schema so a single function works for all.
Callers can filter by doc_type if needed.

Returns raw ChromaDB-style dicts — no prompt formatting here.
"""

import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import chromadb
import config
from pipeline.embeddings.local_embedding import embed

# ---------------------------------------------------------------------------
# ChromaDB — single persistent client shared across all calls
# ---------------------------------------------------------------------------

_client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)
_collection = None


def _get_collection():
    global _collection
    if _collection is None:
        try:
            _collection = _client.get_collection(config.CHROMA_COLLECTION_NAME)
        except Exception as e:
            raise RuntimeError(
                f"ChromaDB collection '{config.CHROMA_COLLECTION_NAME}' not found.\n"
                f"Run ingest_multimodal.py and ingest_multimodal_syllabus.py first.\n"
                f"Original error: {e}"
            )
    return _collection


# ---------------------------------------------------------------------------
# Core retrieval
# ---------------------------------------------------------------------------

def retrieve(
    query: str,
    subject: str | None = None,
    unit: str | None = None,
    doc_type: str | None = None,   # e.g. "syllabus", "notes", None = all
    k: int = 8,
    similarity_threshold: float | None = None,
) -> list[dict]:
    """
    Retrieve relevant chunks with metadata filtering and similarity threshold.

    Returns a list of dicts:
    [
        {
            "text": "...",
            "metadata": { source, unit, subject, title, document_type, confidence, ... },
            "distance": 0.42,
            "similarity": 0.58,
        },
        ...
    ]
    """
    if similarity_threshold is None:
        similarity_threshold = config.SIMILARITY_THRESHOLD

    collection = _get_collection()

    # Build ChromaDB where clause
    filters = []

    if subject:
        filters.append({"subject": subject})

    if unit:
        # Notes ingest stores unit as "unit1"; syllabus ingest stores it as "1"
        # Use $or to match both formats
        filters.append({
            "$or": [
                {"unit": unit},
                {"unit": f"unit{unit}"},
            ]
        })

    if doc_type:
        filters.append({"document_type": doc_type})

    where_clause = None
    if len(filters) == 1:
        where_clause = filters[0]
    elif len(filters) > 1:
        where_clause = {"$and": filters}

    # Embed query
    query_vector = embed([query])[0]

    query_params = {
        "query_embeddings": [query_vector],
        "n_results": k,
        "include": ["documents", "metadatas", "distances"],
    }
    if where_clause:
        query_params["where"] = where_clause

    try:
        results = collection.query(**query_params)
    except Exception as e:
        print(f"[search] ChromaDB query failed: {e}")
        return []

    if not results or not results.get("documents"):
        return []

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    # Apply similarity threshold (cosine space: distance = 1 - similarity)
    max_distance = 1.0 - similarity_threshold

    chunks = []
    for doc, meta, dist in zip(docs, metas, dists):
        if dist <= max_distance:
            chunks.append({
                "text": doc,
                "metadata": meta,
                "distance": dist,
                "similarity": round(1.0 - dist, 4),
            })

    return chunks


def retrieve_notes(
    query: str,
    subject: str | None = None,
    unit: str | None = None,
    k: int = 8,
    similarity_threshold: float | None = None,
) -> list[dict]:
    """Retrieve note chunks (excludes syllabus)."""
    # Don't filter by doc_type=notes because ingestion uses varied values
    # (printed_notes, handwritten_notes, question_paper etc.)
    # Instead exclude only syllabus explicitly.
    if similarity_threshold is None:
        similarity_threshold = config.SIMILARITY_THRESHOLD

    collection = _get_collection()

    filters = []
    if subject:
        filters.append({"subject": subject})
    if unit:
        filters.append({
            "$or": [
                {"unit": unit},
                {"unit": f"unit{unit}"},
            ]
        })

    # Exclude syllabus chunks
    filters.append({"document_type": {"$ne": "syllabus"}})

    where_clause = None
    if len(filters) == 1:
        where_clause = filters[0]
    elif len(filters) > 1:
        where_clause = {"$and": filters}

    query_vector = embed([query])[0]
    query_params = {
        "query_embeddings": [query_vector],
        "n_results": k,
        "include": ["documents", "metadatas", "distances"],
    }
    if where_clause:
        query_params["where"] = where_clause

    try:
        results = collection.query(**query_params)
    except Exception as e:
        print(f"[search] Notes query failed: {e}")
        return []

    if not results or not results.get("documents"):
        return []

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]
    max_distance = 1.0 - similarity_threshold

    return [
        {
            "text": doc,
            "metadata": meta,
            "distance": dist,
            "similarity": round(1.0 - dist, 4),
        }
        for doc, meta, dist in zip(docs, metas, dists)
        if dist <= max_distance
    ]


def retrieve_syllabus(
    query: str,
    subject: str | None = None,
    unit: str | None = None,
    k: int = 3,
    similarity_threshold: float | None = None,
) -> list[dict]:
    """Retrieve syllabus chunks only."""
    if similarity_threshold is None:
        similarity_threshold = config.SIMILARITY_THRESHOLD

    return retrieve(
        query=query,
        subject=subject,
        unit=unit,
        doc_type="syllabus",
        k=k,
        similarity_threshold=similarity_threshold,
    )
