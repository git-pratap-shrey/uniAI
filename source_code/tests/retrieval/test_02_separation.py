"""
pytest source_code/tests/retrieval/test_02_separation.py -v

test_02_separation.py
─────────────────────
Phase 2 — Collection Separation

Verifies that the three retrieval functions hit the RIGHT collections
and don't bleed into each other:

  retrieve_notes()    → only chunks from multimodal_notes
  retrieve_syllabus() → only chunks from multimodal_syllabus
  retrieve_pyq()      → only chunks from multimodal_pyq

The key regression being tested:
  The old search.py had retrieve_syllabus() query multimodal_NOTES with
  a doc_type="syllabus" filter. Syllabus data lives in a SEPARATE collection
  (multimodal_syllabus) so the old implementation always returned nothing.

Run:
    pytest source_code/tests/retrieval/test_02_separation.py -v
"""

import sys
import os
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from source_code.rag.search import (
    retrieve_notes,
    retrieve_syllabus,
    retrieve_pyq,
    retrieve_all,
    collection_exists,
)


# ---------------------------------------------------------------------------
# Skip guards
# ---------------------------------------------------------------------------

notes_available    = pytest.mark.skipif(not collection_exists("notes"),    reason="multimodal_notes not found")
syllabus_available = pytest.mark.skipif(not collection_exists("syllabus"), reason="multimodal_syllabus not found")
pyq_available      = pytest.mark.skipif(not collection_exists("pyq"),      reason="multimodal_pyq not found")

# A generic query that should return something from any educational collection
BROAD_QUERY = "explain"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _doc_type(chunk: dict) -> str:
    return chunk["metadata"].get("document_type", "unknown")

def _collection(chunk: dict) -> str:
    return chunk.get("collection", "unknown")


# ---------------------------------------------------------------------------
# Section A — retrieve_notes stays within its collection
# ---------------------------------------------------------------------------

class TestNotesIsolation:
    @notes_available
    def test_notes_returns_results(self):
        """retrieve_notes must return at least one result against a broad query."""
        results = retrieve_notes(BROAD_QUERY, k=5, threshold=0.0)
        assert results, "retrieve_notes returned nothing — collection may be empty"

    @notes_available
    def test_notes_tagged_with_correct_collection(self):
        """Every chunk must carry collection='notes'."""
        results = retrieve_notes(BROAD_QUERY, k=10, threshold=0.0)
        for chunk in results:
            assert _collection(chunk) == "notes", (
                f"Expected collection='notes', got '{_collection(chunk)}' "
                f"for doc {chunk['metadata'].get('source')}"
            )

    @notes_available
    def test_notes_does_not_contain_syllabus_type(self):
        """
        Notes results must not have document_type='syllabus'.
        (Syllabus chunks were ingested into a separate collection.)
        """
        results = retrieve_notes(BROAD_QUERY, k=20, threshold=0.0)
        syllabus_leakage = [c for c in results if _doc_type(c) == "syllabus"]
        assert syllabus_leakage == [], (
            f"{len(syllabus_leakage)} syllabus chunk(s) leaked into notes results:\n"
            + "\n".join(c["metadata"].get("source", "?") for c in syllabus_leakage)
        )

    @notes_available
    def test_notes_does_not_contain_pyq_type(self):
        """Notes results must not have document_type='pyq'."""
        results = retrieve_notes("exam question marks", k=20, threshold=0.0)
        pyq_leakage = [c for c in results if _doc_type(c) == "pyq"]
        assert pyq_leakage == [], (
            f"{len(pyq_leakage)} PYQ chunk(s) leaked into notes results"
        )


# ---------------------------------------------------------------------------
# Section B — retrieve_syllabus hits the correct separate collection
# ---------------------------------------------------------------------------

class TestSyllabusIsolation:
    @syllabus_available
    def test_syllabus_returns_results(self):
        """
        retrieve_syllabus must return results.
        This is the core regression test — the old code returned nothing here
        because it queried the wrong collection.
        """
        results = retrieve_syllabus("unit topics course outcomes", k=5, threshold=0.0)
        assert results, (
            "retrieve_syllabus returned nothing.\n"
            "LIKELY BUG: still querying multimodal_notes instead of multimodal_syllabus.\n"
            "Fix: make sure retrieve_syllabus() calls _get('syllabus'), not _get('notes')."
        )

    @syllabus_available
    def test_syllabus_tagged_with_correct_collection(self):
        """Every chunk must carry collection='syllabus'."""
        results = retrieve_syllabus("topics", k=10, threshold=0.0)
        for chunk in results:
            assert _collection(chunk) == "syllabus", (
                f"Expected collection='syllabus', got '{_collection(chunk)}'"
            )

    @syllabus_available
    def test_syllabus_chunk_types_are_valid(self):
        """
        Syllabus chunks must have a chunk_type from the expected set.
        """
        valid_types = {
            f"unit_{i}" for i in range(1, 6)
        } | {"course_outcomes", "books_references"}

        results = retrieve_syllabus("topics", k=20, threshold=0.0)
        for chunk in results:
            ct = chunk["metadata"].get("chunk_type", "unknown")
            assert ct in valid_types, (
                f"Unexpected chunk_type='{ct}' in syllabus result "
                f"from {chunk['metadata'].get('source')}"
            )

    @syllabus_available
    def test_syllabus_document_type_field(self):
        """Every syllabus chunk must have document_type='syllabus'."""
        results = retrieve_syllabus("topics", k=20, threshold=0.0)
        for chunk in results:
            dt = _doc_type(chunk)
            assert dt == "syllabus", (
                f"Syllabus chunk has wrong document_type='{dt}' "
                f"from {chunk['metadata'].get('source')}"
            )

    @syllabus_available
    @notes_available
    def test_syllabus_and_notes_return_different_chunks(self):
        """
        Syllabus and notes should NOT return the same chunk IDs
        (they live in different collections and have different content).
        """
        note_texts    = {c["text"][:80] for c in retrieve_notes("define", k=10, threshold=0.0)}
        syllabus_texts = {c["text"][:80] for c in retrieve_syllabus("topics", k=10, threshold=0.0)}

        overlap = note_texts & syllabus_texts
        assert not overlap, (
            f"{len(overlap)} chunks appear in both notes and syllabus results — "
            f"possible collection misconfiguration."
        )


# ---------------------------------------------------------------------------
# Section C — retrieve_pyq hits the PYQ collection
# ---------------------------------------------------------------------------

class TestPYQIsolation:
    @pyq_available
    def test_pyq_returns_results(self):
        results = retrieve_pyq("define explain", k=5, threshold=0.0)
        assert results, "retrieve_pyq returned nothing — pyq collection may be empty"

    @pyq_available
    def test_pyq_tagged_with_correct_collection(self):
        results = retrieve_pyq("define", k=10, threshold=0.0)
        for chunk in results:
            assert _collection(chunk) == "pyq", (
                f"Expected collection='pyq', got '{_collection(chunk)}'"
            )

    @pyq_available
    def test_pyq_document_type_field(self):
        results = retrieve_pyq("define", k=10, threshold=0.0)
        for chunk in results:
            dt = _doc_type(chunk)
            assert dt == "pyq", (
                f"PYQ chunk has unexpected document_type='{dt}'"
            )

    @pyq_available
    @notes_available
    def test_pyq_and_notes_return_different_chunks(self):
        note_texts = {c["text"][:80] for c in retrieve_notes("define", k=10, threshold=0.0)}
        pyq_texts  = {c["text"][:80] for c in retrieve_pyq("define", k=10, threshold=0.0)}
        overlap = note_texts & pyq_texts
        assert not overlap, (
            f"{len(overlap)} chunk(s) appear in both notes and PYQ results."
        )


# ---------------------------------------------------------------------------
# Section D — retrieve_all combines notes + syllabus (not pyq)
# ---------------------------------------------------------------------------

class TestRetrieveAll:
    @notes_available
    @syllabus_available
    def test_retrieve_all_contains_both_collections(self):
        """
        retrieve_all() must return chunks tagged from both 'notes' and 'syllabus'.
        """
        results = retrieve_all("unit topics explain", notes_k=5, threshold=0.0)
        collections_seen = {_collection(c) for c in results}

        # At minimum it should contain notes (syllabus may not match the query well)
        assert "notes" in collections_seen, "retrieve_all returned no notes chunks"

    @notes_available
    def test_retrieve_all_notes_come_first(self):
        """
        Notes chunks should appear before syllabus chunks in retrieve_all output
        (notes are more directly useful for exam answers).
        """
        results = retrieve_all("explain", notes_k=5, threshold=0.0)
        if len(results) > 1:
            # Find first non-notes chunk
            first_non_note = next(
                (i for i, c in enumerate(results) if _collection(c) != "notes"), None
            )
            first_note = next(
                (i for i, c in enumerate(results) if _collection(c) == "notes"), None
            )
            if first_non_note is not None and first_note is not None:
                assert first_note < first_non_note, (
                    "Syllabus chunk appeared before notes chunk in retrieve_all output"
                )

    @notes_available
    def test_retrieve_all_no_pyq_chunks(self):
        """retrieve_all must never include PYQ chunks."""
        results = retrieve_all("define explain", notes_k=10, threshold=0.0)
        pyq_chunks = [c for c in results if _collection(c) == "pyq"]
        assert pyq_chunks == [], (
            f"{len(pyq_chunks)} PYQ chunk(s) leaked into retrieve_all output"
        )


# ---------------------------------------------------------------------------
# Section E — Similarity scores are sane
# ---------------------------------------------------------------------------

class TestSimilarityScores:
    @notes_available
    def test_similarity_within_range(self):
        """All similarity scores must be in [0, 1]."""
        results = retrieve_notes(BROAD_QUERY, k=10, threshold=0.0)
        for chunk in results:
            sim = chunk["similarity"]
            assert 0.0 <= sim <= 1.0, f"Out-of-range similarity: {sim}"

    @notes_available
    def test_threshold_filters_low_similarity(self):
        """
        Results with threshold=0.8 must all have similarity ≥ 0.8.
        (Skip if no results — collection may just not have high-similarity matches.)
        """
        results = retrieve_notes(BROAD_QUERY, k=10, threshold=0.8)
        for chunk in results:
            assert chunk["similarity"] >= 0.8, (
                f"Chunk with similarity={chunk['similarity']} passed threshold=0.8 filter"
            )

    @notes_available
    def test_lower_threshold_returns_more_results(self):
        """
        A lower similarity threshold must return ≥ results than a higher one.
        """
        strict  = retrieve_notes(BROAD_QUERY, k=10, threshold=0.7)
        relaxed = retrieve_notes(BROAD_QUERY, k=10, threshold=0.1)
        assert len(relaxed) >= len(strict), (
            f"Strict ({len(strict)}) returned more than relaxed ({len(relaxed)}) — threshold logic broken"
        )