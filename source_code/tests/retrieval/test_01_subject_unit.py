"""
test_01_subject_unit.py
───────────────────────
Phase 1 — Subject & Unit Filtering Consistency

Checks that metadata filters actually work:
  - Results from retrieve_notes(subject="COA") only contain COA documents.
  - Results from retrieve_notes(unit="3") only contain unit-3 documents.
  - Combined filters narrow results correctly.
  - Missing collections are skipped gracefully.

Run from the project root:
    pytest source_code/tests/retrieval/test_01_subject_unit.py -v
"""

import sys
import os
import re
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from source_code.rag.search import (
    retrieve_notes,
    retrieve_syllabus,
    retrieve_pyq,
    collection_exists,
    normalize_unit,
)


# ---------------------------------------------------------------------------
# Fixtures / skip guards
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module", autouse=True)
def require_notes_collection():
    if not collection_exists("notes"):
        pytest.skip("multimodal_notes collection not found — run ingest_multimodal.py first")


@pytest.fixture(scope="module")
def notes_subject():
    """Return the first known subject that has data, or skip."""
    for subject in ["COA", "PYTHON", "CYBER_SECURITY"]:
        results = retrieve_notes("introduction", subject=subject, k=3, threshold=0.0)
        if results:
            return subject
    pytest.skip("No subject with data found in multimodal_notes")


@pytest.fixture(scope="module")
def notes_unit(notes_subject):
    """Return a unit that has data for the detected subject, or skip."""
    for unit in ["1", "2", "3", "4", "5"]:
        results = retrieve_notes("introduction", subject=notes_subject, unit=unit, k=3, threshold=0.0)
        if results:
            return unit
    pytest.skip("No unit with data found")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _normalize_unit_meta(raw: str | None) -> str | None:
    """Normalise unit metadata value to plain number for comparison."""
    if raw is None:
        return None
    m = re.search(r"\d+", str(raw))
    return str(int(m.group())) if m else None


# ---------------------------------------------------------------------------
# Section A — normalize_unit() unit tests (no DB needed)
# ---------------------------------------------------------------------------

class TestNormalizeUnit:
    def test_plain_number(self):
        assert normalize_unit("1") == "1"

    def test_unit_prefix(self):
        assert normalize_unit("unit1") == "1"

    def test_unit_space(self):
        assert normalize_unit("Unit 3") == "3"

    def test_unit_dash(self):
        assert normalize_unit("unit-5") == "5"

    def test_leading_zero(self):
        assert normalize_unit("03") == "3"

    def test_integer_input(self):
        assert normalize_unit(2) == "2"

    def test_none_input(self):
        assert normalize_unit(None) is None

    def test_garbage_string(self):
        assert normalize_unit("unknown") is None


# ---------------------------------------------------------------------------
# Section B — Subject filtering
# ---------------------------------------------------------------------------

class TestSubjectFilter:
    def test_results_match_subject(self, notes_subject):
        """Every returned chunk must belong to the requested subject."""
        results = retrieve_notes("define", subject=notes_subject, k=10, threshold=0.0)
        assert results, f"Expected results for subject={notes_subject}"
        for chunk in results:
            meta_subject = chunk["metadata"].get("subject", "").upper()
            assert meta_subject == notes_subject.upper(), (
                f"Subject mismatch: expected {notes_subject}, got {meta_subject} "
                f"in doc {chunk['metadata'].get('source')}"
            )

    def test_wrong_subject_returns_nothing(self, notes_subject):
        """
        Using a made-up subject name should return nothing
        (all real subjects are filtered out).
        """
        fake_subject = "ZZZZZ_FAKE_SUBJECT_XYZ"
        results = retrieve_notes("introduction", subject=fake_subject, k=5, threshold=0.0)
        assert results == [], f"Expected no results for fake subject, got {len(results)}"

    def test_no_subject_filter_returns_results(self):
        """Without a subject filter, retrieval should return results from any subject."""
        results = retrieve_notes("introduction", subject=None, k=5, threshold=0.0)
        assert results, "Expected results with no subject filter"

    def test_subject_case_insensitive(self, notes_subject):
        """Subject matching should work regardless of case in the query."""
        lower = notes_subject.lower()
        upper = notes_subject.upper()
        r_lower = retrieve_notes("define", subject=lower, k=5, threshold=0.0)
        r_upper = retrieve_notes("define", subject=upper, k=5, threshold=0.0)
        # Both should return data (same set, possibly in different order)
        assert len(r_lower) == len(r_upper), (
            f"Case sensitivity issue: lower={len(r_lower)} upper={len(r_upper)}"
        )


# ---------------------------------------------------------------------------
# Section C — Unit filtering
# ---------------------------------------------------------------------------

class TestUnitFilter:
    def test_results_match_unit(self, notes_subject, notes_unit):
        """Every returned chunk must belong to the requested unit."""
        results = retrieve_notes(
            "explain", subject=notes_subject, unit=notes_unit, k=10, threshold=0.0
        )
        assert results, f"Expected results for unit={notes_unit}"
        for chunk in results:
            raw = chunk["metadata"].get("unit")
            normalized = _normalize_unit_meta(raw)
            assert normalized == notes_unit, (
                f"Unit mismatch: expected unit {notes_unit}, "
                f"got '{raw}' (normalized: {normalized}) "
                f"from {chunk['metadata'].get('source')}"
            )

    def test_wrong_unit_returns_nothing(self, notes_subject):
        """Unit 99 should not exist, so results should be empty."""
        results = retrieve_notes(
            "define", subject=notes_subject, unit="99", k=5, threshold=0.0
        )
        assert results == [], f"Expected no results for unit=99, got {len(results)}"

    def test_unit_prefix_format(self, notes_subject, notes_unit):
        """
        normalize_unit must handle the 'unit1' format too.
        Results using 'unit1' must equal results using '1'.
        """
        with_prefix = retrieve_notes(
            "explain", subject=notes_subject, unit=f"unit{notes_unit}", k=10, threshold=0.0
        )
        plain = retrieve_notes(
            "explain", subject=notes_subject, unit=notes_unit, k=10, threshold=0.0
        )
        ids_prefix = {c["metadata"].get("source") for c in with_prefix}
        ids_plain  = {c["metadata"].get("source") for c in plain}
        assert ids_prefix == ids_plain, (
            f"unit{notes_unit} and {notes_unit} returned different result sets.\n"
            f"With prefix: {ids_prefix}\nPlain: {ids_plain}"
        )


# ---------------------------------------------------------------------------
# Section D — Combined subject + unit filtering
# ---------------------------------------------------------------------------

class TestCombinedFilter:
    def test_combined_narrows_results(self, notes_subject, notes_unit):
        """
        Applying both subject and unit filters should return ≤ results than subject alone.
        """
        subject_only = retrieve_notes(
            "define", subject=notes_subject, k=20, threshold=0.0
        )
        subject_unit = retrieve_notes(
            "define", subject=notes_subject, unit=notes_unit, k=20, threshold=0.0
        )
        assert len(subject_unit) <= len(subject_only), (
            f"Combined filter returned MORE results ({len(subject_unit)}) "
            f"than subject-only ({len(subject_only)})"
        )

    def test_combined_all_metadata_correct(self, notes_subject, notes_unit):
        """Every result must satisfy BOTH subject and unit constraints."""
        results = retrieve_notes(
            "explain", subject=notes_subject, unit=notes_unit, k=20, threshold=0.0
        )
        for chunk in results:
            meta = chunk["metadata"]
            assert meta.get("subject", "").upper() == notes_subject.upper()
            assert _normalize_unit_meta(meta.get("unit")) == notes_unit


# ---------------------------------------------------------------------------
# Section E — Syllabus collection filtering (if available)
# ---------------------------------------------------------------------------

class TestSyllabusFilter:
    @pytest.fixture(autouse=True)
    def require_syllabus(self):
        if not collection_exists("syllabus"):
            pytest.skip("multimodal_syllabus collection not found")

    def test_syllabus_subject_filter(self, notes_subject):
        results = retrieve_syllabus("topics", subject=notes_subject, k=10, threshold=0.0)
        if not results:
            pytest.skip(f"No syllabus data for subject={notes_subject}")
        for chunk in results:
            meta_subject = chunk["metadata"].get("subject", "").upper()
            assert meta_subject == notes_subject.upper(), (
                f"Syllabus subject mismatch: {meta_subject}"
            )

    def test_syllabus_unit_filter(self, notes_subject, notes_unit):
        results = retrieve_syllabus(
            "topics", subject=notes_subject, unit=notes_unit, k=10, threshold=0.0
        )
        if not results:
            pytest.skip(f"No syllabus data for unit={notes_unit}")
        for chunk in results:
            raw = chunk["metadata"].get("unit", "")
            # Syllabus chunks for course_outcomes / books have empty unit string
            if raw:
                assert _normalize_unit_meta(raw) == notes_unit, (
                    f"Syllabus unit mismatch: {raw}"
                )


# ---------------------------------------------------------------------------
# Section F — PYQ collection filtering (if available)
# ---------------------------------------------------------------------------

class TestPYQFilter:
    @pytest.fixture(autouse=True)
    def require_pyq(self):
        if not collection_exists("pyq"):
            pytest.skip("multimodal_pyq collection not found")

    def test_pyq_subject_filter(self, notes_subject):
        from source_code.rag.search import retrieve_pyq
        results = retrieve_pyq("define", subject=notes_subject, k=10, threshold=0.0)
        if not results:
            pytest.skip(f"No PYQ data for subject={notes_subject}")
        for chunk in results:
            meta_subject = chunk["metadata"].get("subject", "").upper()
            assert meta_subject == notes_subject.upper()

    def test_pyq_unit_filter(self, notes_subject, notes_unit):
        from source_code.rag.search import retrieve_pyq
        results = retrieve_pyq(
            "define", subject=notes_subject, unit=notes_unit, k=10, threshold=0.0
        )
        if not results:
            pytest.skip(f"No PYQ data for unit={notes_unit}")
        for chunk in results:
            raw = chunk["metadata"].get("unit", "")
            if raw:
                assert _normalize_unit_meta(raw) == notes_unit