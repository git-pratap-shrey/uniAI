"""
reranker.py
───────────
A heuristic-based reranking module used as a fast, rule-based alternative
or supplement to the neural Cross-Encoder.

This module adjusts the 'similarity' score from the vector search using
metadata signals like OCR confidence, unit matching, and document type.

Boosting Logic:
1. **OCR Confidence**: Multiplies score by a factor (0.5 to 1.0) based on extraction quality.
2. **Unit Match**: Significant boost if the chunk's unit matches the predicted unit.
3. **Doc Type**: Slight preference for lecture notes over syllabus chunks.
"""


import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import config

def rerank(
    chunks: list[dict],
    predicted_unit: str | None = None,
    top_n: int = None,
) -> list[dict]:
    """
    Apply heuristic boosts and penalties to a list of retrieved chunks.

    Args:
        chunks:         List of chunk dicts (must have "similarity" and "metadata").
        predicted_unit: The unit identifier detected by the router.
        top_n:          Number of top-scored chunks to return.

    Returns:
        A list of chunks sorted by 'final_score' in descending order.
    """
    if top_n is None:
        top_n = config.RERANK_DEFAULT_TOP_N

    scored = []

    for chunk in chunks:
        meta = chunk.get("metadata", {})

        base_sim = chunk.get("similarity", 0.0)

        # Confidence boost: scales similarity by OCR quality.
        # confidence=1.0 → multiplier=1.0, confidence=0.5 → multiplier=0.75
        raw_confidence = float(meta.get("confidence", 0.8))
        confidence_mult = 0.5 + (raw_confidence / 2.0)

        # Unit match boost
        unit_mult = 1.0
        if predicted_unit:
            chunk_unit = str(meta.get("unit", "")).lower().strip()
            # Match "3", "unit3", or "unit 3" against predicted "3"
            normalized = chunk_unit.replace("unit", "").strip()
            if normalized == str(predicted_unit):
                unit_mult = config.RERANK_UNIT_MATCH_BOOST

        # Doc type preference: slight penalty for syllabus in general queries
        doc_type = meta.get("document_type", "")
        type_mult = config.RERANK_SYLLABUS_PENALTY if doc_type == "syllabus" else 1.0

        final_score = base_sim * confidence_mult * unit_mult * type_mult

        scored.append({**chunk, "final_score": round(final_score, 4)})

    scored.sort(key=lambda c: c["final_score"], reverse=True)
    return scored[:top_n]
