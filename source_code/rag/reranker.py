"""
reranker.py
───────────
Adjusts chunk scores using metadata signals after initial retrieval.

Boosting signals:
  - confidence   (OCR quality stored at ingestion, 0.0–1.0)
  - unit match   (bump if chunk unit matches predicted unit)
  - doc_type     (slight preference for notes over syllabus in mixed results)

No database calls. Pure scoring logic.
"""


def rerank(
    chunks: list[dict],
    predicted_unit: str | None = None,
    top_n: int = 5,
) -> list[dict]:
    """
    Re-score and sort chunks by relevance.

    Each chunk is expected to have:
      chunk["similarity"]          — cosine similarity (0–1)
      chunk["metadata"]["confidence"]   — OCR confidence (0–1)
      chunk["metadata"]["unit"]         — unit string
      chunk["metadata"]["document_type"] — e.g. "printed_notes", "syllabus"

    Returns top_n chunks sorted by final_score descending,
    with "final_score" added to each chunk dict.
    """
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
                unit_mult = 1.15

        # Doc type preference: slight penalty for syllabus in general queries
        doc_type = meta.get("document_type", "")
        type_mult = 0.90 if doc_type == "syllabus" else 1.0

        final_score = base_sim * confidence_mult * unit_mult * type_mult

        scored.append({**chunk, "final_score": round(final_score, 4)})

    scored.sort(key=lambda c: c["final_score"], reverse=True)
    return scored[:top_n]
