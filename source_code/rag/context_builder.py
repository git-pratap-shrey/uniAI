"""
context_builder.py
──────────────────
Formats retrieved chunks into clean, LLM-ready context blocks.

No retrieval logic. No scoring logic. Pure formatting.
"""


def build_context(chunks: list[dict]) -> str:
    """
    Format ranked note/syllabus chunks into a context string for the LLM.

    Each chunk block shows source metadata as a header so the LLM can
    reference where the information came from.
    """
    if not chunks:
        return ""

    parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata", {})
        source = meta.get("source", "unknown")
        unit = meta.get("unit", "?")
        title = meta.get("title", "")
        doc_type = meta.get("document_type", "")
        sim = chunk.get("final_score", chunk.get("similarity", 0))

        header_parts = [f"Source {i}"]
        if title and title.lower() not in ("unknown", ""):
            header_parts.append(title)
        if unit and unit.lower() not in ("unknown", ""):
            header_parts.append(f"Unit {unit}")
        if doc_type:
            header_parts.append(doc_type)
        header_parts.append(f"relevance={sim:.2f}")

        header = " | ".join(header_parts)
        parts.append(f"[{header}]\n{chunk['text']}")

    return "\n\n---\n\n".join(parts)


def build_history_block(history: list[dict]) -> str:
    """
    Format conversation history into a compact context block.

    history is a list of {"role": "user"|"assistant", "content": "..."}
    """
    if not history:
        return ""

    lines = ["Previous conversation:"]
    for turn in history:
        role = turn.get("role", "user").upper()
        content = turn.get("content", "").strip()
        if content:
            lines.append(f"{role}: {content}")

    return "\n".join(lines)


def format_sources_for_display(chunks: list[dict]) -> list[str]:
    """
    Format chunks into human-readable source lines for CLI display.

    Returns a list of strings like:
      "python_unit1.pdf (p.1) | Unit unit1 | similarity=0.72"
    """
    lines = []
    for chunk in chunks:
        meta = chunk.get("metadata", {})
        source = meta.get("source", "unknown")
        page = meta.get("page_start", "?")
        unit = meta.get("unit", "?")
        sim = chunk.get("final_score", chunk.get("similarity", 0))
        lines.append(f"{source} (p.{page}) | Unit {unit} | similarity={sim:.2f}")
    return lines
