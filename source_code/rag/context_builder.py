"""
context_builder.py
──────────────────
Transforms raw data (retrieved chunks, history) into structured text
ready for LLM consumption.

This module acts as the 'Prompt Engineering' layer, ensuring that
metadata is preserved and presented in a way that helps the LLM
cite sources accurately.
"""


def build_context(chunks: list[dict]) -> str:
    """
    Convert a list of retrieved chunks into a single formatted context string.

    Each chunk is prefixed with a metadata header (Source #, Title, Unit, Relevance)
    so the LLM can distinguish between different sources.

    Args:
        chunks: List of dictionaries containing "text" and "metadata".

    Returns:
        A formatted string with chunks separated by dividers.
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
    Format the recent conversation history into a compact text block.

    Args:
        history: List of {"role": "user"|"assistant", "content": "..."} dictionaries.

    Returns:
        A string representing the previous conversation turns.
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
    Format retrieved chunks into human-readable citation lines for CLI display.

    Args:
        chunks: List of ranked result dictionaries.

    Returns:
        A list of strings like: "filename.pdf (p.1) | Unit 3 | similarity=0.72"
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
