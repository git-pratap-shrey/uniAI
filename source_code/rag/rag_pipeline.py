"""
rag_pipeline.py
───────────────
The central orchestration module for the Retrieval-Augmented Generation (RAG) system.

This module coordinates the entire lifecycle of a user query:
1.  **Query Expansion**: Enriches the query with context and keywords.
2.  **Routing**: Determines the relevant subject and unit.
3.  **Retrieval**: Fetches candidate chunks from multiple vector collections.
4.  **Reranking**: Refines the order of retrieved chunks using a cross-encoder.
5.  **Context Construction**: Formats the best chunks and history for the LLM.
6.  **Generation**: Calls the LLM (local or cloud) to produce the final answer.

The pipeline is designed to be modular, with each stage delegated to separate modules.
"""

import os
import sys
import re

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from source_code.config import CONFIG
from source_code import models

from rag.hybrid_router import route as hybrid_route
from rag.search import retrieve_notes, retrieve_syllabus
# from rag.reranker import rerank              # heuristic — kept as fallback
from rag.cross_encoder import rerank_cross_encoder
from rag.context_builder import build_context, build_history_block
from rag.query_expander import expand_query
import prompts

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_HISTORY_TURNS = CONFIG["rag"]["history_limit"]  # user+assistant pairs kept in context

FOLLOWUP_PATTERNS = [
    r"^repeat",
    r"^again",
    r"^summarize",
    r"\bprevious\b",
    r"\bearlier\b",
    r"\bexplain that again\b",
]

GENERIC_PATTERNS = [
    r"\bwrite code\b",
    r"\bimplement\b",
    r"\bbeyond syllabus\b",
]


# ---------------------------------------------------------------------------
# Intent helpers
# ---------------------------------------------------------------------------

def _is_followup(query: str) -> bool:
    """
    Check if the user's query is likely a follow-up to the previous context.

    Args:
        query: The raw user query.

    Returns:
        True if the query matches common follow-up patterns, False otherwise.
    """
    q = query.strip().lower()
    for pattern in FOLLOWUP_PATTERNS:
        if re.search(pattern, q):
            return True
    return False


def _is_unit_overview(query: str, unit: str | None) -> bool:
    """
    Detect if the user is asking for a general overview of a unit's topics.

    Args:
        query: User input query.
        unit:  The unit being discussed.

    Returns:
        True if "list topics" or similar intent is detected.
    """
    if not unit:
        return False
    q = query.lower()
    overview_signals = ["topics", "list", "what is in", "overview", "cover", "syllabus"]
    return any(s in q for s in overview_signals)


def _detect_mode(query: str) -> str:
    """
    Determine whether to use 'syllabus-aware' mode or 'generic' chat mode.

    Args:
        query: User input query.

    Returns:
        "syllabus" for retrieval-based answers, "generic" for general LLM knowledge.
    """
    q = query.lower()
    for pattern in GENERIC_PATTERNS:
        if re.search(pattern, q):
            return "generic"
    return "syllabus"


def _trim_history(history: list[dict]) -> list[dict]:
    """
    Keep only the most recent turns in the conversation history to save tokens.

    Args:
        history: The full conversation history.

    Returns:
        A trimmed list of history turns.
    """
    return history[-(MAX_HISTORY_TURNS * 2):]


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _generate(prompt: str) -> str:
    """
    Execute the generation step by calling the centralized models registry.

    Args:
        prompt: The fully constructed system and user prompt.

    Returns:
        The generated text response.
    """
    return models.chat(
        prompt=prompt,
        temperature=CONFIG["model"].get("temperature", 0.3),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def answer_query(
    query: str,
    history: list[dict] | None = None,
    session_subject: str | None = None,
) -> dict:
    """
    The main entry point for the RAG system to process a query and return an answer.

    Args:
        query:           The student's question.
        history:         A list of previous turns (role, content).
        session_subject: An optional subject lock for the current session.

    Returns:
        A dictionary containing:
          - answer: The final generated text.
          - subject/unit: The detected routing metadata.
          - mode: The intent mode (syllabus vs. generic).
          - sources: Human-readable source citations.
          - chunks: The raw ranked chunks used in the context.
    """
    history = _trim_history(history or [])

    expanded_query = expand_query(query)

    # ── 1 & 2. Hybrid Routing (Subject & Unit) ────────────────────────────
    route_res = hybrid_route(expanded_query, session_subject=session_subject)
    subject = route_res.subject
    unit = route_res.unit

    # ── 3. Detect mode ────────────────────────────────────────────────────
    mode = _detect_mode(expanded_query)

    # ── 4. Handle followup — skip retrieval ───────────────────────────────
    if _is_followup(expanded_query) and history:
        history_block = build_history_block(history)
        prompt = prompts.rag_answer(
            query=expanded_query,
            notes_context="",
            history_block=history_block,
            mode=mode,
            subject=subject,
        )
        answer = _generate(prompt)
        return {
            "answer": answer,
            "subject": subject,
            "unit": unit,
            "mode": mode,
            "sources": [],
            "chunks": [],
            "expanded_query": expanded_query,
        }

    # ── 5. Retrieve ───────────────────────────────────────────────────────
    note_chunks = retrieve_notes(expanded_query, subject=subject, unit=unit, k=CONFIG["rag"]["notes_k"])

    # Always retrieve syllabus chunks to give the cross-encoder more candidates
    syllabus_chunks = retrieve_syllabus(expanded_query, subject=subject, unit=unit, k=CONFIG["rag"]["syllabus_k"])

    all_chunks = note_chunks + syllabus_chunks

    # ── 6. Cross-encoder rerank ───────────────────────────────────────────
    ranked = rerank_cross_encoder(
        expanded_query,
        all_chunks,
        top_n=CONFIG["rag"]["cross_encoder"]["pipeline_top_n"],
        candidates=CONFIG["rag"]["cross_encoder"]["candidates"],
    )

    if not ranked:
        mode = "generic"
    elif ranked[0]["final_score"] < CONFIG["rag"]["cross_encoder"]["min_score"]:
        mode = "generic"
        ranked = []

    # ── 7. Build context ──────────────────────────────────────────────────
    notes_context = build_context(ranked)
    history_block = build_history_block(history)

    # ── 8. Build prompt ───────────────────────────────────────────────────
    prompt = prompts.rag_answer(
        query=expanded_query,
        notes_context=notes_context,
        history_block=history_block,
        mode=mode,
        subject=subject,
    )

    # ── 9. Generate ───────────────────────────────────────────────────────
    from rag.context_builder import format_sources_for_display
    answer = _generate(prompt)

    return {
        "answer": answer,
        "subject": subject,
        "unit": unit,
        "mode": mode,
        "sources": format_sources_for_display(ranked),
        "chunks": ranked,
        "expanded_query": expanded_query,
    }
