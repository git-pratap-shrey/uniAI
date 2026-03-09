"""
rag_pipeline.py
───────────────
Orchestrates the full RAG pipeline.

Flow:
  1. detect_subject(query)
  2. detect_unit(query)
  3. retrieve_notes(query, subject, unit)
  4. retrieve_syllabus(query, subject, unit)   ← only for topic-listing queries
  5. rerank(notes + syllabus chunks, unit)
  6. build_context(ranked_chunks)
  7. build_history_block(history)
  8. detect_mode(query)
  9. build prompt
  10. call generation model
  11. return answer + metadata

This file contains orchestration only — no business logic.
"""

import os
import sys
import re

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import config
import ollama

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

MAX_HISTORY_TURNS = config.MAX_HISTORY_TURNS  # user+assistant pairs kept in context

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
    q = query.strip().lower()
    for pattern in FOLLOWUP_PATTERNS:
        if re.search(pattern, q):
            return True
    return False


def _is_unit_overview(query: str, unit: str | None) -> bool:
    """Detect "list topics in unit 3" style queries."""
    if not unit:
        return False
    q = query.lower()
    overview_signals = ["topics", "list", "what is in", "overview", "cover", "syllabus"]
    return any(s in q for s in overview_signals)


def _detect_mode(query: str) -> str:
    q = query.lower()
    for pattern in GENERIC_PATTERNS:
        if re.search(pattern, q):
            return "generic"
    return "syllabus"


def _trim_history(history: list[dict]) -> list[dict]:
    return history[-(MAX_HISTORY_TURNS * 2):]


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _generate(prompt: str) -> str:
    """Call the configured generation model.

    Args:
        prompt: The full prompt to send.
    """
    try:
        if config.MODEL_CHAT.startswith("gemini") and not getattr(config, "USE_OLLAMA_CLOUD", False):
            import google.generativeai as genai
            if not getattr(config, "GEMINI_API_KEY", ""):
                return "⚠ GEMINI_API_KEY not set in config."
            genai.configure(api_key=config.GEMINI_API_KEY)
            model = genai.GenerativeModel(config.MODEL_CHAT)
            response = model.generate_content(prompt)
            return response.text or "⚠ Empty response from Gemini."

        else:
            if getattr(config, "USE_OLLAMA_CLOUD", False):
                headers = {"Authorization": f"Bearer {config.OLLAMA_API_KEY}"} if getattr(config, "OLLAMA_API_KEY", "") else None
                client = ollama.Client(host=config.OLLAMA_BASE_URL, headers=headers)
            else:
                client = ollama.Client(host=config.OLLAMA_LOCAL_URL)
                
            response = client.chat(
                model=config.MODEL_CHAT,
                messages=[{"role": "user", "content": prompt}],
                 options={"num_ctx": config.OLLAMA_CHAT_NUM_CTX, "think": False, "temperature": config.OLLAMA_CHAT_TEMPERATURE},
            )
            return response["message"]["content"]

    except Exception as e:
        return f"⚠ Generation error: {e}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def answer_query(
    query: str,
    history: list[dict] | None = None,
    session_subject: str | None = None,
) -> dict:
    """
    Run the full RAG pipeline for a query.

    Args:
        query:           The student's question.
        history:         List of {"role": ..., "content": ...} turns.
        session_subject: Subject locked for this session (from chat_cli).

    Returns a dict:
    {
        "answer":   str,
        "subject":  str | None,
        "unit":     str | None,
        "mode":     str,
        "sources":  list[str],
        "chunks":   list[dict],   ← ranked chunks (for CLI display)
    }
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
    note_chunks = retrieve_notes(expanded_query, subject=subject, unit=unit, k=config.PIPELINE_NOTES_K)

    # Always retrieve syllabus chunks to give the cross-encoder more candidates
    syllabus_chunks = retrieve_syllabus(expanded_query, subject=subject, unit=unit, k=config.PIPELINE_SYLLABUS_K)

    all_chunks = note_chunks + syllabus_chunks

    # ── 6. Cross-encoder rerank ───────────────────────────────────────────
    ranked = rerank_cross_encoder(
        expanded_query,
        all_chunks,
        top_n=config.PIPELINE_CROSS_RERANK_TOP_N,
        candidates=config.CROSS_ENCODER_CANDIDATES,
    )

    if not ranked:
        mode = "generic"
    elif ranked[0]["final_score"] < config.MIN_CROSS_SCORE:
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
