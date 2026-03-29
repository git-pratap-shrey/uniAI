"""
cross_encoder.py
────────────────
Advanced reranking module that uses a Cross-Encoder model (Qwen3-Reranker)
to evaluate the relevance of (Query, Chunk) pairs.

Unlike standard Bi-Encoders (which use cosine similarity of independent embeddings),
a Cross-Encoder processes the query and the text simultaneously, allowing it to
capture much deeper semantic interactions.

Model Details:
- Model: Qwen3-Reranker-0.6B-seq-cls
- Input: '<Instruct>: {task} \n<Query>: {query} \n<Document>: {text}'
- Output: Relevance score (Logit → Sigmoid → 0-1)
"""

import os
import sys
from source_code import models
from source_code.config import CONFIG

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import config

# Reranking is now handled centrally in models.py


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rerank_cross_encoder(
    query: str,
    chunks: list[dict],
    top_n: int | None = None,
    candidates: int | None = None,
) -> list[dict]:
    """
    Rerank a set of candidate chunks using the Cross-Encoder model.

    This is significantly more accurate than cosine similarity but also
    computationally more expensive. It should be used on a narrowed-down
    list of 'candidates' from a first-pass retrieval.

    Args:
        query:      The student's question.
        chunks:     List of chunk dictionaries from the search module.
        top_n:      Final number of chunks to return after reranking.
        candidates: Maximum number of top-similarity chunks to re-score.

    Returns:
        A list of chunks sorted by their Cross-Encoder 'final_score'.
    """
    if top_n is None:
        top_n = CONFIG["rag"]["cross_encoder"]["pipeline_top_n"]
    if candidates is None:
        candidates = CONFIG["rag"]["cross_encoder"]["candidates"]

    if not chunks:
        return []

    # 1. Pre-sort by cosine similarity, keep top candidates
    sorted_chunks = sorted(chunks, key=lambda c: c.get("similarity", 0), reverse=True)
    candidate_chunks = sorted_chunks[:candidates]

    # 2. Score each chunk using the models registry
    documents = [c["text"] for c in candidate_chunks]
    scores = models.rerank(query, documents)

    # 4. Attach scores and sort
    scored = []
    for chunk, score in zip(candidate_chunks, scores):
        scored.append({
            **chunk,
            "final_score": round(score, 4),
        })

    scored.sort(key=lambda c: c["final_score"], reverse=True)
    return scored[:top_n]
