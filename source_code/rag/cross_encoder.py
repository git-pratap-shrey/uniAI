"""
cross_encoder.py
────────────────
Cross-encoder reranker using Qwen3-Reranker-0.6B (sequence classification).

Uses the pre-converted `tomaarsen/Qwen3-Reranker-0.6B-seq-cls` checkpoint
loaded with AutoModelForSequenceClassification. The model is eagerly loaded
onto GPU at import time and kept in memory.

Public API
----------
  rerank_cross_encoder(query, chunks, top_n, candidates) → list[Chunk]

Each returned chunk has a "final_score" key (sigmoid-normalized, 0–1).
"""

import os
import sys
import time

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import config

# ---------------------------------------------------------------------------
# Model loading — eager, one-time, on GPU
# ---------------------------------------------------------------------------

_MODEL_ID = config.CROSS_ENCODER_MODEL
_MAX_LENGTH = 8192

_device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[cross_encoder] Loading {_MODEL_ID} on {_device} …")
_start = time.time()

_tokenizer = AutoTokenizer.from_pretrained(_MODEL_ID, padding_side="left")
_model = AutoModelForSequenceClassification.from_pretrained(
    _MODEL_ID,
    torch_dtype=torch.float16,
).to(_device).eval()

print(f"[cross_encoder] Model loaded in {time.time() - _start:.1f}s")


# ---------------------------------------------------------------------------
# Prompt formatting (follows Qwen3-Reranker chat template)
# ---------------------------------------------------------------------------

_TASK_INSTRUCTION = (
    "Given a student's academic query, retrieve relevant lecture notes or "
    "syllabus passages that answer the query"
)


def _format_pair(query: str, document: str, instruction: str | None = None) -> str:
    """Format a (query, document) pair using the Qwen3-Reranker chat template."""
    if instruction is None:
        instruction = _TASK_INSTRUCTION

    prefix = (
        '<|im_start|>system\n'
        'Judge whether the Document meets the requirements based on the '
        'Query and the Instruct provided. Note that the answer can only '
        'be "yes" or "no".<|im_end|>\n'
        '<|im_start|>user\n'
    )
    suffix = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'

    return f"{prefix}<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}{suffix}"


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
    Rerank chunks using the cross-encoder model.

    Steps:
      1. Sort chunks by cosine similarity (descending).
      2. Take top `candidates` chunks.
      3. Score each (query, chunk) pair with the cross-encoder.
      4. Return top `top_n` chunks sorted by cross-encoder score.

    Each returned chunk has "final_score" (sigmoid, 0–1) and
    "cross_score_raw" (raw logit) added.

    Args:
        query:      The student's question.
        chunks:     List of chunk dicts from search.py (must have "text" key).
        top_n:      Number of chunks to return (default from config).
        candidates: Max pairs to score (default from config).
    """
    if top_n is None:
        top_n = config.PIPELINE_CROSS_RERANK_TOP_N
    if candidates is None:
        candidates = config.CROSS_ENCODER_CANDIDATES

    if not chunks:
        return []

    # 1. Pre-sort by cosine similarity, keep top candidates
    sorted_chunks = sorted(chunks, key=lambda c: c.get("similarity", 0), reverse=True)
    candidate_chunks = sorted_chunks[:candidates]

    # 2. Format pairs
    pairs = [
        _format_pair(query, chunk["text"])
        for chunk in candidate_chunks
    ]

    # 3. Tokenize and score
    with torch.no_grad():
        inputs = _tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=_MAX_LENGTH,
            return_tensors="pt",
        ).to(_device)

        logits = _model(**inputs).logits.squeeze(-1)

        # Handle single-item case (squeeze removes the batch dimension)
        if logits.dim() == 0:
            logits = logits.unsqueeze(0)

        scores = logits.sigmoid().cpu().tolist()

    # 4. Attach scores and sort
    scored = []
    for chunk, score, logit in zip(candidate_chunks, scores, logits.cpu().tolist()):
        scored.append({
            **chunk,
            "final_score": round(score, 4),
            "cross_score_raw": round(logit, 4),
        })

    scored.sort(key=lambda c: c["final_score"], reverse=True)
    return scored[:top_n]
