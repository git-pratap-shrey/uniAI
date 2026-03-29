"""
embedding_router.py
───────────────────
Routes queries to a specific (Subject, Unit) pair by calculating
cosine similarity between the query embedding and pre-defined
reference embeddings for each unit.

Reference embeddings are generated during the 'Generation of unit embeddings'
maintenance task and stored in a pickle file (unit_embeddings.pkl).
"""

import os
import sys
import pickle
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import config
from pipeline.embeddings.local_embedding import embed

# Load embeddings at import time
_unit_embeddings = {}

if os.path.exists(config.UNIT_EMBEDDINGS_PATH):
    try:
        with open(config.UNIT_EMBEDDINGS_PATH, "rb") as f:
            _unit_embeddings = pickle.load(f)
    except Exception as e:
        print(f"[embedding_router] Could not load embeddings: {e}")

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Compute the cosine similarity between two vectors.

    Args:
        v1: First vector.
        v2: Second vector.

    Returns:
        Similarity score from 0.0 to 1.0 (higher = more similar).
    """
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return dot_product / (norm_v1 * norm_v2)

def route(query: str) -> tuple[str | None, str | None, float]:
    """
    Attempt to route the query to a subject and unit using embedding similarity.

    Args:
        query: The raw user query.

    Returns:
        A tuple of (subject_name, unit_string, confidence_score).
        Returns (None, None, 0.0) if no match exceeds the confidence threshold.
    """
    if not _unit_embeddings:
        return None, None, 0.0
        
    try:
        query_vec = embed([query])[0]
    except Exception as e:
        print(f"[embedding_router] LLM embed error: {e}")
        return None, None, 0.0
        
    best_score = -1.0
    best_match = None
    
    for key, vec in _unit_embeddings.items():
        sim = cosine_similarity(query_vec, vec)
        if sim > best_score:
            best_score = sim
            best_match = key
            
    if best_score > config.EMBEDDING_ROUTER_THRESHOLD and best_match:
        # key format: SUBJECT_UNIT, e.g., CYBER_SECURITY_3
        parts = best_match.rsplit("_", 1)
        if len(parts) == 2:
            return parts[0], parts[1], float(best_score)
            
    return None, None, float(best_score)
