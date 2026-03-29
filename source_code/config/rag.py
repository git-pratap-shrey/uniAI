"""
config/rag.py
─────────
Centralizes all RAG (Retrieval-Augmented Generation) hyperparameters
and pipeline tuning parameters.
"""

# Retrieval & Search thresholds
RAG_CONFIG = {
    "similarity_threshold": 0.35,
    "min_strong_sim": 0.6,
    
    # Default retrieval K (counts)
    "notes_k": 8,
    "syllabus_k": 7,
    "pyq_k": 5,
    
    # Defaults used in retrieval functions
    "notes_k_default": 8,
    "syllabus_k_default": 7,
    "pyq_k_default": 5,
    "pyq_threshold": 0.60,
    
    # Counts for retrieve_all (interleaved)
    "all_notes_k": 6,
    "all_syllabus_k": 7,
    
    # Reranking counts
    "rerank_top_n": 7,
}

# Cross-Encoder Reranker settings
CROSS_ENCODER_CONFIG = {
    "model": "tomaarsen/Qwen3-Reranker-0.6B-seq-cls",
    "min_score": 0.65,
    "candidates": 6,
    "pipeline_top_n": 4, # Top N after cross-reranking
}

# RAG Pipeline tweaks
MAX_HISTORY_TURNS = 4

# Router logic
ROUTER_TEMPERATURE = 0.0
ROUTER_NUM_PREDICT = 10

# Hybrid router thresholds
KEYWORD_MIN_SCORE = 2
EMBEDDING_ROUTER_THRESHOLD = 0.55

# Ingestion settings
MIN_INGEST_CONFIDENCE = 0.3

# Query Expander
QUERY_EXPANDER_MAX_KEYWORDS = 6
