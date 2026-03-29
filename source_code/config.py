"""
source_code/config.py
──────────────────
⚠️  COMPATIBILITY LAYER ⚠️
This file is now a thin wrapper around the modular config package.
Please update imports to: from source_code.config import CONFIG
"""

import os
from pathlib import Path

# Import the new modular config
from .config.main import *
from .config.models import MODEL_CONFIGS, ACTIVE_CHAT_MODEL, get_active_model_config
from .config.rag import RAG_CONFIG as _RAG_CONFIG
from .config.rag import CROSS_ENCODER_CONFIG as _CE_CONFIG

# --- Map new structure back to old top-level variables ---

# Ollama / Environment
# Already imported via from .config.main import *

# Model Configuration
MODEL_EMBEDDING = "qwen3-embedding:4B" # from config.models.EMBEDDING_CONFIG["model"]
EMBEDDING_PROVIDER = "ollama"

VISION_PROVIDER = "ollama"
MODEL_VISION = "qwen3-vl:235b-cloud"

# Use values from active chat model if available
_active_chat = get_active_model_config()
MODEL_CHAT = _active_chat["model"]
CHAT_PROVIDER = ACTIVE_CHAT_MODEL # Maps gemini|ollama|groq

# Router
MODEL_ROUTER = "qwen3.5:4B"
ROUTER_PROVIDER = "ollama"

# Retrieval Configuration
SIMILARITY_THRESHOLD = _RAG_CONFIG["similarity_threshold"]
MIN_STRONG_SIM = _RAG_CONFIG["min_strong_sim"]

# Cross-Encoder Configuration
CROSS_ENCODER_MODEL = _CE_CONFIG["model"]
MIN_CROSS_SCORE = _CE_CONFIG["min_score"]
CROSS_ENCODER_CANDIDATES = _CE_CONFIG["candidates"]
PIPELINE_CROSS_RERANK_TOP_N = _CE_CONFIG["pipeline_top_n"]

# Path Configuration (already imported via main -> paths)
# BASE_DATA_DIR, CHROMA_DB_PATH, etc.

# RAG Pipeline tweaks
PIPELINE_NOTES_K = _RAG_CONFIG["notes_k"]
PIPELINE_SYLLABUS_K = _RAG_CONFIG["syllabus_k"]
PIPELINE_RERANK_TOP_N = _RAG_CONFIG["rerank_top_n"]

RERANK_DEFAULT_TOP_N = _RAG_CONFIG["rerank_top_n"]
RERANK_UNIT_MATCH_BOOST = 1.15
RERANK_SYLLABUS_PENALTY = 0.90

OLLAMA_ROUTER_TEMPERATURE = CONFIG["rag"]["router_temperature"]
OLLAMA_ROUTER_NUM_PREDICT = CONFIG["rag"]["router_num_predict"]

# Search settings
SEARCH_NOTES_K_DEFAULT = _RAG_CONFIG["notes_k"]
SEARCH_SYLLABUS_K_DEFAULT = _RAG_CONFIG["syllabus_k"]
SEARCH_PYQ_K_DEFAULT = _RAG_CONFIG["pyq_k"]
SEARCH_PYQ_THRESHOLD = _RAG_CONFIG["pyq_threshold"]

SEARCH_ALL_NOTES_K = CONFIG["rag"]["all_notes_k"]
SEARCH_ALL_SYLLABUS_K = CONFIG["rag"]["all_syllabus_k"]