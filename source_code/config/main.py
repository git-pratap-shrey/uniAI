"""
config/main.py
───────────
The central access point for the entire configuration system.
Combines environment variables, model profiles, RAG hyperparameters,
and filesystem paths into a single, structured CONFIG dictionary.
"""

from .env import *
from .models import (
    get_active_model_config, 
    MODEL_CONFIGS, 
    EMBEDDING_CONFIG, 
    ROUTER_CONFIG, 
    VISION_CONFIG,
    ACTIVE_CHAT_MODEL
)
from .rag import RAG_CONFIG, CROSS_ENCODER_CONFIG, MAX_HISTORY_TURNS, KEYWORD_MIN_SCORE, EMBEDDING_ROUTER_THRESHOLD, MIN_INGEST_CONFIDENCE, QUERY_EXPANDER_MAX_KEYWORDS
from .paths import *

# The Master Configuration Structure
CONFIG = {
    # App-level
    "env": APP_ENV,
    
    # Machine-specific / Env vars (Flattened for easy migration)
    "OLLAMA_BASE_URL": OLLAMA_BASE_URL,
    "OLLAMA_LOCAL_URL": OLLAMA_LOCAL_URL,
    "GEMINI_API_KEY": GEMINI_API_KEY,
    "GROQ_API_KEY": GROQ_API_KEY,
    "HF_TOKEN": HF_TOKEN,
    "USE_OLLAMA_CLOUD": USE_OLLAMA_CLOUD,

    "model": get_active_model_config(),
    "providers": {
        "chat": ACTIVE_CHAT_MODEL,
        "embedding": EMBEDDING_CONFIG["provider"],
        "embedding_model": EMBEDDING_CONFIG["model"],
        "router": ROUTER_CONFIG["provider"],
        "router_model": ROUTER_CONFIG["model"],
        "vision": VISION_CONFIG["provider"],
        "vision_model": VISION_CONFIG["model"],
    },
    "rag": {
        **RAG_CONFIG,
        "history_limit": MAX_HISTORY_TURNS,
        "cross_encoder": CROSS_ENCODER_CONFIG,
        "router_model": ROUTER_CONFIG["model"],
        "router_temperature": ROUTER_CONFIG["temperature"],
        "router_num_predict": ROUTER_CONFIG["num_predict"],
        "keywords": {
            "min_score": KEYWORD_MIN_SCORE,
            "max_expander": QUERY_EXPANDER_MAX_KEYWORDS,
        },
        "embedding_router_threshold": EMBEDDING_ROUTER_THRESHOLD,
    },
    "paths": {
        "base_data": BASE_DATA_DIR,
        "chroma": CHROMA_DB_PATH,
        "unit_embeddings": UNIT_EMBEDDINGS_PATH,
        "aliases": ALIASES_FILE_PATH,
        "keywords": KEYWORDS_FILE_PATH,
        "collections": {
            "notes": CHROMA_COLLECTION_NAME,
            "syllabus": CHROMA_SYLLABUS_COLLECTION_NAME,
            "pyq": CHROMA_PYQ_COLLECTION_NAME,
        }
    },
    "ingest": {
        "min_confidence": MIN_INGEST_CONFIDENCE,
    }
}
