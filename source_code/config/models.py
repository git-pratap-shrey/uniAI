"""
config/models.py
───────────
Unified AI model profiles for chat, embedding, vision, and routing tasks.
This centralizes both the selection and tuning of our diverse model stack.
"""

from .env import GEMINI_API_KEY, GROQ_API_KEY

# ------------------------------------------------------------------
# Chat Profiles
# ------------------------------------------------------------------

MODEL_CONFIGS = {
    "gemini": {
        "model": "gemini-3.1-flash-lite-preview",
        "temperature": 0.3,
        "top_p": 0.9,
        "max_tokens": 4096,
        "api_key": GEMINI_API_KEY,
    },
    "ollama": {
        "model": "qwen3:8b",
        "temperature": 0.25,
        "top_p": 0.95,
        "num_ctx": 8192,
    },
    "groq": {
        "model": "qwen/qwen3-32b",
        "temperature": 0.6,
        "top_p": 0.95,
        "max_tokens": 4096,
        "api_key": GROQ_API_KEY,
    }
}

ACTIVE_CHAT_MODEL = "gemini"


def get_active_model_config():
    """Return the currently selected chat model profile."""
    return MODEL_CONFIGS[ACTIVE_CHAT_MODEL]

# ------------------------------------------------------------------
# Embedding Configuration
# ------------------------------------------------------------------

EMBEDDING_CONFIG = {
    "provider": "ollama",
    "model": "qwen3-embedding:4B",
}

# ------------------------------------------------------------------
# Router / Classification Configuration
# ------------------------------------------------------------------

ROUTER_CONFIG = {
    "provider": "ollama",
    "model": "gemini-3-flash-preview:latest",
    "temperature": 0.0,
    "num_predict": 50,
}

# ------------------------------------------------------------------
# Vision / VLM Configuration
# ------------------------------------------------------------------

VISION_CONFIG = {
    "provider": "ollama", # ollama | huggingface
    "model": "qwen3-vl:235b-cloud",
    "hf_model_id": "Qwen/Qwen3-VL-235B-A22B-Instruct",
}
