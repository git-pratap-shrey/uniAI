"""
config/env.py
─────────────
Loads all environment variables from .env.
ONLY secrets and machine-specific values should live here.
"""

import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Secrets
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "")

# Machine-specific settings
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_LOCAL_URL = os.getenv("OLLAMA_LOCAL_URL", "http://localhost:11434")
USE_OLLAMA_CLOUD = os.getenv("USE_OLLAMA_CLOUD", "True").lower() == "true"

# App Environment (dev | prod)
APP_ENV = os.getenv("APP_ENV", "dev")
