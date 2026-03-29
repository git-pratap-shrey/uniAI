"""
config/paths.py
───────────────
Centralizes all filesystem paths and database connections using pathlib.
Ensures path consistency across different operating systems.
"""

import os
from pathlib import Path
from .env import OLLAMA_BASE_URL, OLLAMA_LOCAL_URL

# Project root (source_code/)
BASE_DIR = Path(__file__).parent.parent

# Data directory
BASE_DATA_DIR = os.getenv("BASE_DATA_DIR", str(BASE_DIR / "data" / "year_2"))

# Database paths
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", str(BASE_DIR / "chroma"))
UNIT_EMBEDDINGS_PATH = str(BASE_DIR / "pipeline" / "embeddings" / "unit_embeddings.pkl")

# Mapping & Meta paths
ALIASES_FILE_PATH = str(BASE_DIR / "data" / "subject_aliases.json")
KEYWORDS_FILE_PATH = str(BASE_DIR / "data" / "subject_keywords.json")

# Collection Names
CHROMA_COLLECTION_NAME = "multimodal_notes"
CHROMA_SYLLABUS_COLLECTION_NAME = "multimodal_syllabus"
CHROMA_PYQ_COLLECTION_NAME = "multimodal_pyq"
