import os
from dotenv import load_dotenv

# Load environment variables from a .env file if present
load_dotenv()

# ------------------------------------------------------------------
# OLLAMA CONFIGURATION
# ------------------------------------------------------------------

# Base URL for Ollama instance
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# ------------------------------------------------------------------
# MODEL CONFIGURATION
# ------------------------------------------------------------------

# Embedding Model
# Used for generating vector embeddings for text search
MODEL_EMBEDDING = os.getenv("MODEL_EMBEDDING", "mxbai-embed-large")

# Vision / OCR Model
# Used for optical character recognition and image analysis
# User requested "qwen3 vl 235b cloud"
MODEL_VISION = os.getenv("MODEL_VISION", "qwen3 vl 235b cloud")

# Chat / Generative Model
# Used for RAG chat and general text generation
# User requested "qwen3 vl 235b cloud"
MODEL_CHAT = os.getenv("MODEL_CHAT", "qwen3 vl 235b cloud")

# ------------------------------------------------------------------
# PATH CONFIGURATION
# ------------------------------------------------------------------

# Default paths - can be overridden by env vars
BASE_DATA_DIR = os.getenv("BASE_DATA_DIR", r"D:\CODE-workingBuild\uniAI\source_code\data\year_2")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", r"D:\CODE-workingBuild\uniAI\source_code\chroma")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "multimodal_notes")
