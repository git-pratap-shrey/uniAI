import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from a .env file if present
load_dotenv()

# ------------------------------------------------------------------
# OLLAMA CONFIGURATION
# ------------------------------------------------------------------

# Base URL for Ollama instance (use https://api.ollama.com for cloud vision)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
# Local Ollama — always localhost, used for embeddings and local chat
OLLAMA_LOCAL_URL = os.getenv("OLLAMA_LOCAL_URL", "http://localhost:11434")
# Optional API key for Ollama cloud (paid tier)
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "")

# ------------------------------------------------------------------
# MODEL CONFIGURATION
# ------------------------------------------------------------------

# Embedding Model
# Used for generating vector embeddings for text search
# qwen3-embedding:4B has a 32K token context window (vs 512 for mxbai-embed-large)
MODEL_EMBEDDING = os.getenv("MODEL_EMBEDDING", "qwen3-embedding:4B")

# Vision / OCR Model Backend
# Options: "ollama" | "huggingface"
#   ollama      -> Ollama model tag (local or cloud Ollama)
#   huggingface -> HuggingFace Inference API (cloud, no local GPU needed)
MODEL_VISION_BACKEND = os.getenv("MODEL_VISION_BACKEND", "ollama")

# Vision model name — interpreted differently per backend:
#   ollama      -> Ollama model tag, e.g. "qwen3-vl:235b-cloud" or "llava:13b"
#   huggingface -> HuggingFace repo ID, e.g. "Qwen/Qwen3-VL-235B-A22B-Instruct"
MODEL_VISION = os.getenv("MODEL_VISION", "qwen3-vl:235b-cloud")

# HuggingFace cloud Inference API settings
# Get your token from: https://huggingface.co/settings/tokens
HF_TOKEN = os.getenv("HF_TOKEN")                                              # required
MODEL_VISION_HF = os.getenv("MODEL_VISION_HF", "Qwen/Qwen3-VL-235B-A22B-Instruct")

# Chat / Generative Model
# Used for RAG chat and general text generation
MODEL_CHAT = os.getenv("MODEL_CHAT", "qwen2.5-coder:3b")

# Router / Classification Model
# Fast local model used specifically for extracting keywords and context switching
MODEL_ROUTER = os.getenv("MODEL_ROUTER", "qwen3.5:4B")

# ------------------------------------------------------------------
# RETRIEVAL CONFIGURATION
# ------------------------------------------------------------------

# Minimum cosine similarity to keep a search result (0.0 = keep all, 1.0 = exact match only)
# Distance in ChromaDB cosine space = 1.0 - similarity, so threshold 0.3 → max_distance 0.7
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.35"))

# Minimum similarity the top-ranked chunk must have to use retrieved context.
# If the best chunk falls below this, the pipeline falls back to generic mode.
MIN_STRONG_SIM = float(os.getenv("MIN_STRONG_SIM", "0.6"))

# ------------------------------------------------------------------
# PATH CONFIGURATION
# ------------------------------------------------------------------

# Resolve paths relative to this file so defaults work on any machine.
# Override via .env for custom locations.
_SOURCE_CODE_DIR = Path(__file__).parent

BASE_DATA_DIR = os.getenv("BASE_DATA_DIR", str(_SOURCE_CODE_DIR / "data" / "year_2"))
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", str(_SOURCE_CODE_DIR / "chroma"))
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "multimodal_notes")
CHROMA_SYLLABUS_COLLECTION_NAME = os.getenv("CHROMA_SYLLABUS_COLLECTION_NAME", "multimodal_syllabus")
CHROMA_PYQ_COLLECTION_NAME = os.getenv("CHROMA_PYQ_COLLECTION_NAME", "multimodal_pyq")

# Minimum OCR confidence to ingest a chunk (0.0 = ingest everything, 1.0 = perfect only)
# Chunks below this threshold are likely illegible and hurt retrieval quality
MIN_INGEST_CONFIDENCE = float(os.getenv("MIN_INGEST_CONFIDENCE", "0.3"))