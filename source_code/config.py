import os
from dotenv import load_dotenv

# Load environment variables from a .env file if present
load_dotenv()

# ------------------------------------------------------------------
# OLLAMA CONFIGURATION
# ------------------------------------------------------------------

# Base URL for Ollama instance (use https://api.ollama.com for cloud)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
# Optional API key for Ollama cloud (paid tier)
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "")

# ------------------------------------------------------------------
# MODEL CONFIGURATION
# ------------------------------------------------------------------

# Embedding Model
# Used for generating vector embeddings for text search
MODEL_EMBEDDING = os.getenv("MODEL_EMBEDDING", "mxbai-embed-large")

# Vision / OCR Model Backend
# Options: "ollama" | "gemini" | "huggingface"
#   ollama      -> Ollama model tag (local or cloud Ollama)
#   gemini      -> Google Gemini API (cloud)
#   huggingface -> HuggingFace Inference API (cloud, no local GPU needed)
MODEL_VISION_BACKEND = os.getenv("MODEL_VISION_BACKEND", "ollama")

# Vision model name — interpreted differently per backend:
#   ollama      -> Ollama model tag, e.g. "qwen3-vl:235b-cloud" or "llava:13b"
#   gemini      -> Gemini model name, e.g. "gemini-2.5-flash"
#   huggingface -> HuggingFace repo ID, e.g. "Qwen/Qwen3-VL-235B-A22B-Instruct"
MODEL_VISION = os.getenv("MODEL_VISION", "qwen3-vl:235b-cloud")

# HuggingFace cloud Inference API settings
# Get your token from: https://huggingface.co/settings/tokens
HF_TOKEN = os.getenv("HF_TOKEN")                                             # required
MODEL_VISION_HF = os.getenv("MODEL_VISION_HF", "Qwen/Qwen3-VL-235B-A22B-Instruct")

# Chat / Generative Model
# Used for RAG chat and general text generation
# Chat / Generative Model
# Used for RAG chat and general text generation
MODEL_CHAT = os.getenv("MODEL_CHAT", "gemma3:4b")

# Gemini Configuration (Legacy / Option)
MODEL_GEMINI = os.getenv("MODEL_GEMINI", "gemini-2.5-flash")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ------------------------------------------------------------------
# PATH CONFIGURATION
# ------------------------------------------------------------------

# Default paths - can be overridden by env vars
BASE_DATA_DIR = os.getenv("BASE_DATA_DIR", r"D:\CODE-workingBuild\uniAI\source_code\data\year_2")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", r"D:\CODE-workingBuild\uniAI\source_code\chroma")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "multimodal_notes")
