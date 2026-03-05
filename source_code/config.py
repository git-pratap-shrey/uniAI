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

# Control whether to use Ollama Cloud API or Local
USE_OLLAMA_CLOUD = os.getenv("USE_OLLAMA_CLOUD", "True").lower() in ("true", "1", "t")

# Chat / Generative Model
# Used for RAG chat and general text generation
MODEL_CHAT = os.getenv("MODEL_CHAT", "gemini-3-flash-preview:latest")

# Router / Classification Model
# Fast local model used specifically for extracting keywords and context switching
MODEL_ROUTER = os.getenv("MODEL_ROUTER", "qwen3.5:4b")

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
# CROSS-ENCODER CONFIGURATION
# ------------------------------------------------------------------

# HuggingFace model ID for cross-encoder reranker
CROSS_ENCODER_MODEL = os.getenv("CROSS_ENCODER_MODEL", "tomaarsen/Qwen3-Reranker-0.6B-seq-cls")
# CROSS_ENCODER_MODEL = os.getenv("CROSS_ENCODER_MODEL", "Qwen/Qwen3-Reranker-0.6B")

# Gate threshold: if the top cross-encoder score is below this, fall back to generic
MIN_CROSS_SCORE = float(os.getenv("MIN_CROSS_SCORE", "0.65"))

# Max (query, chunk) pairs to send to the cross-encoder (controls latency)
CROSS_ENCODER_CANDIDATES = int(os.getenv("CROSS_ENCODER_CANDIDATES", "6"))

# Top N chunks to keep after cross-encoder reranking
PIPELINE_CROSS_RERANK_TOP_N = int(os.getenv("PIPELINE_CROSS_RERANK_TOP_N", "4"))

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

# ------------------------------------------------------------------
# RAG PIPELINE & RETRIEVAL TWEAKS
# ------------------------------------------------------------------

# rag/rag_pipeline.py
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "4"))
OLLAMA_CHAT_NUM_CTX = int(os.getenv("OLLAMA_CHAT_NUM_CTX", "8192"))
OLLAMA_CHAT_TEMPERATURE = float(os.getenv("OLLAMA_CHAT_TEMPERATURE", "0.25"))

# Retrieval counts in pipeline
PIPELINE_NOTES_K = int(os.getenv("PIPELINE_NOTES_K", "8"))
PIPELINE_SYLLABUS_K = int(os.getenv("PIPELINE_SYLLABUS_K", "3"))
PIPELINE_RERANK_TOP_N = int(os.getenv("PIPELINE_RERANK_TOP_N", "5"))

# rag/reranker.py
RERANK_DEFAULT_TOP_N = int(os.getenv("RERANK_DEFAULT_TOP_N", "5"))
RERANK_UNIT_MATCH_BOOST = float(os.getenv("RERANK_UNIT_MATCH_BOOST", "1.15"))
RERANK_SYLLABUS_PENALTY = float(os.getenv("RERANK_SYLLABUS_PENALTY", "0.90"))

# rag/router.py
OLLAMA_ROUTER_TEMPERATURE = float(os.getenv("OLLAMA_ROUTER_TEMPERATURE", "0.0"))
OLLAMA_ROUTER_NUM_PREDICT = int(os.getenv("OLLAMA_ROUTER_NUM_PREDICT", "10"))

# Hybrid router thresholds
KEYWORD_MIN_SCORE = int(os.getenv("KEYWORD_MIN_SCORE", "2"))
EMBEDDING_ROUTER_THRESHOLD = float(os.getenv("EMBEDDING_ROUTER_THRESHOLD", "0.55"))
UNIT_EMBEDDINGS_PATH = os.getenv(
    "UNIT_EMBEDDINGS_PATH",
    str(_SOURCE_CODE_DIR / "pipeline" / "embeddings" / "unit_embeddings.pkl"),
)

# rag/search.py
SEARCH_NOTES_K_DEFAULT = int(os.getenv("SEARCH_NOTES_K_DEFAULT", "8"))
SEARCH_SYLLABUS_K_DEFAULT = int(os.getenv("SEARCH_SYLLABUS_K_DEFAULT", "5"))
SEARCH_PYQ_K_DEFAULT = int(os.getenv("SEARCH_PYQ_K_DEFAULT", "5"))
SEARCH_PYQ_THRESHOLD = float(os.getenv("SEARCH_PYQ_THRESHOLD", "0.60"))
SEARCH_ALL_NOTES_K = int(os.getenv("SEARCH_ALL_NOTES_K", "6"))
SEARCH_ALL_SYLLABUS_K = int(os.getenv("SEARCH_ALL_SYLLABUS_K", "3"))