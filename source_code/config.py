import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from a .env file (for secrets and paths)
load_dotenv()

# ------------------------------------------------------------------
# OLLAMA CONFIGURATION
# ------------------------------------------------------------------

# Ollama instance URLs and keys
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_LOCAL_URL = os.getenv("OLLAMA_LOCAL_URL", "http://localhost:11434")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "")

# ------------------------------------------------------------------
# MODEL CONFIGURATION (Architectural Constants)
# ------------------------------------------------------------------

# Embedding Model
# qwen3-embedding:4B has a 32K token context window
MODEL_EMBEDDING = os.getenv("MODEL_EMBEDDING", "qwen3-embedding:4B")

# Vision / OCR Model Backend (ollama | huggingface)
MODEL_VISION_BACKEND = os.getenv("MODEL_VISION_BACKEND", "ollama")

# Vision model name
MODEL_VISION = os.getenv("MODEL_VISION", "qwen3-vl:235b-cloud")

# HuggingFace cloud Inference API settings
HF_TOKEN = os.getenv("HF_TOKEN", "") 
MODEL_VISION_HF = os.getenv("MODEL_VISION_HF", "Qwen/Qwen3-VL-235B-A22B-Instruct")

# Control whether to use Ollama Cloud API or Local
USE_OLLAMA_CLOUD = os.getenv("USE_OLLAMA_CLOUD", "True").lower() == "true"

# Chat / Generative Model
MODEL_CHAT = os.getenv("MODEL_CHAT", "gemini-3-flash-preview:latest")

# Optional API key for Gemini models
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Router / Classification Model
# If MODEL_ROUTER is not in env, fallback to qwen3.5:4b
MODEL_ROUTER = os.getenv("MODEL_ROUTER", "qwen3.5:4B")

# ------------------------------------------------------------------
# RETRIEVAL CONFIGURATION
# ------------------------------------------------------------------

# Minimum cosine similarity to keep a search result
SIMILARITY_THRESHOLD = 0.35

# Minimum similarity the top-ranked chunk must have
MIN_STRONG_SIM = 0.6

# ------------------------------------------------------------------
# CROSS-ENCODER CONFIGURATION
# ------------------------------------------------------------------

# HuggingFace model ID for cross-encoder reranker
CROSS_ENCODER_MODEL = "tomaarsen/Qwen3-Reranker-0.6B-seq-cls"

# Gate threshold: if the top cross-encoder score is below this, fall back to generic
MIN_CROSS_SCORE = 0.65

# Max (query, chunk) pairs to send to the cross-encoder
CROSS_ENCODER_CANDIDATES = 6

# Top N chunks to keep after cross-encoder reranking
PIPELINE_CROSS_RERANK_TOP_N = 4

# ------------------------------------------------------------------
# PATH CONFIGURATION (Environment Dependent)
# ------------------------------------------------------------------

# Resolve paths relative to this file so defaults work on any machine.
_SOURCE_CODE_DIR = Path(__file__).parent

BASE_DATA_DIR = os.getenv("BASE_DATA_DIR", str(_SOURCE_CODE_DIR / "data" / "year_2"))

# Query Expander & Router Maps
ALIASES_FILE_PATH = os.getenv("ALIASES_FILE_PATH", str(_SOURCE_CODE_DIR / "data" / "subject_aliases.json"))
KEYWORDS_FILE_PATH = os.getenv("KEYWORDS_FILE_PATH", str(_SOURCE_CODE_DIR / "data" / "subject_keywords.json"))
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", str(_SOURCE_CODE_DIR / "chroma"))

# Collection Names
CHROMA_COLLECTION_NAME = "multimodal_notes"
CHROMA_SYLLABUS_COLLECTION_NAME = "multimodal_syllabus"
CHROMA_PYQ_COLLECTION_NAME = "multimodal_pyq"

# Minimum OCR confidence to ingest a chunk
MIN_INGEST_CONFIDENCE = 0.3

# ------------------------------------------------------------------
# RAG PIPELINE & RETRIEVAL TWEAKS
# ------------------------------------------------------------------

# rag/rag_pipeline.py
MAX_HISTORY_TURNS = 4
OLLAMA_CHAT_NUM_CTX = 8192
OLLAMA_CHAT_TEMPERATURE = 0.25

# Retrieval counts in pipeline
PIPELINE_NOTES_K = 8
PIPELINE_SYLLABUS_K = 7
PIPELINE_RERANK_TOP_N = 7

# rag/reranker.py
RERANK_DEFAULT_TOP_N = 7
RERANK_UNIT_MATCH_BOOST = 1.15
RERANK_SYLLABUS_PENALTY = 0.90

# rag/router.py
OLLAMA_ROUTER_TEMPERATURE = 0.0
OLLAMA_ROUTER_NUM_PREDICT = 10

# Hybrid router thresholds
KEYWORD_MIN_SCORE = 2
EMBEDDING_ROUTER_THRESHOLD = 0.55
UNIT_EMBEDDINGS_PATH = os.getenv(
    "UNIT_EMBEDDINGS_PATH",
    str(_SOURCE_CODE_DIR / "pipeline" / "embeddings" / "unit_embeddings.pkl"),
)

# rag/search.py
SEARCH_NOTES_K_DEFAULT = 8
SEARCH_SYLLABUS_K_DEFAULT = 7
SEARCH_PYQ_K_DEFAULT = 5
SEARCH_PYQ_THRESHOLD = 0.60
SEARCH_ALL_NOTES_K = 6
SEARCH_ALL_SYLLABUS_K = 7

# rag/query_expander.py
QUERY_EXPANDER_MAX_KEYWORDS = 6