"""
utils.py
────────
Shared helpers for all source_code scripts.

Centralises:
  • Image encoding   — pil_to_base64, pil_to_bytes
  • JSON parsing     — extract_first_json
  • Embedding        — get_embedding (persistent Ollama client, keep_alive)
  • ChromaDB         — get_chroma_collection
  • VLM client       — build_vlm_client (Ollama with optional cloud auth)
"""

import base64
import io
import json

import chromadb
import ollama
from PIL import Image

import config

# ── Persistent Ollama client for embeddings (keeps model warm in VRAM) ─────────
_embed_client = ollama.Client(host=config.OLLAMA_LOCAL_URL)

# ── Cached ChromaDB collections (keyed by collection name) ───────────────────
_chroma_collections: dict[str, chromadb.Collection] = {}


# ──────────────────────────────────────────────────────────────────────────────
# IMAGE HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def pil_to_base64(img: Image.Image) -> str:
    """Encode a PIL image as a base64 PNG data-URI string (for HuggingFace API)."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


def pil_to_bytes(img: Image.Image) -> bytes:
    """Encode a PIL image as raw PNG bytes (for Ollama API)."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def pil_to_jpeg_bytes(img: Image.Image, quality: int = 85) -> bytes:
    """Encode a PIL image as JPEG bytes (for Ollama cloud — ~5-10x smaller than PNG)."""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


# ──────────────────────────────────────────────────────────────────────────────
# JSON PARSING
# ──────────────────────────────────────────────────────────────────────────────

def extract_first_json(text: str) -> dict | None:
    """
    Extract the first complete JSON object from a (possibly noisy) VLM response.
    Uses brace-counting to find the matching closing brace.
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(text[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        if depth == 0:
            try:
                return json.loads(text[start : i + 1])
            except json.JSONDecodeError:
                return None
    return None


# ──────────────────────────────────────────────────────────────────────────────
# EMBEDDING
# ──────────────────────────────────────────────────────────────────────────────

def get_embedding(text: str) -> list[float]:
    """
    Generate a vector embedding for `text` using the configured local Ollama model.
    Uses a persistent client with keep_alive to avoid cold-start delays.
    """
    response = _embed_client.embeddings(
        model=config.MODEL_EMBEDDING,
        prompt=text,
        keep_alive="10m",
    )
    return response["embedding"]


# ──────────────────────────────────────────────────────────────────────────────
# CHROMADB
# ──────────────────────────────────────────────────────────────────────────────

def get_chroma_collection(collection_name: str = None) -> chromadb.Collection:
    """
    Return (or create) a ChromaDB collection. Results are cached per collection
    name for the lifetime of the process so we only open the DB once per script run.

    Args:
        collection_name: Name of the collection to open. Defaults to
                         config.CHROMA_COLLECTION_NAME when omitted.
    """
    name = collection_name or config.CHROMA_COLLECTION_NAME
    if name not in _chroma_collections:
        client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)
        _chroma_collections[name] = client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )
    return _chroma_collections[name]


# ──────────────────────────────────────────────────────────────────────────────
# VLM CLIENT
# ──────────────────────────────────────────────────────────────────────────────

def build_vlm_client() -> ollama.Client:
    """
    Build an Ollama client for the vision model.
    Uses OLLAMA_BASE_URL (may be cloud) + optional Bearer auth.
    """
    headers = {}
    if config.OLLAMA_API_KEY:
        headers["Authorization"] = f"Bearer {config.OLLAMA_API_KEY}"
    return ollama.Client(host=config.OLLAMA_BASE_URL, headers=headers)
