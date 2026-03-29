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
from source_code import models
from PIL import Image

from source_code.config import CONFIG

# Persistent clients are now managed by models.py

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
    Generate a vector embedding for `text` using the centralized models registry.
    """
    return models.embed([text])[0]


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
    name = collection_name or CONFIG["paths"]["collections"]["notes"]
    if name not in _chroma_collections:
        client = chromadb.PersistentClient(path=CONFIG["paths"]["chroma"])
        _chroma_collections[name] = client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )
    return _chroma_collections[name]


# ──────────────────────────────────────────────────────────────────────────────
# VLM CLIENT
# ──────────────────────────────────────────────────────────────────────────────

# build_vlm_client is deprecated. Use models.vision() instead.
