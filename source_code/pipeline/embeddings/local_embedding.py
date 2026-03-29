import ollama
import os
import sys

# --- Ensure imports work regardless of working directory ---
current_dir = os.path.dirname(os.path.abspath(__file__))
source_code_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if source_code_root not in sys.path:
    sys.path.append(source_code_root)

from source_code.config import CONFIG
from source_code import models

def embed(texts: list[str]) -> list[list[float]]:
    """
    Generate vector embeddings for a list of strings using the models registry.
    """
    return models.embed(texts, provider=CONFIG["providers"].get("embedding", "ollama"))
