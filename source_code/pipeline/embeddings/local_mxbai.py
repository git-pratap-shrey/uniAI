import ollama
import sys
import os

# --- Ensure imports work regardless of working directory ---
# Add source_code root to path if not already there
current_dir = os.path.dirname(os.path.abspath(__file__))
source_code_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if source_code_root not in sys.path:
    sys.path.append(source_code_root)

import config

def embed(texts: list[str]) -> list[list[float]]:
    vectors = []
    for text in texts:
        res = ollama.embeddings(
            model=config.MODEL_EMBEDDING,
            prompt=text
        )
        vectors.append(res["embedding"])
    return vectors
