import ollama
import os
import sys

# --- Ensure imports work regardless of working directory ---
current_dir = os.path.dirname(os.path.abspath(__file__))
source_code_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if source_code_root not in sys.path:
    sys.path.append(source_code_root)

import config

# Persistent client with keep_alive — avoids cold-start on every query
_client = ollama.Client(host=config.OLLAMA_LOCAL_URL)


def embed(texts: list[str]) -> list[list[float]]:
    vectors = []
    for text in texts:
        res = _client.embeddings(
            model=config.MODEL_EMBEDDING,
            prompt=text,
            keep_alive="10m",
        )
        vectors.append(res["embedding"])
    return vectors
