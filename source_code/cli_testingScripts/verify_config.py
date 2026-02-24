import sys
import os
import ollama

# --- Ensure imports work regardless of working directory ---
# Add source_code root to path if not already there
current_dir = os.path.dirname(os.path.abspath(__file__))
source_code_root = os.path.abspath(os.path.join(current_dir, ".."))
if source_code_root not in sys.path:
    sys.path.append(source_code_root)

import config
from pipeline.embeddings.local_embedding import embed

def test_config():
    print("--- Configuration Verification ---")
    print(f"OLLAMA_BASE_URL: {config.OLLAMA_BASE_URL}")
    print(f"MODEL_EMBEDDING: {config.MODEL_EMBEDDING}")
    print(f"MODEL_VISION: {config.MODEL_VISION}")
    print(f"MODEL_CHAT: {config.MODEL_CHAT}")
    print(f"BASE_DATA_DIR: {config.BASE_DATA_DIR}")
    print(f"CHROMA_DB_PATH: {config.CHROMA_DB_PATH}")
    print("--------------------------------")

    try:
        print("\nChecking Ollama Connection...")
        models = ollama.list()
        print("✅ Ollama is reachable.")
        
        print("\nChecking Embedding Model...")
        vector = embed(["test embedding"])
        if vector and len(vector[0]) > 0:
            print(f"✅ Embedding successful. Vector length: {len(vector[0])}")
        else:
            print("❌ Embedding returned empty result.")

    except Exception as e:
        print(f"❌ Verification Failed: {e}")

if __name__ == "__main__":
    test_config()
