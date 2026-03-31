import sys
import os
import ollama

# --- Ensure imports work regardless of working directory ---
# Add source_code root to path if not already there
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from source_code.config import CONFIG
from source_code import models
from source_code.pipeline.embeddings.local_embedding import embed

def test_config():
    print("--- Configuration Verification ---")
    print(f"OLLAMA_BASE_URL: {CONFIG.get('OLLAMA_BASE_URL')}")
    print(f"MODEL_EMBEDDING: {CONFIG['providers']['embedding']}")
    print(f"MODEL_CHAT: {CONFIG['model']['model']}")
    print(f"BASE_DATA_DIR: {CONFIG['paths']['base_data']}")
    print(f"CHROMA_DB_PATH: {CONFIG['paths']['chroma']}")
    print("--------------------------------")

    try:
        import ollama
        print("\nChecking Ollama Connection...")
        # Get list of models to verify connectivity
        _ = ollama.list()
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
