import chromadb
import os
import sys

# Ensure project root is in sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from source_code.config.main import CONFIG

def list_collections():
    path = CONFIG["paths"]["chroma"]
    print(f"Connecting to Chroma at: {path}")
    client = chromadb.PersistentClient(path=path)
    collections = client.list_collections()
    print(f"Found {len(collections)} collections:")
    for c in collections:
        print(f"  - {c.name}")

if __name__ == "__main__":
    list_collections()
