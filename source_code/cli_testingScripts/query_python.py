import sys
import os
import re

# --- FIX IMPORT PATH ---
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import chromadb
from pipeline.embeddings.local_mxbai import embed

# -------- CONFIG ----------
CHROMA_PATH = r"D:\CODE-workingBuild\uniAI\source_code\chroma\python"
COLLECTION_NAME = "python"


# -------- PRETTY PRINTERS --------

def print_result(idx, doc, meta):
    print(f"\nResult #{idx+1}")
    print("-" * 80)
    print(doc[:500] + ("..." if len(doc) > 500 else ""))
    print("\nSource:", meta)
    print("-" * 80)


# -------- METADATA SEARCH DETECTOR --------

def detect_unit_query(query: str):
    """
    Extract unit number from queries like:
        unit4
        unit 4
        show unit4
        give me unit 4 notes
    """
    q = query.lower().strip()
    m = re.search(r"unit\s*([1-9])", q)
    if m:
        return f"unit{m.group(1)}"
    return None


# -------- MAIN QUERY FUNCTION --------

def chroma_query(user_query: str):
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(COLLECTION_NAME)

    # 1️⃣ Check if user is asking for a specific UNIT
    unit = detect_unit_query(user_query)

    if unit:
        print(f"\n🎯 Detected UNIT query → Filtering by metadata: unit={unit}\n")

        # Dummy 1024-D vector so Chroma doesn't try to embed text
        dummy_vec = [[0.0] * 1024]

        results = collection.query(
            query_embeddings=dummy_vec,
            n_results=20,
            where={"unit": unit}
        )

    else:
        # 2️⃣ Normal semantic search - USE YOUR CUSTOM EMBEDDING
        print("\n🔍 Semantic search...\n")
        
        # ⚠️ FIX: Manually embed using YOUR mxbai model
        query_embedding = embed([user_query])  # Returns list of embeddings
        
        results = collection.query(
            query_embeddings=query_embedding,  # Changed from query_texts
            n_results=5
        )

    # Handle empty results
    if not results or not results["documents"] or len(results["documents"][0]) == 0:
        print("❌ No results found.")
        return

    # 3️⃣ Pretty print results
    docs = results["documents"][0]
    metas = results["metadatas"][0]

    print("📌 RESULTS:")
    for idx, (d, m) in enumerate(zip(docs, metas)):
        print_result(idx, d, m)

# -------- CLI LOOP --------

if __name__ == "__main__":
    print("\n✨ Python Notes Query Engine (Chroma + MXBAI)\n")

    while True:
        query = input("\n🔍 Ask something: ").strip()
        if query.lower() in ["exit", "quit", "q"]:
            print("\n👋 Goodbye!")
            break

        chroma_query(query)
