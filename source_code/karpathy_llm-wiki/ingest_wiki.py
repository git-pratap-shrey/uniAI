"""
ingest_wiki.py
──────────────
Ingests wiki/*.md pages (produced by compile_wiki.py) into a separate
ChromaDB collection so you can compare wiki vs raw-chunk retrieval quality.

Collection name: <CHROMA_COLLECTION_NAME>_wiki  (e.g. multimodal_notes_wiki)

Run:
  python source_code/ingest_wiki.py --subject PYTHON
"""

import os
import sys
import argparse
from pathlib import Path

import chromadb
import ollama

current_dir = os.path.dirname(os.path.abspath(__file__))
# Ensure project root is in sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from source_code.config.main import CONFIG

EMBED_MODEL     = CONFIG["providers"]["embedding_model"]
CHROMA_PATH     = CONFIG["paths"]["chroma"]
COLLECTION_NAME = CONFIG["paths"]["collections"]["notes"] + "_wiki"  # separate collection

_ollama_client = ollama.Client(host=CONFIG["OLLAMA_LOCAL_URL"])


def get_embedding(text: str) -> list[float]:
    r = _ollama_client.embeddings(
        model=EMBED_MODEL,
        prompt=text,
        keep_alive="10m"
    )
    return r["embedding"]


def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )


def ingest_wiki(subject: str):
    print(f"\n📥 Wiki Ingestion — {subject}")
    print(f"   Collection: {COLLECTION_NAME}\n")

    wiki_root = Path(CONFIG["paths"]["base_data"]) / subject / "wiki"
    if not wiki_root.exists():
        print(f"❌ Wiki folder not found: {wiki_root}")
        print("   Run compile_wiki.py first.")
        return

    pages = sorted(wiki_root.rglob("*.md"))
    pages = [p for p in pages if p.name != "index.md"]
    print(f"Found {len(pages)} wiki pages.")

    collection = get_collection()
    ingested = skipped = 0

    for page_path in pages:
        content = page_path.read_text(encoding="utf-8").strip()
        if not content:
            continue

        # Extract unit from folder structure: wiki/<unit>/<filename>.md
        rel = page_path.relative_to(wiki_root)
        unit = rel.parts[0] if len(rel.parts) > 1 else "unknown"
        topic = page_path.stem.replace("_", " ").title()

        doc_id = f"wiki_{subject}_{unit}_{page_path.stem}"

        existing = collection.get(ids=[doc_id])
        if existing and existing["ids"]:
            skipped += 1
            continue

        # Use first 4000 chars for embedding (wiki pages are pre-structured, this is enough)
        vector = get_embedding(content[:4000])

        collection.upsert(
            ids=[doc_id],
            embeddings=[vector],
            documents=[content],
            metadatas=[{
                "subject":  subject,
                "unit":     unit,
                "topic":    topic,
                "source":   str(page_path.relative_to(Path(CONFIG["paths"]["base_data"]))),
                "type":     "wiki",
            }]
        )
        ingested += 1
        print(f"  ✅ {unit}/{page_path.name}")

    print(f"\n✅ Done. Ingested: {ingested}, Skipped: {skipped}")
    print(f"   Collection '{COLLECTION_NAME}' is ready for testing.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True)
    args = parser.parse_args()
    ingest_wiki(args.subject.upper())