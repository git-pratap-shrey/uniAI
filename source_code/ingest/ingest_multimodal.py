import json
import os
import sys
from pathlib import Path

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import config
from utils import get_embedding, get_chroma_collection

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------

BASE_PATH = config.BASE_DATA_DIR

# ------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------

def build_embedding_text(data: dict) -> str:
    """
    Build a rich embedding string from extracted metadata.
    Prioritizes full_text for semantic search quality,
    with structured metadata as supplementary context.
    """
    meta = data.get("extracted_metadata", {})

    # Primary: full OCR text (the most important for search)
    full_text = meta.get("full_text", "").strip()

    # Secondary: structured metadata for enrichment
    title = meta.get("title", "")
    unit = data.get("unit", meta.get("unit", ""))
    topics = ", ".join(meta.get("topics", []))
    concepts = ", ".join(meta.get("key_concepts", []))
    subject = data.get("subject", "")

    prefix_parts = []
    if subject:
        prefix_parts.append(f"Subject: {subject}")
    if unit:
        prefix_parts.append(f"Unit: {unit}")
    if title:
        prefix_parts.append(f"Title: {title}")
    if topics:
        prefix_parts.append(f"Topics: {topics}")
    if concepts:
        prefix_parts.append(f"Key Concepts: {concepts}")

    prefix = " | ".join(prefix_parts)

    if full_text:
        # Combine metadata prefix with actual content
        max_text_len = 4000
        truncated = full_text[:max_text_len]
        return f"{prefix}\n\n{truncated}" if prefix else truncated
    elif prefix:
        return prefix
    else:
        return data.get("description", "")


# ------------------------------------------------------------------
# MAIN INGESTION
# ------------------------------------------------------------------

def ingest_descriptions():
    print("--- Multimodal Ingestion Start ---")
    print(f"Target Collection: {config.CHROMA_COLLECTION_NAME}")

    collection = get_chroma_collection()

    root_path = Path(BASE_PATH)
    json_files = sorted(root_path.rglob("chunk_*.json"))

    print(f"Found {len(json_files)} chunk JSONs to ingest.")

    ingested = 0
    skipped = 0

    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            if data.get("extracted_metadata", {}).get("document_type") == "question_paper":
                print(f"   -> Skipping {json_file.name} (question_paper, belongs in PYQ pipeline)")
                skipped += 1
                continue

            confidence = data.get("extracted_metadata", {}).get("confidence", 1.0)
            if confidence < config.MIN_INGEST_CONFIDENCE:
                print(f"   -> Skipping {json_file.name} (confidence {confidence:.2f} below threshold {config.MIN_INGEST_CONFIDENCE})")
                skipped += 1
                continue

            embedding_text = build_embedding_text(data)
            if not embedding_text.strip():
                print(f"   -> Skipping {json_file.name} (empty content)")
                skipped += 1
                continue

            file_name = data.get("source_pdf", "unknown")
            page_start = data.get("page_start", 0)
            page_end = data.get("page_end", 0)
            subject = data.get("subject", "unknown")
            doc_id = f"{subject}_{file_name}_p{page_start}-{page_end}"

            existing = collection.get(ids=[doc_id])
            if existing and existing["ids"]:
                skipped += 1
                continue

            # qwen3-embedding:4B supports 32K tokens — 4000 chars is well within limit
            vector = get_embedding(embedding_text[:4000])

            meta = data.get("extracted_metadata", {})
            collection.upsert(
                ids=[doc_id],
                embeddings=[vector],
                documents=[embedding_text],
                metadatas=[{
                    "source": file_name,
                    "page_start": page_start,
                    "page_end": page_end,
                    "unit": str(meta.get("unit", data.get("unit", "unknown"))),
                    "subject": data.get("subject", "unknown"),
                    "title": meta.get("title", "unknown"),
                    "document_type": meta.get("document_type", "unknown"),
                    "confidence": meta.get("confidence", 0),
                }]
            )
            ingested += 1
            print(f"   ✅ {doc_id} — {meta.get('title', 'untitled')}")

        except Exception as e:
            print(f"   ❌ Failed: {json_file.name}: {e}")

    print(f"\n✅ Ingestion Complete. Ingested: {ingested}, Skipped: {skipped}")


if __name__ == "__main__":
    ingest_descriptions()