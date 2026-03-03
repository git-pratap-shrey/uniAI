import json
import os
import sys
import re
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

def normalize_unit(unit):
    """
    Normalize unit to a clean numeric string.

    Accepts:
        1
        "1"
        "unit1"
        "Unit 1"
        "UNIT-1"
        "unit 03"

    Returns:
        "1", "2", etc.
        or "unknown" if invalid
    """

    if unit is None:
        return "unknown"

    unit_str = str(unit).strip().lower()

    if not unit_str:
        return "unknown"

    match = re.search(r"\d+", unit_str)
    if match:
        return str(int(match.group()))  # remove leading zeros

    return "unknown"


def build_embedding_text(data: dict) -> str:
    """
    Build rich embedding text.
    """

    meta = data.get("extracted_metadata", {})

    full_text = meta.get("full_text", "").strip()
    title = meta.get("title", "")
    subject = data.get("subject", "")

    raw_unit = data.get("unit")
    normalized_unit = normalize_unit(raw_unit)

    topics = ", ".join(meta.get("topics", []))
    concepts = ", ".join(meta.get("key_concepts", []))

    prefix_parts = []

    if subject:
        prefix_parts.append(f"Subject: {subject}")

    if normalized_unit != "unknown":
        prefix_parts.append(f"Unit: {normalized_unit}")

    if title:
        prefix_parts.append(f"Title: {title}")

    if topics:
        prefix_parts.append(f"Topics: {topics}")

    if concepts:
        prefix_parts.append(f"Key Concepts: {concepts}")

    prefix = " | ".join(prefix_parts)

    if full_text:
        max_text_len = 4000
        truncated = full_text[:max_text_len]
        return f"{prefix}\n\n{truncated}" if prefix else truncated

    return prefix or data.get("description", "")


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

            meta = data.get("extracted_metadata", {})

            # Skip question papers (handled elsewhere)
            if meta.get("document_type") == "question_paper":
                skipped += 1
                continue

            # Skip low confidence
            confidence = meta.get("confidence", 1.0)
            if confidence < config.MIN_INGEST_CONFIDENCE:
                skipped += 1
                continue

            embedding_text = build_embedding_text(data)
            if not embedding_text.strip():
                skipped += 1
                continue

            file_name = data.get("source_pdf", "unknown")
            page_start = data.get("page_start", 0)
            page_end = data.get("page_end", 0)
            subject = data.get("subject", "unknown")

            doc_id = f"{subject}_{file_name}_p{page_start}-{page_end}"

            # Skip if already exists
            existing = collection.get(ids=[doc_id])
            if existing and existing["ids"]:
                skipped += 1
                continue

            vector = get_embedding(embedding_text[:4000])

            raw_unit = meta.get("unit") or data.get("unit")
            normalized_unit = normalize_unit(raw_unit)

            collection.upsert(
                ids=[doc_id],
                embeddings=[vector],
                documents=[embedding_text],
                metadatas=[{
                    "source": file_name,
                    "page_start": page_start,
                    "page_end": page_end,
                    "unit": normalized_unit,
                    "subject": subject,
                    "title": meta.get("title", "unknown"),
                    "document_type": meta.get("document_type", "unknown"),
                    "confidence": confidence,
                }]
            )

            if normalized_unit == "unknown":
                print(f"⚠ Unknown unit for {doc_id}")

            ingested += 1
            print(f"   ✅ {doc_id} — {meta.get('title', 'untitled')}")

        except Exception as e:
            print(f"   ❌ Failed: {json_file.name}: {e}")

    print(f"\n✅ Ingestion Complete. Ingested: {ingested}, Skipped: {skipped}")


if __name__ == "__main__":
    ingest_descriptions()