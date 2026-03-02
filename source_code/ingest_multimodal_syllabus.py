"""
ingest_multimodal_syllabus.py
──────────────────────────────
Ingests the syllabus chunk JSONs produced by extract_multimodal_syllabus.py
into ChromaDB. Keeps the notes ingestion pipeline (ingest_multimodal.py)
completely untouched.

Expected input files (written by extract_multimodal_syllabus.py):
  <subject>/syllabus/syllabus_unit_1.json
  <subject>/syllabus/syllabus_unit_2.json
  <subject>/syllabus/syllabus_unit_3.json
  <subject>/syllabus/syllabus_unit_4.json
  <subject>/syllabus/syllabus_unit_5.json
  <subject>/syllabus/syllabus_co.json
  <subject>/syllabus/syllabus_books.json

Usage:
  python source_code/ingest_multimodal_syllabus.py
"""

import os
import sys
import json
from pathlib import Path

# ── ensure source_code/ is on the path ────────────────────────────────────────
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

import config
from utils import get_embedding, get_chroma_collection

# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def build_syllabus_embedding_text(data: dict) -> str:
    """
    Build a rich embedding string for a syllabus chunk.
    Syllabus chunks have a flat schema — no 'extracted_metadata' wrapper.
    """
    subject          = data.get("subject", "")
    subject_name     = data.get("subject_name", "")
    syllabus_version = data.get("syllabus_version", "")
    chunk_type       = data.get("chunk_type", "")
    unit             = data.get("unit", "")
    unit_title       = data.get("unit_title", "")
    topics           = ", ".join(data.get("topics", []))
    full_text        = data.get("full_text", "").strip()

    prefix_parts = []
    if subject_name or subject:
        prefix_parts.append(f"Subject: {subject_name or subject}")
    if syllabus_version:
        prefix_parts.append(f"Syllabus: {syllabus_version}")
    if unit:
        prefix_parts.append(f"Unit: {unit}")
    if unit_title:
        prefix_parts.append(f"Title: {unit_title}")
    if chunk_type:
        prefix_parts.append(f"Section: {chunk_type.replace('_', ' ').title()}")
    if topics:
        prefix_parts.append(f"Topics: {topics}")

    prefix = " | ".join(prefix_parts)
    if full_text:
        return f"{prefix}\n\n{full_text[:4000]}" if prefix else full_text[:4000]
    return prefix


# ──────────────────────────────────────────────────────────────────────────────
# MAIN INGESTION
# ──────────────────────────────────────────────────────────────────────────────

def ingest_syllabuses():
    print("--- Syllabus Ingestion Start ---")
    print(f"Target Collection : {config.CHROMA_SYLLABUS_COLLECTION_NAME}")
    print(f"Scanning           : {config.BASE_DATA_DIR}")

    collection = get_chroma_collection(config.CHROMA_SYLLABUS_COLLECTION_NAME)
    root_path  = Path(config.BASE_DATA_DIR)

    json_files = sorted(root_path.rglob("syllabus_*.json"))
    print(f"Found {len(json_files)} syllabus chunk JSON(s).\n")

    ingested = skipped = errors = 0

    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Safety check — only process files tagged as syllabus
            if data.get("type") != "syllabus":
                print(f"   -> {json_file.name}: not a syllabus chunk, skipping.")
                skipped += 1
                continue

            embedding_text = build_syllabus_embedding_text(data)
            if not embedding_text.strip():
                print(f"   -> {json_file.name}: empty content, skipping.")
                skipped += 1
                continue

            # Stable, collision-free ID
            source_pdf = data.get("source_pdf", "unknown")
            chunk_type = data.get("chunk_type", "unknown")
            subject    = data.get("subject", "unknown")
            doc_id     = f"syllabus_{subject}_{source_pdf}_{chunk_type}"

            existing = collection.get(ids=[doc_id])
            if existing and existing["ids"]:
                print(f"   -> {doc_id}: already ingested, skipping.")
                skipped += 1
                continue

            vector = get_embedding(embedding_text[:4000])
            collection.upsert(
                ids=[doc_id],
                embeddings=[vector],
                documents=[embedding_text],
                metadatas=[{
                    # Fields shared with notes for cross-query compatibility
                    "source":           source_pdf,
                    "page_start":       0,
                    "page_end":         0,
                    "unit":             str(data.get("unit") or ""),
                    "subject":          data.get("subject", "unknown"),
                    "title":            data.get("unit_title", chunk_type),
                    "document_type":    "syllabus",
                    # Syllabus-specific fields
                    "syllabus_version": data.get("syllabus_version", "unknown"),
                    "chunk_type":       chunk_type,
                    "confidence":       1.0,
                }],
            )
            ingested += 1
            print(f"   ✅ {doc_id}")

        except Exception as exc:
            print(f"   ❌ Failed: {json_file.name}: {exc}")
            errors += 1

    print(f"\n✅ Syllabus Ingestion Complete.")
    print(f"   Ingested : {ingested}")
    print(f"   Skipped  : {skipped}")
    if errors:
        print(f"   Errors   : {errors}")


if __name__ == "__main__":
    ingest_syllabuses()