import json
import os
import re
import sys
from pathlib import Path

# Ensure project root is on sys.path for source_code imports
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
# Also add source_code/ for direct imports like 'utils'
SOURCE_DIR = os.path.join(ROOT_DIR, "source_code")
if SOURCE_DIR not in sys.path:
    sys.path.append(SOURCE_DIR)

from source_code.config import CONFIG
from utils import get_embedding, get_chroma_collection

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------

BASE_PATH = CONFIG["paths"]["base_data"]

# ------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------

def normalize_unit(unit):
    """Normalize unit to a clean numeric string."""
    if unit is None:
        return "unknown"
    s = str(unit).strip().lower()
    m = re.search(r"\d+", s)
    return str(int(m.group())) if m else "unknown"


# Exact title blocklist (normalised to lower-case, stripped)
_GARBAGE_TITLES = {
    "thank you",
    "rrsimt classes",
    "gateway classes application promotion",
    "aktu full courses (paid)",
    "aktu full courses",
    "subscribe",
    "thank you slide",
}

# Keywords that signal promotional / non-educational content
_PROMO_KEYWORDS = [
    "download", "google play", "play store", "install",
    "subscribe", "youtube", "whatsapp", "telegram",
    "paid course", "paid courses", "link in description",
    "scan qr", "qr code",
]


def is_garbage_chunk(section_meta: dict) -> bool:
    """
    Return True if this section should be skipped because it is
    promotional, non-educational, or near-empty.
    """
    title = section_meta.get("section_title", "").strip().lower()
    full_text = section_meta.get("full_text", "").strip()
    topics = section_meta.get("topics", [])
    concepts = section_meta.get("key_concepts", [])

    # Exact title blocklist
    if title in _GARBAGE_TITLES:
        return True

    # Promotional keywords (>= 2 hits)
    text_lower = full_text.lower()
    hits = sum(1 for kw in _PROMO_KEYWORDS if kw in text_lower)
    if hits >= 2:
        return True

    # Very short text and no structured content
    if len(full_text) < 80 and not topics and not concepts:
        return True

    return False


def build_embedding_text(data: dict) -> str:
    """
    Build rich embedding text for a section chunk.

    New format:
    Subject: COA | Unit: 4 | Type: notes | Topics: t1, t2 | Concepts: c1, c2 | Title: RISC Architecture

    full_text (truncated to 4000 chars)

    Topics moved before title for more embedding weight on keywords.
    Type field added for document_type context.
    """
    meta = data.get("extracted_metadata", {})

    full_text = meta.get("full_text", "").strip()
    section_title = meta.get("section_title", "")
    subject = data.get("subject", "").upper()
    normalized_unit = normalize_unit(data.get("unit"))

    topics = ", ".join(meta.get("topics", []))
    concepts = ", ".join(meta.get("key_concepts", []))

    prefix_parts = []

    if subject:
        prefix_parts.append(f"Subject: {subject}")

    if normalized_unit != "unknown":
        prefix_parts.append(f"Unit: {normalized_unit}")

    prefix_parts.append("Type: notes")

    if topics:
        prefix_parts.append(f"Topics: {topics}")

    if concepts:
        prefix_parts.append(f"Concepts: {concepts}")

    if section_title:
        prefix_parts.append(f"Title: {section_title}")

    prefix = " | ".join(prefix_parts)

    if full_text:
        max_text_len = 4000
        truncated = full_text[:max_text_len]
        return f"{prefix}\n\n{truncated}" if prefix else truncated

    return prefix or ""


# ------------------------------------------------------------------
# MAIN INGESTION
# ------------------------------------------------------------------

def ingest_descriptions():
    print("--- Multimodal Ingestion Start ---")
    print(f"Target Collection: {CONFIG['paths']['collections']['notes']}")

    collection = get_chroma_collection()

    root_path = Path(BASE_PATH)
    # New naming: <pdf_stem>_p{start}-{end}_{chunk_idx}.json
    # Specifically targets files like hand_unit1_p1-1_0.json within notes subfolders
    json_files = sorted(root_path.rglob("notes/**/*_p*-*_*.json"))
    
    print(f"Found {len(json_files)} section JSONs to ingest.")

    ingested = 0
    skipped = 0

    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            meta = data.get("extracted_metadata", {})

            # Skip section-level data: if "sections" key exists this is a
            # per-page file from the new sectioning pipeline — skip it,
            # we only ingest individual section JSONs
            if isinstance(meta, dict) and "sections" in meta:
                skipped += 1
                continue

            # Skip low confidence
            confidence = meta.get("confidence", 1.0)
            if confidence < CONFIG["ingest"]["min_confidence"]:
                skipped += 1
                continue

            # Skip garbage / promotional sections
            if is_garbage_chunk(meta):
                skipped += 1
                continue

            embedding_text = build_embedding_text(data)
            if not embedding_text.strip():
                skipped += 1
                continue

            subject = data.get("subject", "unknown").upper()
            raw_unit = data.get("unit")
            normalized_unit = normalize_unit(raw_unit)

            page_start = data.get("page_start", 0)
            page_end = data.get("page_end", 0)
            chunk_idx = data.get("chunk_idx", 0)
            section_index = data.get("section_index", 0)

            # Information-heavy doc ID
            # Format: {SUBJECT}_unit{unit}_notes_{pdf_stem}_p{start}-{end}_{chunk_idx}
            pdf_stem = json_file.parent.name  # parent folder is the PDF stem
            doc_id = f"{subject}_unit{normalized_unit}_notes_{pdf_stem}_p{page_start}-{page_end}_{chunk_idx}"

            # Skip if already exists
            existing = collection.get(ids=[doc_id])
            if existing and existing["ids"]:
                skipped += 1
                continue

            vector = get_embedding(embedding_text[:4000])

            page_count = max(1, page_end - page_start + 1) if page_end and page_start else 1

            collection.upsert(
                ids=[doc_id],
                embeddings=[vector],
                documents=[embedding_text],
                metadatas=[{
                    "source": f"{pdf_stem}.pdf",
                    "page_start": page_start,
                    "page_end": page_end,
                    "page_count": page_count,
                    "unit": normalized_unit,
                    "subject": subject,
                    "title": meta.get("section_title", "unknown"),
                    "document_type": "notes",
                    "chunk_idx": chunk_idx,
                    "section_index": section_index,
                    "confidence": confidence,
                }]
            )

            if normalized_unit == "unknown":
                print(f"⚠ Unknown unit for {doc_id}")

            ingested += 1
            print(f"   ✅ {doc_id} — {meta.get('section_title', 'untitled')}")

        except Exception as e:
            print(f"   ❌ Failed: {json_file.name}: {e}")

    print(f"\n✅ Ingestion Complete. Ingested: {ingested}, Skipped: {skipped}")


if __name__ == "__main__":
    ingest_descriptions()
