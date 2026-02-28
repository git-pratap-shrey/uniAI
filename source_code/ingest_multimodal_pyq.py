import json
from pathlib import Path
import config
from utils import get_embedding, get_chroma_collection

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------

BASE_PATH = config.BASE_DATA_DIR

# ------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------

def build_pyq_embedding_text(q: dict) -> str:
    """
    Builds the search-optimized string to be embedded.
    We embed the actual question text and some key metadata.
    """
    prefix_parts = []
    
    subject = q.get("subject", "")
    unit = q.get("unit")
    year = q.get("year")
    
    if subject:
        prefix_parts.append(f"Subject: {subject}")
    if unit:
        prefix_parts.append(f"Unit: {unit}")
    if year:
        prefix_parts.append(f"Year: {year}")

    prefix = " | ".join(prefix_parts)
    q_text = q.get("question_text", "").strip()
    
    if q_text:
        return f"{prefix}\n\nQuestion:\n{q_text}" if prefix else q_text
    return prefix

# ------------------------------------------------------------------
# MAIN INGESTION
# ------------------------------------------------------------------

def ingest_pyqs():
    print("--- PYQ Ingestion Start ---")
    print(f"Target Collection: {config.CHROMA_PYQ_COLLECTION_NAME}")

    collection = get_chroma_collection(config.CHROMA_PYQ_COLLECTION_NAME)

    root_path = Path(BASE_PATH)
    # the processed jsons are put in pyqs_processed subfolders
    json_files = sorted(root_path.rglob("pyqs_processed/*_processed.json"))

    print(f"Found {len(json_files)} PYQ JSON files to ingest.")

    ingested = 0
    skipped = 0

    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                questions_list = json.load(f)
                
            for q_data in questions_list:
                doc_id = q_data.get("question_id")
                if not doc_id:
                    skipped += 1
                    continue
                
                # Check existance
                existing = collection.get(ids=[doc_id])
                if existing and existing["ids"]:
                    skipped += 1
                    continue
                    
                embedding_text = build_pyq_embedding_text(q_data)
                if not embedding_text.strip() or len(q_data.get("question_text", "").strip()) < 5:
                    skipped += 1
                    continue
                    
                vector = get_embedding(embedding_text[:4000])
                
                # Push into chromadb
                collection.upsert(
                    ids=[doc_id],
                    embeddings=[vector],
                    documents=[embedding_text],
                    metadatas=[{
                        "source": q_data.get("source_pdf", "unknown"),
                        "unit": str(q_data.get("unit", "unknown")),
                        "subject": q_data.get("subject", "unknown"),
                        "document_type": "pyq",
                        "year": q_data.get("year", 2023),
                        "marks": q_data.get("marks") if q_data.get("marks") is not None else 0
                    }]
                )
                ingested += 1
            print(f"   ✅ Processed file: {json_file.name}")
        except Exception as e:
            print(f"   ❌ Failed: {json_file.name}: {e}")

    print(f"\n✅ Ingestion Complete. Ingested: {ingested} questions, Skipped: {skipped} questions")

if __name__ == "__main__":
    ingest_pyqs()