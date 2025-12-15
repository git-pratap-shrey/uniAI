import os
import re

import fitz  # PyMuPDF
import chromadb
from pipeline.embeddings.local_mxbai import embed # pyright: ignore[reportMissingImports]

from dotenv import load_dotenv
load_dotenv()
# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------

CHROMA_PATH = os.getenv("CHROMA_PATH")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
BASE_PATH = r"D:\CODE-workingBuild\uniAI\source_code\data\year_2\python"


# ------------------------------------------------------------------
# TEXT CLEANING
# ------------------------------------------------------------------

BULLET_CHARS = {
    "", "•", "●", "▪", "◦", "■", "□",
    "\u2022", "\uf0b7", "\uf0d8", "\uf0a7",
}


def clean_text(text: str) -> str:
    """
    Normalize extracted text:
    - remove common bullet symbols
    - collapse excessive whitespace
    """
    if not text:
        return ""

    for bullet in BULLET_CHARS:
        text = text.replace(bullet, " ")

    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ------------------------------------------------------------------
# PDF TEXT EXTRACTION
# ------------------------------------------------------------------

def extract_pdf_text(pdf_path: str) -> str:
    """
    Prefer sidecar .txt file (from OCR pipeline).
    Fallback to direct PDF extraction if not available.
    """
    txt_path = os.path.splitext(pdf_path)[0] + ".txt"

    if os.path.exists(txt_path):
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                return clean_text(f.read())
        except Exception as e:
            print(f"⚠ Failed to read {txt_path}: {e}")

    # Fallback: extract directly from PDF
    doc = fitz.open(pdf_path)
    pages = []

    for page in doc:
        text = clean_text(page.get_text("text"))
        if text:
            pages.append(text)

    doc.close()
    return "\n\n".join(pages)


# ------------------------------------------------------------------
# LOAD PDFS + METADATA
# ------------------------------------------------------------------

def load_pdfs(folder_path: str) -> list[dict]:
    """
    Walk BASE_PATH and load all PDFs with metadata:
    - category: notes / pyq / syllabus
    - unit: unit1..unit5 (only for notes)
    - filename
    - relative path
    - full extracted text
    """
    docs = []

    for root, _, files in os.walk(folder_path):
        for file in files:
            if not file.lower().endswith(".pdf"):
                continue

            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, folder_path)
            parts = rel_path.split(os.sep)

            category = parts[0] if parts else "unknown"
            unit = "unknown"

            if category.lower() == "notes" and len(parts) > 1:
                if parts[1].lower().startswith("unit"):
                    unit = parts[1].lower()

            filename = os.path.splitext(file)[0]

            print(f"📄 Extracting: {rel_path}")
            try:
                text = extract_pdf_text(full_path)
            except Exception as e:
                print(f"❌ Failed: {rel_path} → {e}")
                continue

            if not text:
                print(f"⚠ No text found: {rel_path}")
                continue

            print(f"   → {len(text)} characters extracted")

            docs.append({
                "text": text,
                "category": category,
                "unit": unit,
                "filename": filename,
                "path": rel_path,
            })

    return docs


# ------------------------------------------------------------------
# CHUNKING
# ------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int = 150) -> list[str]:
    """
    Split text into fixed-size word chunks.
    """
    words = text.split()
    return [
        " ".join(words[i:i + chunk_size])
        for i in range(0, len(words), chunk_size)
        if words[i:i + chunk_size]
    ]


# ------------------------------------------------------------------
# STORE IN CHROMA
# ------------------------------------------------------------------

def store_to_chroma(docs: list[dict]) -> None:
    if not docs:
        print("❌ No documents to store.")
        return

    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Reset collection if it exists
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"🧹 Deleted collection '{COLLECTION_NAME}'")
    except Exception:
        pass

    collection = client.create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    chunks, metadatas, ids = [], [], []

    for doc in docs:
        doc_chunks = chunk_text(doc["text"])
        print(
            f"\n📘 {doc['filename']} "
            f"({doc['category']}, {doc['unit']}) → {len(doc_chunks)} chunks"
        )

        for idx, chunk in enumerate(doc_chunks):
            chunks.append(chunk)
            metadatas.append({
                "category": doc["category"],
                "unit": doc["unit"],
                "source": doc["filename"],
                "path": doc["path"],
                "chunk_index": idx,
            })
            ids.append(f"{doc['filename']}_chunk_{idx}")

    if not chunks:
        print("❌ No chunks generated.")
        return

    print(f"🔍 Embedding {len(chunks)} chunks...")
    embeddings = embed(chunks)

    print("💾 Writing to ChromaDB...")
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids,
    )

    print(f"✅ Stored {len(chunks)} chunks at {CHROMA_PATH}")


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------

if __name__ == "__main__":
    print(f"📂 Loading PDFs from: {BASE_PATH}")
    docs = load_pdfs(BASE_PATH)

    print(f"\n📑 Documents ready: {len(docs)}")
    for d in docs:
        print(f"  → {d['filename']} ({d['category']}, unit={d['unit']})")

    print("\n🔧 Storing embeddings...")
    store_to_chroma(docs)

    print("\n✔ Pipeline complete.")
