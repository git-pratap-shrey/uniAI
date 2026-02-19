import os
import json
import re
from pathlib import Path

# ---------------- CONFIG ---------------- #

BASE_PATH = r"D:\CODE-workingBuild\uniAI\source_code\data\year_2"
OUTPUT_FILE = "chunks_ready_for_embedding.jsonl"

MIN_CHUNK_LEN = 120   # characters
MAX_CHUNK_LEN = 1200  # soft cap

# ---------------- HELPERS ---------------- #

def infer_metadata_from_path(txt_path: Path) -> dict:
    parts = txt_path.as_posix().split("/")

    return {
        "year": parts[-6],
        "subject": parts[-5],
        "source": parts[-4],     # notes | pyqs | syllabus
        "unit": parts[-3],
        "topic": txt_path.stem.lower(),
    }


def detect_chunk_type(text: str) -> str:
    t = text.lower()

    if re.search(r"\bdefinition\b|\bis defined as\b|\bmeans\b", t):
        return "definition"
    if re.search(r"\badvantages?\b", t):
        return "advantages"
    if re.search(r"\bdisadvantages?\b", t):
        return "disadvantages"
    if re.search(r"\bsteps?\b|\bprocedure\b", t):
        return "steps"
    if re.search(r"\balgorithm\b", t):
        return "algorithm"
    if re.search(r"\bcompare\b|\bdifferentiate\b|\bvs\b", t):
        return "comparison"
    if re.search(r"\bexample\b", t):
        return "example"
    if re.search(r"\bexplain\b|\bworking\b|\boverview\b", t):
        return "explanation"
    if re.search(r"\bformula\b|=", t):
        return "formula"

    return "general"


def exam_priority(chunk_type: str) -> str:
    if chunk_type in {"definition", "algorithm", "steps", "comparison", "formula"}:
        return "high"
    if chunk_type in {"advantages", "disadvantages", "explanation"}:
        return "medium"
    return "low"


def split_by_structure(text: str) -> list[str]:
    """
    Split text using academic signals instead of token count
    """
    splits = re.split(
        r"\n(?=\d+\.\d+|\d+\s+|definition\b|advantages\b|disadvantages\b|explain\b)",
        text,
        flags=re.IGNORECASE
    )
    return [s.strip() for s in splits if s.strip()]


def merge_weak_chunks(chunks: list[dict]) -> list[dict]:
    merged = []
    buffer = None

    for chunk in chunks:
        if len(chunk["text"]) < MIN_CHUNK_LEN:
            if buffer:
                buffer["text"] += "\n" + chunk["text"]
            else:
                buffer = chunk
        else:
            if buffer:
                merged.append(buffer)
                buffer = None
            merged.append(chunk)

    if buffer:
        merged.append(buffer)

    return merged


# ---------------- MAIN LOGIC ---------------- #

all_chunks = []

for root, _, files in os.walk(BASE_PATH):
    for file in files:
        if not file.endswith(".txt"):
            continue

        txt_path = Path(root) / file
        metadata = infer_metadata_from_path(txt_path)

        with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
            raw_text = f.read().strip()

        if len(raw_text) < 50:
            continue  # skip junk pages

        structured_blocks = split_by_structure(raw_text)

        page_chunks = []

        for block in structured_blocks:
            if len(block) > MAX_CHUNK_LEN:
                block = block[:MAX_CHUNK_LEN]

            ctype = detect_chunk_type(block)

            chunk = {
                "text": block,
                **metadata,
                "chunk_type": ctype,
                "exam_priority": exam_priority(ctype),
                "confidence": "high",
            }

            page_chunks.append(chunk)

        page_chunks = merge_weak_chunks(page_chunks)
        all_chunks.extend(page_chunks)


# ---------------- SAVE OUTPUT ---------------- #

with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
    for chunk in all_chunks:
        out.write(json.dumps(chunk, ensure_ascii=False) + "\n")

print(f"✅ Done. Generated {len(all_chunks)} high-quality chunks.")
print(f"📄 Output file: {OUTPUT_FILE}")
