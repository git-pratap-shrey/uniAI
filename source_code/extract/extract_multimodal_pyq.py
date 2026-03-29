import os
import re
import json
import fitz
import time
from pathlib import Path

import sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from source_code.config import CONFIG
from source_code import models
from utils import (
    pil_to_base64,
    pil_to_jpeg_bytes,
    extract_first_json,
)
from prompts import pyq_unit_classification

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------

BASE_PATH = CONFIG["paths"]["base_data"]
BACKEND = CONFIG["providers"]["vision"].lower()

# Vision model configuration now handled by models.vision()
if BACKEND == "huggingface":
    from huggingface_hub import InferenceClient as _HFClient
    # HF_MODEL_ID will be set by CONFIG["providers"]["vision_model"]

def get_syllabus_topics(subject: str) -> str:
    """Finds the syllabus JSON for the subject and extracts units."""
    syllabus_dir = Path(BASE_PATH) / subject / "syllabus"
    if not syllabus_dir.exists():
        return "Unit 1: Basic Concepts\nUnit 2: Intermediate\nUnit 3: Advanced\nUnit 4: Applications\nUnit 5: Case Studies"

    for json_file in syllabus_dir.rglob("chunk_*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            topics = data.get("extracted_metadata", {}).get("topics", [])
            if topics:
                return "\n".join([f"Unit {i+1}: {topic}" for i, topic in enumerate(topics)])
        except:
            pass
    return "Unit 1: Basics\nUnit 2: Core\nUnit 3: Advanced\nUnit 4: Tools\nUnit 5: Applications"


def load_pdf(pdf_path: Path):
    from prompts import PYQ_VLM_TRANSCRIPTION
    doc = fitz.open(str(pdf_path))
    text = ""
    # Ollama cloud: 1.0 avoids Cloudflare 524 timeouts; HuggingFace: 1.5 for better OCR
    _scale = 1.0 if BACKEND == "ollama" else 1.5
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(matrix=fitz.Matrix(_scale, _scale))

        MAX_RETRIES = 3
        raw_response = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                # Use centralized models.vision() for all providers
                if BACKEND == "huggingface":
                    from PIL import Image
                    import io
                    img = Image.open(io.BytesIO(pix.tobytes("png")))
                    print(f"   -> Call HF Vision API (Page {page_num+1})...", end="", flush=True)
                    raw_response = models.vision(
                        images=[img],
                        prompt=PYQ_VLM_TRANSCRIPTION,
                        provider=CONFIG["providers"]["vision"],
                        model=CONFIG["providers"]["vision_model"]
                    )
                    print(" done.")
                else:
                    # Convert to PIL then JPEG (5-10x smaller than raw PNG bytes)
                    from PIL import Image as _PIL
                    import io as _io
                    img = _PIL.open(_io.BytesIO(pix.tobytes("png")))
                    print(f"   -> Call Ollama Vision API (Page {page_num+1})...", end="", flush=True)
                    raw_response = models.vision(
                        images=[pil_to_jpeg_bytes(img)],
                        prompt=PYQ_VLM_TRANSCRIPTION,
                        provider=CONFIG["providers"]["vision"],
                        model=CONFIG["providers"]["vision_model"]
                    )
                    print(" done.")
                break
            except Exception as e:
                err_str = str(e)
                if attempt < MAX_RETRIES:
                    wait = 15 * attempt
                    print(f" ⚠ Attempt {attempt} failed for page {page_num+1}: {err_str[:120]}")
                    print(f"   Retrying in {wait}s...", end="", flush=True)
                    time.sleep(wait)
                else:
                    print(f" ❌ Failed after {MAX_RETRIES} attempts for page {page_num+1}: {err_str[:120]}")

        if raw_response is not None:
            text += raw_response + "\n"
    doc.close()
    return text


def normalize_text(text: str) -> str:
    lines = [line.strip() for line in text.split('\n') if line.strip()]

    merged_lines = []
    q_pattern = re.compile(r'^(Q\d+\.|\d+\.|\(\w\)|[a-z]\)|[a-zA-Z]\.)', re.IGNORECASE)
    sec_pattern = re.compile(r'^(SECTION\s+[A-Z]|Part\s+[A-Z])', re.IGNORECASE)

    for line in lines:
        if not merged_lines:
            merged_lines.append(line)
            continue

        if q_pattern.match(line) or sec_pattern.match(line) or line.isupper() or line.startswith("Attempt"):
            if merged_lines[-1].endswith("-"):
                merged_lines[-1] = merged_lines[-1][:-1] + line
            else:
                merged_lines.append(line)
        else:
            if merged_lines[-1].endswith("-"):
                merged_lines[-1] = merged_lines[-1][:-1] + line
            else:
                merged_lines[-1] += " " + line

    return "\n".join(merged_lines)


def clean_question_text(q_text: str) -> tuple:
    """
    Strip trailing marks, CO numbers, pipe separators, and tab-separated
    numbers from question text. Returns (cleaned_text, marks_or_None).

    Handles all observed formats:
      "Why is security hard?                    10"   (bare trailing number)
      "Define Cyber Crime. | 2"                       (pipe separator)
      "What is SSL?\t\t\t\t\t4"                      (tab + CO number)
      "Explain X (10 marks)"                          (inline marks)
      "Explain X [10]"                                (bracket marks)
    """
    marks = None

    # 1. Inline marks: (10 marks), [10]
    inline = re.search(r'\((\d+)\s*marks?\)|\[(\d+)\]', q_text, re.IGNORECASE)
    if inline:
        marks = int(next(g for g in inline.groups() if g is not None))
        q_text = q_text[:inline.start()] + q_text[inline.end():]

    # 2. Strip watermarks early so they don't confuse later steps
    q_text = re.sub(r'(?i)(uptukhabar\.net|downloaded from[^\n]*)', '', q_text).strip()

    # 3. Pipe-separated trailing number:  "Question text | 2"
    pipe = re.match(r'^(.*?)\s*\|\s*(\d+)\s*$', q_text.strip())
    if pipe:
        q_text = pipe.group(1).strip()
        if marks is None:
            marks = int(pipe.group(2))

    # 4. Trailing bare number (tabs or spaces then digits at end of line)
    trailing = re.match(r'^(.*\S)\s+(\d{1,2})\s*$', q_text.strip())
    if trailing:
        candidate_marks = int(trailing.group(2))
        if 1 <= candidate_marks <= 100:
            q_text = trailing.group(1).strip()
            if marks is None:
                marks = candidate_marks

    # 5. Final whitespace normalisation
    q_text = re.sub(r'\s+', ' ', q_text).strip()

    return q_text, marks


def detect_metadata(text: str, pdf_path: Path):
    parts = pdf_path.parts
    try:
        year_idx = parts.index("year_2")
        subject = parts[year_idx + 1]
    except:
        subject = "UNKNOWN"

    year = 2023
    year_match = re.search(r'(20\d{2})', text)
    if year_match:
        year = int(year_match.group(1))

    subject_code = "UNKNOWN"
    code_match = re.search(r'([A-Z]{2,3}\d{3,4}[A-Z]?)', text)
    if code_match:
        subject_code = code_match.group(1)

    program = "B.Tech"
    return subject, subject_code, year, program


def get_unit_classification(question_text: str, syllabus_text: str) -> int:
    prompt = pyq_unit_classification(question_text, syllabus_text)
    MAX_RETRIES = 3
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = models.chat(
                prompt=prompt,
                model=CONFIG["model"]["model"],
                provider=CONFIG["providers"]["chat"]
            )
            raw_output = response.strip()
            match = re.search(r'(\d)', raw_output)
            if match:
                unit = int(match.group(1))
                if 1 <= unit <= 5:
                    return unit
            return 1
        except Exception as e:
            err_str = str(e)
            if attempt < MAX_RETRIES:
                wait = 15 * attempt
                print(f" ⚠ Classification attempt {attempt} failed: {err_str[:120]}")
                print(f"   Retrying in {wait}s...", end="", flush=True)
                time.sleep(wait)
            else:
                print(f" ❌ Classification failed after {MAX_RETRIES} attempts: {err_str[:120]}")
    return 1


def section_slug(section_label: str) -> str:
    """Convert 'SECTION B' -> 'sec_b', 'Part C' -> 'part_c' for use in IDs."""
    s = section_label.strip().lower()
    s = re.sub(r'\s+', '_', s)
    return s


def process_pyq(pdf_path: Path):
    print(f"\n📄 Processing PYQ: {pdf_path.name}")

    output_dir = pdf_path.parent / "pyqs_processed"
    out_file = output_dir / f"{pdf_path.stem}_processed.json"
    if out_file.exists():
        print(f"   -> Already processed ({out_file.name}). Skipping.")
        return

    raw_text = load_pdf(pdf_path)
    clean_text = normalize_text(raw_text)

    subject, subject_code, year, program = detect_metadata(raw_text, pdf_path)
    syllabus_topics = get_syllabus_topics(subject)

    # Section splitting — matches "SECTION A", "SECTION B", "Part A", etc.
    section_pattern = re.compile(r'(SECTION\s+[A-Z]|Part\s+[A-Z])', re.IGNORECASE)
    section_splits = section_pattern.split(clean_text)

    # section_splits = [pre_text, label1, body1, label2, body2, ...]
    chunks = []
    if section_splits[0].strip():
        chunks.append(("General", section_splits[0]))
    for i in range(1, len(section_splits), 2):
        if i + 1 < len(section_splits):
            chunks.append((section_splits[i].strip(), section_splits[i + 1]))

    if not chunks:
        chunks = [("General", clean_text)]

    questions_data = []

    for current_section, section_text in chunks:
        lines = section_text.split("\n")

        for line in lines:
            line = line.strip()
            if not line:
                continue

            lower_line = line.lower()
            if any(x in lower_line for x in [
                "attempt any", "attempt all", "compulsory",
                "choose any", "answer any", "note:", "time:", "total marks"
            ]):
                continue

            q_match = re.match(r'^(Q\d+|\d+)\.?\s*(.*)', line, re.IGNORECASE)
            sub_match = re.match(r'^([a-zA-Z]\)|\([a-zA-Z]\)|[a-zA-Z]\.)\s*(.*)', line, re.IGNORECASE)

            if q_match or sub_match:
                if q_match:
                    q_num = q_match.group(1).upper()
                    raw_q_text = q_match.group(2)
                else:
                    q_num = f"Sub_{sub_match.group(1).strip('()')}"
                    raw_q_text = sub_match.group(2)

                q_text, marks = clean_question_text(raw_q_text)

                if len(q_text) <= 5:
                    continue

                unit = get_unit_classification(q_text, syllabus_topics)

                # Include section slug in ID to prevent cross-section collisions
                slug = section_slug(current_section)
                q_id = f"{subject.lower()}_{year}_{pdf_path.stem}_{slug}_u{unit}_{q_num.lower()}"

                questions_data.append({
                    "question_id": q_id,
                    "program": program,
                    "subject": subject,
                    "subject_code": subject_code,
                    "year": year,
                    "unit": unit,
                    "marks": marks,
                    "question_text": q_text,
                    "section": current_section,
                    "question_number": q_num,
                    "extracted_topic": None,
                    "difficulty_estimate": None,
                    "source_pdf": pdf_path.name
                })

    # Save Output
    if questions_data:
        output_dir.mkdir(exist_ok=True)
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(questions_data, f, indent=2, ensure_ascii=False)
        print(f" ✅ Saved {len(questions_data)} questions to {out_file.name}")
    else:
        print(f" ⚠ No questions found in {pdf_path.name}")


def process_pyq_folders(base_path_str: str):
    root_path = Path(base_path_str)

    pdfs = [
        p for p in root_path.rglob("*.pdf")
        if "pyqs_processed" not in p.parts and (p.parent.name == "pyqs" or "pyqs" in p.parts)
    ]

    print(f"Found {len(pdfs)} PYQ PDFs in {base_path_str}")

    for pdf in pdfs:
        process_pyq(pdf)

    print("\n--- All PYQs processed successfully ---")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract multimodal pyqs from PDFs.")
    parser.add_argument("--path", default=BASE_PATH, help="Target directory for pyq PDFs")
    args = parser.parse_args()
    process_pyq_folders(args.path)