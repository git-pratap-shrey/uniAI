import os
import re
import json
import fitz
import time
from pathlib import Path

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

import config
from utils import build_vlm_client
from prompts import pyq_unit_classification

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------

BASE_PATH = config.BASE_DATA_DIR
OLLAMA_CLIENT = build_vlm_client()

def get_syllabus_topics(subject: str) -> str:
    """Finds the syllabus JSON for the subject and extracts units."""
    syllabus_dir = Path(BASE_PATH) / subject / "syllabus"
    if not syllabus_dir.exists():
        return "Unit 1: Basic Concepts\nUnit 2: Intermediate\nUnit 3: Advanced\nUnit 4: Applications\nUnit 5: Case Studies"
    
    # just read the first json chunk we can find
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
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
        
        MAX_RETRIES = 3
        raw_response = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = OLLAMA_CLIENT.chat(
                    model=config.MODEL_VISION,
                    messages=[{
                        'role': 'user',
                        'content': PYQ_VLM_TRANSCRIPTION,
                        'images': [pix.tobytes("png")]
                    }]
                )
                raw_response = response['message']['content'].strip()
                break
            except Exception as e:
                err_str = str(e)
                if attempt < MAX_RETRIES:
                    wait = 15 * attempt  # 15s, 30s, 45s
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
    # Remove repeated headers/footers (naive approach: just strip leading/trailing whitespace lines)
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    # Merge lines that belong to the same question.
    # We assume a line not starting with a question marker, section marker, or specific keyword is a continuation.
    merged_lines = []
    
    q_pattern = re.compile(r'^(Q\d+\.|\d+\.|\(\w\)|[a-z]\)|[a-zA-Z]\.)', re.IGNORECASE)
    sec_pattern = re.compile(r'^(Part\s+[A-Z]|Section\s+[A-Z])', re.IGNORECASE)
    
    for line in lines:
        if not merged_lines:
            merged_lines.append(line)
            continue
        
        # If line looks like a new start, append it
        if q_pattern.match(line) or sec_pattern.match(line) or line.isupper() or line.startswith("Attempt"):
            if merged_lines[-1].endswith("-"): # hyphenation fix
                merged_lines[-1] = merged_lines[-1][:-1] + line
            else:
                merged_lines.append(line)
        else:
            # continuation of previous line
            if merged_lines[-1].endswith("-"):
                merged_lines[-1] = merged_lines[-1][:-1] + line
            else:
                merged_lines[-1] += " " + line
                
    return "\n".join(merged_lines)

def detect_metadata(text: str, pdf_path: Path):
    # Subject from path
    parts = pdf_path.parts
    try:
        year_idx = parts.index("year_2")
        subject = parts[year_idx + 1]
    except:
        subject = "UNKNOWN"
        
    year = 2023 # default
    year_match = re.search(r'(20\d{2})', text)
    if year_match:
        year = int(year_match.group(1))
        
    # Attempt to find subject code like BCC301
    subject_code = "UNKNOWN"
    code_match = re.search(r'([A-Z]{2,3}\d{3,4}[A-Z]?)', text)
    if code_match:
        subject_code = code_match.group(1)
        
    program = "B.Tech" # Assuming B.Tech
    return subject, subject_code, year, program

def get_unit_classification(question_text: str, syllabus_text: str) -> int:
    prompt = pyq_unit_classification(question_text, syllabus_text)
    MAX_RETRIES = 3
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = OLLAMA_CLIENT.chat(
                model=config.MODEL_CHAT,
                messages=[{'role': 'user', 'content': prompt}]
            )
            raw_output = response['message']['content'].strip()
            # extract first number
            match = re.search(r'(\d)', raw_output)
            if match:
                unit = int(match.group(1))
                if 1 <= unit <= 5:
                    return unit
            return 1 # default if not parsed well
        except Exception as e:
            err_str = str(e)
            if attempt < MAX_RETRIES:
                wait = 15 * attempt  # 15s, 30s, 45s
                print(f" ⚠ Classification attempt {attempt} failed: {err_str[:120]}")
                print(f"   Retrying in {wait}s...", end="", flush=True)
                time.sleep(wait)
            else:
                print(f" ❌ Classification failed after {MAX_RETRIES} attempts: {err_str[:120]}")
    return 1

def process_pyq(pdf_path: Path):
    print(f"\n📄 Processing PYQ: {pdf_path.name}")
    
    # Check if already processed
    output_dir = pdf_path.parent / "pyqs_processed"
    out_file = output_dir / f"{pdf_path.stem}_processed.json"
    if out_file.exists():
        print(f"   -> Already processed ({out_file.name}). Skipping.")
        return
        
    raw_text = load_pdf(pdf_path)
    clean_text = normalize_text(raw_text)
    
    subject, subject_code, year, program = detect_metadata(raw_text, pdf_path)
    syllabus_topics = get_syllabus_topics(subject)
    
    # Split by section
    section_splits = re.split(r'(Part\s+[A-Z]|Section\s+[A-Z]|^PART\s*.\s*[CBA])', clean_text, flags=re.IGNORECASE | re.MULTILINE)
    
    questions_data = []
    
    current_section = "General"
    
    # section_splits will be [text_before_section1, Section1, text_in_section1, Section2, text_in_section2...]
    # let's iterate robustly
    chunks = [section_splits[0]] if section_splits[0].strip() else []
    for i in range(1, len(section_splits), 2):
        if i+1 < len(section_splits):
            chunks.append((section_splits[i].strip(), section_splits[i+1]))
            
    # if it didn't split (no sections found), fallback
    if len(chunks) == 0:
        chunks = [("General", clean_text)]
        
    for chunk in chunks:
        if isinstance(chunk, tuple):
            current_section, section_text = chunk
        else:
            current_section = "General"
            section_text = chunk
            
        lines = section_text.split("\n")
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # skip meta instructions
            lower_line = line.lower()
            if any(x in lower_line for x in ["attempt any", "attempt all", "compulsory", "choose any", "answer any"]):
                continue
                
            # match Q1. Question 1. etc
            q_match = re.match(r'^(Q\d+|\d+)\.?\s*(.*)', line, re.IGNORECASE)
            sub_match = re.match(r'^([a-zA-Z]\)|\([a-zA-Z]\)|[a-zA-Z]\.)\s*(.*)', line, re.IGNORECASE)
            
            if q_match or sub_match:
                if q_match:
                    q_num = q_match.group(1).upper()
                    q_text = q_match.group(2)
                else:
                    q_num = f"Sub_{sub_match.group(1).strip('()')}"
                    q_text = sub_match.group(2)
                
                # Extract marks
                marks = None
                marks_match = re.search(r'\((\d+)\s*marks?\)|\[(\d+)\]|(\d+)\s*M|\((\d+)\)', q_text, re.IGNORECASE)
                if marks_match:
                    for i in range(1, 5):
                        if marks_match.group(i):
                            marks = int(marks_match.group(i))
                            break
                    # remove marks from text
                    q_text = q_text.replace(marks_match.group(0), "").strip()
                
                # classify unit
                unit = get_unit_classification(q_text, syllabus_topics)
                
                q_id = f"{subject.lower()}_{year}_{pdf_path.stem}_u{unit}_{q_num.lower()}"
                
                q_obj = {
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
                }
                if len(q_text) > 5: # basic sanity check
                    questions_data.append(q_obj)

    # Save Output
    if questions_data:
        output_dir = pdf_path.parent / "pyqs_processed"
        output_dir.mkdir(exist_ok=True)
        out_file = output_dir / f"{pdf_path.stem}_processed.json"
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(questions_data, f, indent=2, ensure_ascii=False)
        print(f" ✅ Saved {len(questions_data)} questions to {out_file.name}")
    else:
        print(f" ⚠ No questions found in {pdf_path.name}")


def process_pyq_folders(base_path_str: str):
    root_path = Path(base_path_str)
    
    # robustly find all pdfs inside pyqs folders, excluding pyqs_processed
    pdfs = [
        p for p in root_path.rglob("*.pdf") 
        if "pyqs_processed" not in p.parts and (p.parent.name == "pyqs" or "pyqs" in p.parts)
    ]
    
    print(f"Found {len(pdfs)} PYQ PDFs in {base_path_str}")
    
    for pdf in pdfs:
        process_pyq(pdf)
        
    print("\n--- All PYQs processed successfully ---")


if __name__ == "__main__":
    process_pyq_folders(BASE_PATH)