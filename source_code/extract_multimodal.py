import os
import fitz  # PyMuPDF
import json
import time
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai
import ollama
from PIL import Image
import io

# --- Ensure imports work regardless of working directory ---
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

import config

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------

# Use configuration from config.py
BASE_PATH = config.BASE_DATA_DIR
MODEL_NAME = config.MODEL_VISION  # Defaults to "qwen3 vl 235b cloud"
CHUNK_SIZE = 2  # Pages per chunk — kept small for OCR accuracy

# Constants for model types
MODEL_TYPE_OLLAMA = "ollama"
MODEL_TYPE_GEMINI = "gemini"

def get_model_type(model_name):
    if "gemini" in model_name.lower():
        return MODEL_TYPE_GEMINI
    return MODEL_TYPE_OLLAMA

# Configure Gemini if needed
if get_model_type(MODEL_NAME) == MODEL_TYPE_GEMINI:
    if not config.GEMINI_API_KEY:
         print("⚠️ Gemini model selected but GEMINI_API_KEY not set in config/env.")
    else:
        genai.configure(api_key=config.GEMINI_API_KEY)


# ------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------

def infer_metadata_from_path(pdf_path: Path) -> dict:
    parts = pdf_path.parts
    try:
        # Find 'year_2' in path to anchor specific folder structure
        # Adjust index based on actual path structure if 'year_2' is not unique
        year_idx = parts.index("year_2") 
        subject = parts[year_idx + 1]
        doc_type = parts[year_idx + 2]
        unit = parts[year_idx + 3]
    except (ValueError, IndexError):
        subject, doc_type, unit = "unknown", "unknown", "unknown"

    return {
        "subject": subject,
        "type": doc_type,
        "unit": unit,
        "source_pdf": pdf_path.name,
    }


def render_pages_to_images(doc, start_page: int, end_page: int, return_bytes=False) -> list:
    """
    Render PDF pages to images.
    If using Ollama, we need raw bytes.
    If using Gemini, we might use PIL Images.
    """
    images = []
    for page_num in range(start_page, end_page):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img_bytes = pix.tobytes("png")
        
        if return_bytes:
            images.append(img_bytes)
        else:
            img = Image.open(io.BytesIO(img_bytes))
            images.append(img)
            
    return images


def extract_first_json(text: str):
    """Extract the first valid JSON object from a string by brace-counting."""
    start_index = text.find('{')
    if start_index == -1:
        return None

    count = 0
    for i, char in enumerate(text[start_index:], start=start_index):
        if char == '{':
            count += 1
        elif char == '}':
            count -= 1
        if count == 0:
            try:
                return json.loads(text[start_index:i+1])
            except json.JSONDecodeError:
                return None
    return None


# ------------------------------------------------------------------
# PROMPT
# ------------------------------------------------------------------

PROMPT = """You are an OCR + metadata extraction system for university course materials.

You will receive images of PDF pages. These may be handwritten notes, printed slides, question papers, or diagrams. Text CANNOT be copy-pasted from these — you must read them visually.

## Your Tasks

### Task 1 — Full OCR
Read ALL visible text from the images. Transcribe it faithfully:
- Preserve headings, bullet points, numbering, and structure
- For handwritten text, do your best to read it accurately
- For code snippets, preserve indentation and syntax
- Skip watermarks, page numbers, and headers/footers
- If a diagram is present, describe it briefly in [DIAGRAM: ...]

### Task 2 — Structured Metadata
Classify and tag the content you extracted.

## Output Format
Return ONLY a valid JSON object (no markdown fences, no extra text):

{
  "full_text": "The complete transcribed text from all pages, preserving structure with newlines",
  "title": "The topic title visible on the pages (e.g. 'Functions in Python', '2023 End Sem Paper')",
  "unit": "Unit number if identifiable (e.g. '1', '3'), else null",
  "document_type": "One of: question_paper, handwritten_notes, printed_notes, syllabus, lab_manual, other",
  "topics": ["List of specific subtopics covered in these pages"],
  "key_concepts": ["Important definitions, formulas, theorems, or algorithms mentioned"],
  "diagrams_present": false,
  "content_quality": "One of: clear, partially_legible, illegible",
  "confidence": 0.85
}

## Rules
- full_text must contain the ACTUAL text from the pages. This is the most important field.
- Be thorough — every readable sentence matters for search.
- Do NOT invent content that isn't visible.
- If pages are completely illegible, set confidence to 0.1 and full_text to empty string.
- confidence is a float between 0.0 and 1.0 reflecting OCR accuracy.
"""

# ------------------------------------------------------------------
# CORE LOGIC
# ------------------------------------------------------------------

def process_pdf(pdf_path: Path):
    print(f"\n📄 Processing: {pdf_path.name}")
    print(f"   Model: {MODEL_NAME}")
    
    model_type = get_model_type(MODEL_NAME)

    metadata_base = infer_metadata_from_path(pdf_path)
    output_dir = pdf_path.parent
    txt_path = pdf_path.with_suffix(".txt")

    # Verify file exists before opening
    if not pdf_path.exists():
        print(f"❌ File not found: {pdf_path}")
        return

    try:
        doc = fitz.open(str(pdf_path)) # Convert Path to string for PyMuPDF
    except Exception as e:
        print(f"❌ Failed to open PDF {pdf_path.name}: {e}")
        return
    
    total_pages = len(doc)

    # Accumulate full text for the .txt file
    all_text_parts = []

    for start_page in range(0, total_pages, CHUNK_SIZE):
        end_page = min(start_page + CHUNK_SIZE, total_pages)

        # JSON named chunk_start_end.json for ingestion compatibility
        json_filename = f"chunk_{start_page + 1}_{end_page}.json"
        json_path = output_dir / json_filename

        if json_path.exists():
            # Load existing text for the .txt file
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                ft = existing.get("extracted_metadata", {}).get("full_text", "")
                if ft:
                    all_text_parts.append(f"\n--- PAGES {start_page+1}-{end_page} ---\n{ft}")
            except Exception:
                pass
            print(f"   -> Chunk {start_page + 1}-{end_page} already processed. Skipping.")
            continue

        print(f"   -> Processing Chunk {start_page + 1}-{end_page}...", end="", flush=True)

        try:
            # Prepare Images
            if model_type == MODEL_TYPE_OLLAMA:
                # Ollama expects bytes
                images = render_pages_to_images(doc, start_page, end_page, return_bytes=True)
                
                # Initialize client explicitly with config to avoid default host issues
                client = ollama.Client(host=config.OLLAMA_BASE_URL)

                # Ollama chat/generate
                response = client.chat(
                    model=MODEL_NAME,
                    messages=[{
                        'role': 'user',
                        'content': PROMPT,
                        'images': images
                    }]
                )
                raw_response = response['message']['content'].strip()

            elif model_type == MODEL_TYPE_GEMINI:
                # Gemini expects PIL images
                images = render_pages_to_images(doc, start_page, end_page, return_bytes=False)
                gemini_model = genai.GenerativeModel(MODEL_NAME)
                
                response = gemini_model.generate_content(
                    [PROMPT] + images,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.1,
                        max_output_tokens=8192,
                    )
                )
                raw_response = response.text.strip()
            
            else:
                 raise ValueError(f"Unsupported model type for {MODEL_NAME}")

            structured_data = extract_first_json(raw_response)

            if structured_data is None:
                print(" ⚠ No valid JSON. Saving raw.")
                structured_data = {"raw_description": raw_response, "full_text": raw_response}

            # Collect full text
            full_text = structured_data.get("full_text", "")
            if full_text:
                all_text_parts.append(f"\n--- PAGES {start_page+1}-{end_page} ---\n{full_text}")

            # Save chunk JSON
            chunk_data = {
                **metadata_base,
                "page_start": start_page + 1,
                "page_end": end_page,
                "extracted_metadata": structured_data,
                "processed_by": MODEL_NAME,
                "chunk_size": end_page - start_page,
            }

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(chunk_data, f, indent=2, ensure_ascii=False)

            print(" ✅ Done.")
            
            if model_type == MODEL_TYPE_GEMINI:
                time.sleep(4)  # Rate limit for Gemini
            else:
                time.sleep(1) # Small pause for local/cloud ollama

        except Exception as e:
            print(f" ❌ Failed: {e}")
            time.sleep(5)

    doc.close()

    # Write combined .txt file with all OCR text
    if all_text_parts:
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"# OCR: {pdf_path.name}\n")
            f.write("".join(all_text_parts))
        print(f"   📝 Saved full text -> {txt_path.name}")


def process_all_folders(base_path_str: str):
    root_path = Path(base_path_str)
    
    pdfs = sorted(root_path.rglob("*.pdf"))
    print(f"Found {len(pdfs)} PDFs in {base_path_str}")

    for pdf in pdfs:
        process_pdf(pdf)

    print("\n--- All PDFs processed successfully ---")


if __name__ == "__main__":
    process_all_folders(BASE_PATH)
