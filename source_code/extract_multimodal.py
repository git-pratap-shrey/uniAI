import os
import fitz  # PyMuPDF
import json
import time
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image
import io

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found in .env")

genai.configure(api_key=GEMINI_API_KEY)

BASE_PATH = r"D:\CODE-workingBuild\uniAI\source_code\data\year_2"
MODEL_NAME = "gemini-2.5-flash"
CHUNK_SIZE = 2  # Pages per chunk — kept small for OCR accuracy

# ------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------

def infer_metadata_from_path(pdf_path: Path) -> dict:
    parts = pdf_path.parts
    try:
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


def render_pages_to_images(doc, start_page: int, end_page: int) -> list:
    """Render PDF pages to PIL Images at 2x zoom (in-memory)."""
    images = []
    for page_num in range(start_page, end_page):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img = Image.open(io.BytesIO(pix.tobytes("png")))
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

def process_pdf(pdf_path: Path, model):
    print(f"\n📄 Processing: {pdf_path.name}")

    metadata_base = infer_metadata_from_path(pdf_path)
    output_dir = pdf_path.parent
    txt_path = pdf_path.with_suffix(".txt")

    doc = fitz.open(pdf_path)
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
            images = render_pages_to_images(doc, start_page, end_page)

            response = model.generate_content(
                [PROMPT] + images,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=8192,
                )
            )

            raw_response = response.text.strip()
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
            time.sleep(4)  # Rate limit (Gemini free tier ~15 RPM)

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
    model = genai.GenerativeModel(MODEL_NAME)

    pdfs = sorted(root_path.rglob("*.pdf"))
    print(f"Found {len(pdfs)} PDFs in {base_path_str}")

    for pdf in pdfs:
        process_pdf(pdf, model)

    print("\n--- All PDFs processed successfully ---")


if __name__ == "__main__":
    process_all_folders(BASE_PATH)
