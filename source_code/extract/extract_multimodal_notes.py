# python -m source_code.extract.extract_multimodal_notes

import os
import re
import fitz  # PyMuPDF
import json
import time
from pathlib import Path

# --- Ensure imports work regardless of working directory ---
import sys
# Ensure project root is on sys.path for source_code imports
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
# Also add source_code/ for direct imports
SOURCE_DIR = os.path.join(ROOT_DIR, "source_code")
if SOURCE_DIR not in sys.path:
    sys.path.append(SOURCE_DIR)

from source_code.config import CONFIG
from source_code import models
from utils import pil_to_base64, pil_to_jpeg_bytes, extract_first_json
from prompts import notes_extraction

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------

BASE_PATH = CONFIG["paths"]["base_data"]

# Backend/Provider settings now handled by models.py
BACKEND = CONFIG["providers"]["vision"].lower()
MODEL_NAME = CONFIG["providers"]["vision_model"]


# ------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------

def infer_metadata_from_path(pdf_path: Path) -> dict:
    """
    Infer metadata from flattened path.
    Expected structure: <SUBJECT>/notes/unit<N>/*.pdf
    """
    parts = pdf_path.parts
    try:
        notes_idx = parts.index("notes")
        subject = parts[notes_idx - 1]          # e.g. 'COA'
        unit_str = parts[notes_idx + 1]         # e.g. 'unit4'
        # Normalize unit to numeric string
        m = re.search(r"\d+", unit_str.lower())
        unit = str(int(m.group())) if m else "unknown"
    except (ValueError, IndexError):
        subject, unit = "unknown", "unknown"

    return {
        "subject": subject,
        "type": "notes",
        "unit": unit,
        "source_pdf": pdf_path.name,
    }


def normalize_unit(unit):
    """Normalize unit to a clean numeric string."""
    if unit is None:
        return "unknown"
    s = str(unit).strip().lower()
    m = re.search(r"\d+", s)
    return str(int(m.group())) if m else "unknown"


# ------------------------------------------------------------------
# RENDERING
# ------------------------------------------------------------------

def render_pages_to_images(doc, start_page: int, end_page: int, return_bytes=False, scale=2.0) -> list:
    """
    Render PDF pages to images.
    scale: DPI multiplier — lower = smaller payload, higher = better OCR quality.
    Ollama cloud: scale=1.0 + JPEG avoids Cloudflare 524 timeouts; HuggingFace: scale=2.0 PNG.
    """
    import io
    from PIL import Image
    images = []
    for page_num in range(start_page, end_page):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
        img_bytes = pix.tobytes("png")

        if return_bytes:
            images.append(img_bytes)
        else:
            img = Image.open(io.BytesIO(img_bytes))
            images.append(img)

    return images


# ------------------------------------------------------------------
# CORE LOGIC: Semantic Sectioning with Topic Feedback Loop
# ------------------------------------------------------------------

def process_pdf(pdf_path: Path):
    print(f"\n📄 Processing: {pdf_path.name}")
    print(f"   Provider: {CONFIG['providers']['vision']}  |  Model: {CONFIG['providers']['vision_model']}")

    metadata_base = infer_metadata_from_path(pdf_path)
    output_dir = pdf_path.parent / pdf_path.stem
    output_dir.mkdir(exist_ok=True)
    txt_path = output_dir / (pdf_path.stem + ".txt")

    if not pdf_path.exists():
        print(f"❌ File not found: {pdf_path}")
        return

    try:
        doc = fitz.open(str(pdf_path))
    except Exception as e:
        print(f"❌ Failed to open PDF {pdf_path.name}: {e}")
        return

    total_pages = len(doc)
    all_text_parts = []

    # Running topic list for this PDF — starts empty on page 1
    existing_topics: list[str] = []

    # Global chunk counter for this PDF
    global_chunk_idx = 0

    for start_page in range(0, total_pages):
        end_page = start_page + 1
        page_num_1based = start_page + 1

        # New JSON naming: <pdf_stem>_p{start}-{end}_{chunk_idx}.json
        pdf_stem = pdf_path.stem
        
        all_existing_for_page = sorted(output_dir.glob(f"{pdf_stem}_p{page_num_1based}-{page_num_1based}_*.json"))

        if len(all_existing_for_page) > 0:
            # Rehydrate existing metadata into the running topic list
            try:
                for existing_file in all_existing_for_page:
                    with open(existing_file, "r", encoding="utf-8") as f:
                        existing_data = json.load(f)
                    
                    sec = existing_data.get("extracted_metadata", {})
                    title = sec.get("section_title", "")
                    if title and title not in existing_topics:
                        existing_topics.append(title)
                    ft = sec.get("full_text", "")
                    if ft:
                        all_text_parts.append(f"\n--- PAGE {page_num_1based} [{title}] ---\n{ft}")
                
                chunks_this_page = len(all_existing_for_page)
            except Exception:
                chunks_this_page = 1

            print(f"   -> Page {page_num_1based} already processed. Skipping ({chunks_this_page} chunks).")
            global_chunk_idx += chunks_this_page
            continue

        print(f"   -> Processing Page {page_num_1based} (topics so far: {len(existing_topics)})...", end="", flush=True)

        MAX_RETRIES = 3
        raw_response = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                if BACKEND == "ollama":
                    images_pil = render_pages_to_images(doc, start_page, end_page, return_bytes=False, scale=1.0)
                    images = [pil_to_jpeg_bytes(img) for img in images_pil]
                else:
                    images = render_pages_to_images(doc, start_page, end_page, return_bytes=True)

                prompt = notes_extraction(existing_topics if existing_topics else None)
                raw_response = models.vision(
                    images=images,
                    prompt=prompt,
                    provider=CONFIG["providers"]["vision"],
                    model=CONFIG["providers"]["vision_model"]
                )
                if isinstance(raw_response, str) and raw_response.startswith("⚠ Vision Error"):
                    raise Exception(raw_response)
                break
            except Exception as e:
                err_str = str(e)
                if attempt < MAX_RETRIES:
                    wait = 5 * attempt
                    print(f" ⚠ Attempt {attempt} failed: {err_str[:120]}")
                    print(f"   Retrying in {wait}s...", end="", flush=True)
                    time.sleep(wait)
                else:
                    print(f" ❌ Failed after {MAX_RETRIES} attempts: {err_str[:120]}")

        if raw_response is None:
            time.sleep(5)
            global_chunk_idx += 1
            continue

        structured_data = extract_first_json(raw_response)
        if structured_data is None:
            print(" ⚠ No valid JSON. Saving raw.")
            structured_data = {
                "extracted_metadata": {
                    "raw_description": raw_response,
                    "full_text": raw_response,
                    "page_has_diagram": False,
                    "content_quality": "partially_legible",
                    "confidence": 0.5,
                },
                "sections": [{
                    "section_title": "Untitled",
                    "is_new_topic": True,
                    "full_text": raw_response,
                    "topics": [],
                    "key_concepts": [],
                    "has_diagram": False,
                }]
            }

        sections = structured_data.get("sections", [])
        if not sections:
            print(" ⚠ No sections extracted. Skipping page.")
            global_chunk_idx += 1
            time.sleep(1)
            continue

        # Update running topic list with newly discovered topics
        for sec in sections:
            title = sec.get("section_title", "")
            if title and sec.get("is_new_topic", False) and title not in existing_topics:
                existing_topics.append(title)
                print(f"\n   + New topic: {title}")

        # Write one JSON per section
        for sec_idx, sec in enumerate(sections):
            sec_start = page_num_1based
            sec_end = page_num_1based
            chunk_id = global_chunk_idx + sec_idx

            sec_json_filename = f"{pdf_stem}_p{sec_start}-{sec_end}_{chunk_id}.json"
            sec_json_path = output_dir / sec_json_filename

            chunk_data = {
                **metadata_base,
                "page_start": sec_start,
                "page_end": sec_end,
                "extracted_metadata": sec,
                "processed_by": MODEL_NAME,
                "chunk_size": 1,
                "section_index": sec_idx,
                "chunk_idx": chunk_id,
            }
            with open(sec_json_path, "w", encoding="utf-8") as f:
                json.dump(chunk_data, f, indent=2, ensure_ascii=False)

            ft = sec.get("full_text", "")
            if ft:
                all_text_parts.append(f"\n--- PAGE {page_num_1based} [{sec.get('section_title', 'Untitled')}] ---\n{ft}")

        print(f" ✅ Done ({len(sections)} section(s)).")
        global_chunk_idx += len(sections)
        time.sleep(1)

    doc.close()

    if all_text_parts:
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"# OCR: {pdf_path.name}\n")
            f.write("".join(all_text_parts))
        print(f"   📝 Saved full text -> {txt_path.name}")


def process_all_folders(base_path_str: str):
    root_path = Path(base_path_str)

    pdfs = [p for p in sorted(root_path.rglob("*.pdf")) if "notes" in p.parts]
    print(f"Found {len(pdfs)} notes PDFs in {base_path_str}")

    for pdf in pdfs:
        process_pdf(pdf)

    print("\n--- All notes PDFs processed successfully ---")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract multimodal notes from PDFs.")
    parser.add_argument("--path", default=BASE_PATH, help="Target directory for notes PDFs")
    args = parser.parse_args()
    process_all_folders(args.path)
