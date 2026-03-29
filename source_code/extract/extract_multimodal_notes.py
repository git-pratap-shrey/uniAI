import os
import fitz  # PyMuPDF
import json
import time
from pathlib import Path

# --- Ensure imports work regardless of working directory ---
import sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from source_code.config import CONFIG
from source_code import models
from utils import pil_to_base64, pil_to_jpeg_bytes, extract_first_json
from prompts import NOTES_EXTRACTION

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------

BASE_PATH = CONFIG["paths"]["base_data"]
CHUNK_SIZE = 1  # Pages per chunk

# Backend/Provider settings now handled by models.py
BACKEND = CONFIG["providers"]["vision"].lower()
MODEL_NAME = CONFIG["providers"]["vision_model"]


# ------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------

def infer_metadata_from_path(pdf_path: Path) -> dict:
    parts = pdf_path.parts
    try:
        year_idx = parts.index("year_2")
        subject  = parts[year_idx + 1]   # e.g. 'COA'
        doc_type = parts[year_idx + 2]   # e.g. 'notes'
        unit     = parts[year_idx + 3]   # e.g. 'unit1'
    except (ValueError, IndexError):
        subject, doc_type, unit = "unknown", "unknown", "unknown"

    return {
        "subject": subject,
        "type": doc_type,
        "unit": unit,
        "source_pdf": pdf_path.name,
    }


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
# PROMPT
# ------------------------------------------------------------------
# Moved to prompts.NOTES_EXTRACTION

# ------------------------------------------------------------------
# CORE LOGIC
# ------------------------------------------------------------------

def process_pdf(pdf_path: Path):
    print(f"\n📄 Processing: {pdf_path.name}")
    print(f"   Provider: {CONFIG['providers']['vision']}  |  Model: {CONFIG['model']['model']}")

    metadata_base = infer_metadata_from_path(pdf_path)
    # Write all chunk JSONs and .txt into a per-PDF subfolder so that
    # multiple PDFs in the same unit folder never collide on chunk names.
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

    for start_page in range(0, total_pages, CHUNK_SIZE):
        end_page = min(start_page + CHUNK_SIZE, total_pages)

        json_filename = f"chunk_{start_page + 1}_{end_page}.json"
        json_path = output_dir / json_filename

        if json_path.exists():
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

        MAX_RETRIES = 3
        raw_response = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                # Delegate vision call to central registry
                if BACKEND == "ollama":
                    # Render as PIL then re-encode as JPEG (5-10x smaller than PNG)
                    images_pil = render_pages_to_images(doc, start_page, end_page, return_bytes=False, scale=1.0)
                    images = [pil_to_jpeg_bytes(img) for img in images_pil]
                else: 
                     # For HuggingFace or others, use default scaling or bytes
                     images = render_pages_to_images(doc, start_page, end_page, return_bytes=True)

                raw_response = models.vision(
                    images=images,
                    prompt=NOTES_EXTRACTION,
                    provider=CONFIG["providers"]["vision"],
                    model=CONFIG["providers"]["vision_model"]
                )

                break  # success — exit retry loop

            except Exception as e:
                err_str = str(e)
                if attempt < MAX_RETRIES:
                    wait = 5 * attempt  # 15s, 30s, 45s
                    print(f" ⚠ Attempt {attempt} failed: {err_str[:120]}")
                    print(f"   Retrying in {wait}s...", end="", flush=True)
                    time.sleep(wait)
                else:
                    print(f" ❌ Failed after {MAX_RETRIES} attempts: {err_str[:120]}")

        if raw_response is None:
            time.sleep(5)
            continue

        structured_data = extract_first_json(raw_response)
        if structured_data is None:
            print(" ⚠ No valid JSON. Saving raw.")
            structured_data = {"raw_description": raw_response, "full_text": raw_response}

        full_text = structured_data.get("full_text", "")
        if full_text:
            all_text_parts.append(f"\n--- PAGES {start_page+1}-{end_page} ---\n{full_text}")

        chunk_data = {
            **metadata_base,
            "page_start": start_page + 1,
            "page_end": end_page,
            "extracted_metadata": structured_data,
            "processed_by": MODEL_NAME,  # Use centralized model name
            "chunk_size": end_page - start_page,
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(chunk_data, f, indent=2, ensure_ascii=False)

        print(" ✅ Done.")
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
