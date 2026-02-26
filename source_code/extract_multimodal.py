import os
import fitz  # PyMuPDF
import json
import time
from pathlib import Path

# --- Ensure imports work regardless of working directory ---
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

import config
from utils import pil_to_base64, extract_first_json, build_vlm_client
from prompts import NOTES_EXTRACTION

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------

BASE_PATH = config.BASE_DATA_DIR
CHUNK_SIZE = 2  # Pages per chunk — kept small for OCR accuracy

# Backend options (set MODEL_VISION_BACKEND in config.py or .env)
# "ollama"      -> local/cloud Ollama model (MODEL_VISION tag)
# "huggingface" -> HuggingFace transformers (MODEL_VISION_HF repo id)
BACKEND = config.MODEL_VISION_BACKEND.lower()

MODEL_NAME = config.MODEL_VISION

# ---- HuggingFace backend setup ----
if BACKEND == "huggingface":
    HF_MODEL_ID = config.MODEL_VISION_HF
    if not config.HF_TOKEN:
        print("⚠️  HuggingFace backend selected but HF_TOKEN not set in config/env.")
        print("   Get your token at: https://huggingface.co/settings/tokens")
    else:
        from huggingface_hub import InferenceClient as _HFClient
        HF_CLIENT = _HFClient(
            base_url="https://router.huggingface.co/v1",
            api_key=config.HF_TOKEN,
        )
        print(f"✅ HuggingFace Inference API ready → {HF_MODEL_ID}")

elif BACKEND == "ollama":
    _ollama_client = build_vlm_client()

else:
    raise ValueError(
        f"Unsupported MODEL_VISION_BACKEND: '{BACKEND}'. "
        "Choose 'ollama' or 'huggingface'."
    )


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
    Ollama cloud uses 1.5 to avoid 524 timeouts; HuggingFace uses 2.0.
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
    print(f"   Backend: {BACKEND}  |  Model: {MODEL_NAME if BACKEND != 'huggingface' else config.MODEL_VISION_HF}")

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
                if BACKEND == "ollama":
                    images = render_pages_to_images(doc, start_page, end_page, return_bytes=True, scale=1.5)
                    response = _ollama_client.chat(
                        model=MODEL_NAME,
                        messages=[{
                            'role': 'user',
                            'content': NOTES_EXTRACTION,
                            'images': images,
                        }]
                    )
                    raw_response = response['message']['content'].strip()

                elif BACKEND == "huggingface":
                    images = render_pages_to_images(doc, start_page, end_page, return_bytes=False)
                    hf_messages = [{
                        "role": "user",
                        "content": [
                            *[{
                                "type": "image_url",
                                "image_url": {"url": pil_to_base64(img)},
                            } for img in images],
                            {"type": "text", "text": NOTES_EXTRACTION},
                        ],
                    }]
                    hf_response = HF_CLIENT.chat_completion(
                        model=HF_MODEL_ID,
                        messages=hf_messages,
                        max_tokens=8192,
                    )
                    raw_response = hf_response.choices[0].message.content.strip()

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
            "processed_by": MODEL_NAME if BACKEND != "huggingface" else HF_MODEL_ID,
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

    pdfs = sorted(root_path.rglob("*.pdf"))
    print(f"Found {len(pdfs)} PDFs in {base_path_str}")

    for pdf in pdfs:
        process_pdf(pdf)

    print("\n--- All PDFs processed successfully ---")


if __name__ == "__main__":
    process_all_folders(BASE_PATH)
