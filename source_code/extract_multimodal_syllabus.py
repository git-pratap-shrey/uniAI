"""
extract_multimodal_syllabus.py
──────────────────────────────
Extracts structured syllabus data from a single-page (or few-page) university
syllabus PDF and writes exactly 7 chunk JSON files per syllabus:

  syllabus_unit_1.json  – Unit I topics
  syllabus_unit_2.json  – Unit II topics
  syllabus_unit_3.json  – Unit III topics
  syllabus_unit_4.json  – Unit IV topics
  syllabus_unit_5.json  – Unit V topics
  syllabus_co.json      – Course Outcomes with Bloom's levels
  syllabus_books.json   – Textbooks + Reference books

Each file follows the schema:
{
  "subject": "COA",
  "type": "syllabus",
  "syllabus_version": "BCS302",
  "chunk_type": "unit_1" | "unit_2" | ... | "course_outcomes" | "books_references",
  "unit": "1",            # only for unit chunks
  "topics": [...],        # topic strings extracted from the syllabus
  "full_text": "...",     # human-readable representation of this section
  "source_pdf": "coa_syllabus.pdf",
  "processed_by": "<model>",
}

Usage:
  python extract_multimodal_syllabus.py
"""

import os
import sys
import json
import time
import base64
import io
from pathlib import Path

import fitz          # PyMuPDF
from PIL import Image
from dotenv import load_dotenv

# ── ensure source_code/ is on the path ────────────────────────────────────────
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

import config

load_dotenv()

# ──────────────────────────────────────────────────────────────────────────────
# BACKEND SETUP  (mirrors extract_multimodal.py)
# ──────────────────────────────────────────────────────────────────────────────
BACKEND    = config.MODEL_VISION_BACKEND.lower()
MODEL_NAME = config.MODEL_VISION

if BACKEND == "gemini":
    import google.generativeai as genai
    if not config.GEMINI_API_KEY:
        print("⚠️  Gemini backend selected but GEMINI_API_KEY not set.")
    else:
        genai.configure(api_key=config.GEMINI_API_KEY)

elif BACKEND == "huggingface":
    HF_MODEL_ID = config.MODEL_VISION_HF
    if not config.HF_TOKEN:
        print("⚠️  HuggingFace backend selected but HF_TOKEN not set.")
    else:
        from huggingface_hub import InferenceClient as _HFClient
        HF_CLIENT = _HFClient(
            base_url="https://router.huggingface.co/v1",
            api_key=config.HF_TOKEN,
        )

elif BACKEND == "ollama":
    import ollama as _ollama
    _ollama_headers = {}
    if config.OLLAMA_API_KEY:
        _ollama_headers["Authorization"] = f"Bearer {config.OLLAMA_API_KEY}"
    _ollama_client = _ollama.Client(host=config.OLLAMA_BASE_URL, headers=_ollama_headers)

# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def render_pdf_to_images(pdf_path: Path, scale: float = 2.0) -> list:
    """Render every page of the PDF to a PIL Image."""
    doc = fitz.open(str(pdf_path))
    images = []
    for page in doc:
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        images.append(img)
    doc.close()
    return images


def pil_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


def pil_to_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def extract_first_json(text: str):
    """Extract the first complete JSON object from a (possibly noisy) string."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(text[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        if depth == 0:
            try:
                return json.loads(text[start : i + 1])
            except json.JSONDecodeError:
                return None
    return None


# ──────────────────────────────────────────────────────────────────────────────
# VLM PROMPT
# ──────────────────────────────────────────────────────────────────────────────

SYLLABUS_PROMPT = """\
You are a precise syllabus extraction system for university course documents.

You will receive image(s) of a university course syllabus page (typically a single \
dense table). Extract ALL information and return it as a single valid JSON object \
with the structure below. Do NOT output markdown fences, comments, or any text \
outside the JSON.

Required JSON structure:

{
  "syllabus_version": "The course code printed on the syllabus, e.g. 'BCS302', 'BCC302'. If multiple codes appear (e.g. 'BCC302 / BCC402H'), use the primary/first one.",
  "subject_name": "Full subject title, e.g. 'Computer Organization and Architecture'",
  "units": [
    {
      "unit_number": 1,
      "unit_title": "Short title if present, e.g. 'Introduction' or 'Arithmetic and logic unit'",
      "topics": [
        "Each distinct sub-topic as a separate string. Split by commas/semicolons where appropriate.",
        "..."
      ],
      "proposed_lectures": 8,
      "full_text": "Complete verbatim text block for this unit exactly as printed"
    }
  ],
  "course_outcomes": [
    {
      "co_number": 1,
      "description": "Full CO description text",
      "blooms_level": ["K1", "K2"]
    }
  ],
  "textbooks": [
    "Full citation string for each textbook"
  ],
  "reference_books": [
    "Full citation string for each reference book (if a separate reference section exists, else empty array)"
  ]
}

Rules:
- Extract EXACTLY as many units as appear (usually 5).
- topics must be individual sub-topic strings — split compound topics at commas where sensible.
- blooms_level is an array of strings like ["K1", "K2"] or ["K3"].
- If textbooks and references are in the same list with no separation, put all in textbooks and leave reference_books empty.
- full_text for each unit should be the complete raw sentence(s) from the syllabus for that unit.
- Do not invent content not visible in the image.
"""

# ──────────────────────────────────────────────────────────────────────────────
# VLM CALL
# ──────────────────────────────────────────────────────────────────────────────

def call_vlm(images: list, max_retries: int = 3) -> dict | None:
    """
    Send syllabus images to the configured VLM backend.
    Returns parsed JSON dict or None on failure.
    """
    for attempt in range(1, max_retries + 1):
        try:
            raw = None

            # ── GEMINI ───────────────────────────────────────────────────────
            if BACKEND == "gemini":
                model = genai.GenerativeModel(MODEL_NAME)
                response = model.generate_content(
                    [SYLLABUS_PROMPT] + images,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.1,
                        max_output_tokens=8192,
                    ),
                )
                raw = response.text.strip()

            # ── OLLAMA ───────────────────────────────────────────────────────
            elif BACKEND == "ollama":
                img_bytes = [pil_to_bytes(img) for img in images]
                response = _ollama_client.chat(
                    model=MODEL_NAME,
                    messages=[{
                        "role": "user",
                        "content": SYLLABUS_PROMPT,
                        "images": img_bytes,
                    }],
                )
                raw = response["message"]["content"].strip()

            # ── HUGGINGFACE ──────────────────────────────────────────────────
            elif BACKEND == "huggingface":
                messages = [{
                    "role": "user",
                    "content": [
                        *[{"type": "image_url", "image_url": {"url": pil_to_base64(img)}}
                          for img in images],
                        {"type": "text", "text": SYLLABUS_PROMPT},
                    ],
                }]
                hf_resp = HF_CLIENT.chat_completion(
                    model=HF_MODEL_ID,
                    messages=messages,
                    max_tokens=8192,
                )
                raw = hf_resp.choices[0].message.content.strip()

            else:
                raise ValueError(f"Unsupported backend: '{BACKEND}'")

            parsed = extract_first_json(raw)
            if parsed:
                return parsed
            print(f"   ⚠ Attempt {attempt}: VLM returned no valid JSON. Raw preview:\n   {raw[:200]}")

        except Exception as exc:
            err = str(exc)[:120]
            if attempt < max_retries:
                wait = 15 * attempt
                print(f"   ⚠ Attempt {attempt} failed: {err} — retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"   ❌ All {max_retries} attempts failed: {err}")

    return None


# ──────────────────────────────────────────────────────────────────────────────
# CHUNK BUILDERS
# ──────────────────────────────────────────────────────────────────────────────

def _base_meta(subject: str, syllabus_version: str, source_pdf: str, model: str) -> dict:
    return {
        "subject": subject,
        "type": "syllabus",
        "syllabus_version": syllabus_version,
        "source_pdf": source_pdf,
        "processed_by": model,
    }


def build_unit_chunk(unit_data: dict, base: dict) -> dict:
    n = unit_data.get("unit_number", 0)
    topics_raw = unit_data.get("topics", [])
    full_text = unit_data.get("full_text", "")
    title = unit_data.get("unit_title", f"Unit {n}")
    lectures = unit_data.get("proposed_lectures", None)

    # Build a rich full_text if VLM didn't include one
    if not full_text and topics_raw:
        full_text = f"Unit {n} – {title}\n" + "\n".join(f"- {t}" for t in topics_raw)

    prefix = f"Unit {n} – {title}"
    if lectures:
        prefix += f" ({lectures} lectures)"

    return {
        **base,
        "chunk_type": f"unit_{n}",
        "unit": str(n),
        "unit_title": title,
        "topics": topics_raw,
        "proposed_lectures": lectures,
        "full_text": f"{prefix}\n{full_text}",
    }


def build_co_chunk(cos: list, base: dict) -> dict:
    lines = []
    for co in cos:
        num = co.get("co_number", "?")
        desc = co.get("description", "")
        bl   = ", ".join(co.get("blooms_level", []))
        lines.append(f"CO {num} [{bl}]: {desc}")

    return {
        **base,
        "chunk_type": "course_outcomes",
        "unit": None,
        "topics": [f"CO {co.get('co_number')}: {co.get('description','')}" for co in cos],
        "course_outcomes": cos,
        "full_text": "Course Outcomes:\n" + "\n".join(lines),
    }


def build_books_chunk(textbooks: list, reference_books: list, base: dict) -> dict:
    all_books = []
    lines = []

    if textbooks:
        lines.append("Textbooks:")
        for i, b in enumerate(textbooks, 1):
            lines.append(f"  {i}. {b}")
            all_books.append(b)

    if reference_books:
        lines.append("\nReference Books:")
        for i, b in enumerate(reference_books, 1):
            lines.append(f"  {i}. {b}")
            all_books.append(b)

    return {
        **base,
        "chunk_type": "books_references",
        "unit": None,
        "topics": all_books,
        "textbooks": textbooks,
        "reference_books": reference_books,
        "full_text": "\n".join(lines),
    }


# ──────────────────────────────────────────────────────────────────────────────
# MAIN PROCESSOR
# ──────────────────────────────────────────────────────────────────────────────

def infer_subject_from_path(pdf_path: Path) -> str:
    """
    Infer study subject from the path.
    Expected structure: .../year_2/<SUBJECT>/syllabus/<name>.pdf
    """
    parts = pdf_path.parts
    try:
        year_idx = parts.index("year_2")
        return parts[year_idx + 1].lower()         # e.g. 'COA' or 'PYTHON'
    except (ValueError, IndexError):
        return "unknown"


def process_syllabus(pdf_path: Path, force: bool = False):
    """
    Process a single syllabus PDF and write 7 chunk JSONs beside it.

    Output files (in same directory as the PDF):
      syllabus_unit_1.json … syllabus_unit_5.json
      syllabus_co.json
      syllabus_books.json
    """
    print(f"\n📖 Syllabus: {pdf_path.name}")
    print(f"   Backend : {BACKEND}  |  Model: {MODEL_NAME}")

    output_dir   = pdf_path.parent
    subject      = infer_subject_from_path(pdf_path)

    # ── Skip if all 7 chunks already exist ──
    expected_files = (
        [output_dir / f"syllabus_unit_{n}.json" for n in range(1, 6)]
        + [output_dir / "syllabus_co.json", output_dir / "syllabus_books.json"]
    )
    if not force and all(f.exists() for f in expected_files):
        print("   ✅ All 7 chunk files already exist — skipping.")
        return

    # ── Render PDF to images ──
    print("   Rendering pages...", end="", flush=True)
    try:
        images = render_pdf_to_images(pdf_path, scale=2.0)
    except Exception as exc:
        print(f"\n   ❌ Could not render PDF: {exc}")
        return
    print(f" {len(images)} page(s).")

    # ── Call VLM ──
    print("   Calling VLM...", end="", flush=True)
    parsed = call_vlm(images)
    if not parsed:
        print("\n   ❌ VLM extraction failed. Skipping this syllabus.")
        return
    print(" ✅ Parsed.")

    # ── Extract fields ──
    syllabus_version = parsed.get("syllabus_version", "unknown")
    subject_name     = parsed.get("subject_name", subject)
    units            = parsed.get("units", [])
    cos              = parsed.get("course_outcomes", [])
    textbooks        = parsed.get("textbooks", [])
    reference_books  = parsed.get("reference_books", [])

    model_id = MODEL_NAME if BACKEND != "huggingface" else config.MODEL_VISION_HF
    base = _base_meta(subject, syllabus_version, pdf_path.name, model_id)
    base["subject_name"] = subject_name

    # ── Write unit chunks ──
    written = 0
    for unit_data in units:
        n = unit_data.get("unit_number")
        if n is None:
            continue
        out_file = output_dir / f"syllabus_unit_{n}.json"
        if out_file.exists() and not force:
            print(f"   -> Unit {n} chunk exists — skipping.")
            continue
        chunk = build_unit_chunk(unit_data, base)
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(chunk, f, indent=2, ensure_ascii=False)
        print(f"   ✅ {out_file.name}")
        written += 1

    # ── Write CO chunk ──
    co_file = output_dir / "syllabus_co.json"
    if cos and (not co_file.exists() or force):
        chunk = build_co_chunk(cos, base)
        with open(co_file, "w", encoding="utf-8") as f:
            json.dump(chunk, f, indent=2, ensure_ascii=False)
        print(f"   ✅ {co_file.name}")
        written += 1
    elif not cos:
        print("   ⚠️  No course outcomes found in VLM response.")

    # ── Write books chunk ──
    books_file = output_dir / "syllabus_books.json"
    if (textbooks or reference_books) and (not books_file.exists() or force):
        chunk = build_books_chunk(textbooks, reference_books, base)
        with open(books_file, "w", encoding="utf-8") as f:
            json.dump(chunk, f, indent=2, ensure_ascii=False)
        print(f"   ✅ {books_file.name}")
        written += 1
    elif not textbooks and not reference_books:
        print("   ⚠️  No books found in VLM response.")

    print(f"\n   📦 Done — {written} chunk(s) written to {output_dir}")


def process_all_syllabuses(base_path_str: str, force: bool = False):
    """
    Scan the data directory tree for all files named *syllabus*.pdf
    (inside a 'syllabus' folder) and process each one.
    """
    root = Path(base_path_str)
    pdfs = sorted(root.rglob("*syllabus*.pdf"))

    if not pdfs:
        # Fallback: any PDF inside a folder literally named 'syllabus'
        pdfs = sorted(p for p in root.rglob("*.pdf") if "syllabus" in p.parent.name.lower())

    print(f"Found {len(pdfs)} syllabus PDF(s) under {base_path_str}")

    for pdf in pdfs:
        process_syllabus(pdf, force=force)

    print("\n--- All syllabuses processed ---")


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract syllabus PDFs into structured JSON chunks."
    )
    parser.add_argument(
        "--path",
        default=config.BASE_DATA_DIR,
        help="Root data directory (default: config.BASE_DATA_DIR)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-process and overwrite existing chunk files.",
    )
    parser.add_argument(
        "--pdf",
        default=None,
        help="Process a single PDF file instead of scanning the whole tree.",
    )
    args = parser.parse_args()

    if args.pdf:
        pdf_file = Path(args.pdf)
        if not pdf_file.exists():
            print(f"❌ File not found: {pdf_file}")
            sys.exit(1)
        process_syllabus(pdf_file, force=args.force)
    else:
        process_all_syllabuses(args.path, force=args.force)
