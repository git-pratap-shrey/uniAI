import os
import fitz  # PyMuPDF
import ollama
import json

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------

BASE_PATH = r"D:\CODE-workingBuild\uniAI\source_code\data\year_2"
MODEL_NAME = "deepseek-ocr"

# ------------------------------------------------------------------
# METADATA HELPERS
# ------------------------------------------------------------------

def infer_metadata_from_path(pdf_path: str) -> dict:
    """
    Expected path structure:
    year_2/{subject}/{type}/unitX/filename.pdf
    """
    parts = pdf_path.replace("\\", "/").split("/")

    try:
        year = parts[-6]
        subject = parts[-5].lower()
        doc_type = parts[-4].lower()     # notes | pyqs | syllabus
        unit = parts[-3].lower() if "unit" in parts[-3].lower() else "unknown"
    except IndexError:
        year, subject, doc_type, unit = "unknown", "unknown", "unknown", "unknown"

    return {
        "year": year,
        "subject": subject,
        "type": doc_type,
        "unit": unit,
        "source_pdf": os.path.basename(pdf_path),
    }

# ------------------------------------------------------------------
# OCR EXTRACTION
# ------------------------------------------------------------------

def extract_and_save_by_page(pdf_path: str):
    print(f"   > Opening: {os.path.basename(pdf_path)}")

    txt_path = os.path.splitext(pdf_path)[0] + ".txt"
    meta_path = os.path.splitext(pdf_path)[0] + ".json"

    metadata = infer_metadata_from_path(pdf_path)
    metadata["pages"] = []

    doc = fitz.open(pdf_path)

    with open(txt_path, "w", encoding="utf-8") as text_out:
        for page_num in range(len(doc)):
            print(f"     * OCR Page {page_num + 1}/{len(doc)}...", end="", flush=True)

            try:
                page = doc.load_page(page_num)
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                img_data = pix.tobytes("png")

                response = ollama.generate(
                    model=MODEL_NAME,
                    prompt=(
                        "Perform OCR. Remove watermarks, page numbers, "
                        "and OCR noise. Preserve headings if possible."
                    ),
                    images=[img_data],
                    stream=False,
                    keep_alive=0
                )

                page_text = response.get("response", "").strip()

                text_out.write(f"\n--- PAGE {page_num + 1} ---\n")
                text_out.write(page_text + "\n")

                metadata["pages"].append({
                    "page": page_num + 1,
                    "ocr": True,
                    "char_count": len(page_text)
                })

                pix = None
                img_data = None

                print(" Success.")

            except Exception as err:
                print(f" FAILED.")
                text_out.write(
                    f"\n--- PAGE {page_num + 1} ERROR ---\n{err}\n"
                )
                metadata["pages"].append({
                    "page": page_num + 1,
                    "ocr": True,
                    "error": str(err)
                })

    doc.close()

    with open(meta_path, "w", encoding="utf-8") as meta_out:
        json.dump(metadata, meta_out, indent=2)

    print(f"   ✔ Saved text + metadata")

# ------------------------------------------------------------------
# WALK DATASET
# ------------------------------------------------------------------

def process_all_folders(base_path: str):
    print("--- Recursive Local DeepSeek-OCR Start ---")

    for root, _, files in os.walk(base_path):
        pdf_files = [f for f in files if f.lower().endswith(".pdf")]

        if not pdf_files:
            continue

        print(f"\n[Folder] {root}")

        for filename in pdf_files:
            pdf_path = os.path.join(root, filename)
            txt_path = os.path.splitext(pdf_path)[0] + ".txt"

            if os.path.exists(txt_path):
                print(f"   > Skipping (already exists): {filename}")
                continue

            extract_and_save_by_page(pdf_path)

# ------------------------------------------------------------------

if __name__ == "__main__":
    process_all_folders(BASE_PATH)
    print("\n--- All local folders processed successfully ---")