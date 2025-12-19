import os
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def parse_metadata_from_path(pdf_path: str) -> dict:
    parts = pdf_path.replace("\\", "/").split("/")
    meta = {
        "year": None,
        "subject": None,
        "category": None,
        "unit": None,
        "source_file": os.path.basename(pdf_path),
    }
    for i, p in enumerate(parts):
        if p.startswith("year_"):
            meta["year"] = p
            meta["subject"] = parts[i + 1]
            meta["category"] = parts[i + 2]
            if meta["category"] == "notes":
                meta["unit"] = parts[i + 3]
            else:
                meta["unit"] = "syllabus"
    return meta


def page_has_meaningful_text(text: str, min_chars: int = 50) -> bool:
    return len(text.strip()) >= min_chars


def ocr_page(page: fitz.Page, dpi: int = 600, lang: str = "eng") -> str:
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    image = Image.open(io.BytesIO(pix.tobytes("png")))
    return pytesseract.image_to_string(image, lang=lang)


def extract_text_hybrid(pdf_path: str,
                        use_ocr: bool = True,
                        ocr_lang: str = "eng",
                        min_chars_for_text: int = 50) -> str:

    doc = fitz.open(pdf_path)
    collected = []
    ocr_pages = 0
    direct_text_pages = 0

    for idx in range(len(doc)):
        page = doc[idx]
        text = page.get_text("text")
        text_length = len(text.strip())

        should_ocr = use_ocr and not page_has_meaningful_text(text, min_chars_for_text)

        if should_ocr:
            print(f"  [Page {idx+1}] Text extraction: {text_length} chars → TRIGGERING OCR")
            try:
                ocr_text = ocr_page(page, dpi=300, lang=ocr_lang)
                ocr_length = len(ocr_text.strip())
                print(f"  [Page {idx+1}] OCR result: {ocr_length} chars")
                
                if page_has_meaningful_text(ocr_text, 10):
                    text = ocr_text
                    ocr_pages += 1
                    print(f"  [Page {idx+1}] ✓ OCR text used")
                else:
                    print(f"  [Page {idx+1}] ✗ OCR returned too little text ({ocr_length} < 10)")
            except Exception as e:
                print(f"  [Page {idx+1}] ✗ OCR FAILED: {e}")
        else:
            direct_text_pages += 1
            print(f"  [Page {idx+1}] Direct text extraction: {text_length} chars → SKIPPED OCR")

        collected.append(f"--- PAGE {idx+1} ---\n{text}\n")

    doc.close()
    
    print(f"\n>>> Summary: {direct_text_pages} pages direct text, {ocr_pages} pages OCR'd\n")
    return "\n".join(collected)


def process_folder(main_folder: str, output_in_same_folder: bool = True):
    for root, dirs, files in os.walk(main_folder):
        for file in files:
            if file.lower().endswith(".pdf"):
                pdf_path = os.path.join(root, file)

                print(f"\n>>> Extracting: {pdf_path}")

                use_ocr = "syllabus" not in root.lower()
                extracted_text = extract_text_hybrid(pdf_path, use_ocr=use_ocr, min_chars_for_text=100)

                if output_in_same_folder:
                    txt_path = os.path.splitext(pdf_path)[0] + ".txt"
                else:
                    out_root = os.path.join(main_folder, "_extracted_text")
                    os.makedirs(out_root, exist_ok=True)
                    txt_path = os.path.join(out_root, file.replace(".pdf", ".txt"))

                meta = parse_metadata_from_path(pdf_path)

                header = (
                    f"YEAR: {meta['year']}\n"
                    f"SUBJECT: {meta['subject']}\n"
                    f"CATEGORY: {meta['category']}\n"
                    f"UNIT: {meta['unit']}\n"
                    f"SOURCE_FILE: {meta['source_file']}\n"
                    "----------------------------------------\n\n"
                )

                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(header)
                    f.write(extracted_text)

                print(f"✓ Saved → {txt_path}")


if __name__ == "__main__":
    MAIN_FOLDER = r"D:\CODE-workingBuild\uniAI\source_code\data\year_2"
    process_folder(MAIN_FOLDER)