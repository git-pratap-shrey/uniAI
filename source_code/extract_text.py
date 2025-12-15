import os
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io

# If on Windows, need to specify Tesseract path:
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def page_has_meaningful_text(text: str, min_chars: int = 50) -> bool:
    return len(text.strip()) >= min_chars


def ocr_page(page: fitz.Page, dpi: int = 300, lang: str = "eng") -> str:
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

    for idx in range(len(doc)):
        page = doc[idx]
        text = page.get_text("text")

        should_ocr = use_ocr and not page_has_meaningful_text(text, min_chars_for_text)

        if should_ocr:
            print(f"[OCR] Page {idx+1} in {os.path.basename(pdf_path)} needs OCR...")
            try:
                ocr_text = ocr_page(page, dpi=300, lang=ocr_lang)
                if page_has_meaningful_text(ocr_text, 10):
                    text = ocr_text
            except Exception as e:
                print(f"   OCR failed for page {idx+1}: {e}")

        collected.append(f"--- PAGE {idx+1} ---\n{text}\n")

    doc.close()
    return "\n".join(collected)


def process_folder(main_folder: str, output_in_same_folder: bool = True):
    """
    Walk through all subfolders and process every PDF found.
    """
    for root, dirs, files in os.walk(main_folder):
        for file in files:
            if file.lower().endswith(".pdf"):
                pdf_path = os.path.join(root, file)

                print(f"\n>>> Extracting: {pdf_path}")

                extracted_text = extract_text_hybrid(pdf_path)

                # Output file path
                if output_in_same_folder:
                    txt_path = os.path.splitext(pdf_path)[0] + ".txt"
                else:
                    # put in central "output" folder
                    out_root = os.path.join(main_folder, "_extracted_text")
                    os.makedirs(out_root, exist_ok=True)
                    txt_path = os.path.join(out_root, file.replace(".pdf", ".txt"))

                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(extracted_text)

                print(f"✔ Saved → {txt_path}")


if __name__ == "__main__":
    MAIN_FOLDER = r"D:\CODE-workingBuild\uniAI\source_code\data\year_2\python"   # YOUR MAIN FOLDER HERE
    process_folder(MAIN_FOLDER)
