# from pypdf import PdfReader

# path = r"D:\CODE-workingBuild\uniAI\source_code\data\year_2\python\notes\unit1\python unit 1.pdf"  # replace with real path

# reader = PdfReader(path)

# for i, page in enumerate(reader.pages):
#     text = page.extract_text()
#     print(f"\n---------- PAGE {i+1} ----------\n")
#     print(text)


# import pdfplumber

# with pdfplumber.open(path) as pdf:
#     for i, page in enumerate(pdf.pages):
#         text = page.extract_text()
#         print(f"\n--- PAGE {i+1} ---\n{text}")

# import fitz

# doc = fitz.open(path)
# for i, page in enumerate(doc):
#     text = page.get_text()
#     print(f"\n--- PAGE {i+1} ---\n{text}")

import fitz
import re

def extract_clean_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""

    for page in doc:
        t = page.get_text("text")
        t = re.sub(r"\s+", " ", t)  # normalize spacing
        t = t.replace("", "")     # remove OCR bullets
        text += t + "\n\n"

    return text
