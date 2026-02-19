import fitz  # PyMuPDF
import io
from google.cloud import vision
import os

# Make sure GOOGLE_APPLICATION_CREDENTIALS environment variable points to your JSON key file
# export GOOGLE_APPLICATION_CREDENTIALS="path/to/your-key.json"

def pdf_to_text(pdf_path, output_txt_path):
    """Convert PDF to text using Google Vision API"""
    
    # Set credentials path
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"D:\CODE-workingBuild\cr.json"
    client = vision.ImageAnnotatorClient()
    doc = fitz.open(pdf_path)
    all_text = []
    
    print(f"Processing {pdf_path}...")
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        
        # Convert page to image
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better quality
        image_bytes = pix.tobytes("png")
        
        # Call Google Vision API
        image = vision.Image(content=image_bytes)
        response = client.document_text_detection(image=image)
        
        if response.full_text_annotation:
            text = response.full_text_annotation.text
            all_text.append(f"--- PAGE {page_num + 1} ---\n{text}\n")
            print(f"  ✓ Page {page_num + 1} done")
        else:
            print(f"  ✗ Page {page_num + 1} failed")
    
    doc.close()
    
    # Write to file
    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(all_text))
    
    print(f"✓ Saved to {output_txt_path}")


if __name__ == "__main__":
    # Edit these paths
    PDF_FILE = r"D:\CODE-workingBuild\uniAI\source_code\data\year_2\COA\notes\unit1\hand_unit1.pdf"
    OUTPUT_FILE = r"D:\CODE-workingBuild\uniAI\source_code\data\year_2\COA\notes\unit1\hand_unit1.txt"
    
    pdf_to_text(PDF_FILE, OUTPUT_FILE)