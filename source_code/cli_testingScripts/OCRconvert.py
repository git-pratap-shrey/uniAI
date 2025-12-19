import os
import time
import fitz  # PyMuPDF
import ollama

# 1. Configuration
# Use a raw string (r"") for your Windows path
folder_path = r"D:\CODE-workingBuild\uniAI\source_code\data\year_2"
MODEL_NAME = "deepseek-ocr"

def extract_and_save_by_page(pdf_path, txt_path):
    print(f"   > Opening: {os.path.basename(pdf_path)}")
    
    # Open the PDF
    doc = fitz.open(pdf_path)
    
    # Loop through each page
    for page_num in range(len(doc)):
        print(f"     * OCR Page {page_num + 1}/{len(doc)}...", end="", flush=True)
        
        try:
            # 2. Convert Page to Image for Vision model
            page = doc.load_page(page_num)
            # Use 2.0 matrix for clear OCR (standard resolution)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img_data = pix.tobytes("png")
            
            # 3. Request OCR from local Ollama
            # CRITICAL: keep_alive=0 resets memory after every page to prevent crashes
            response = ollama.generate(
                model=MODEL_NAME,
                prompt="Free OCR. remove watermarks and clean up any sentences from bad ocr result.",
                images=[img_data],
                stream=False,
                keep_alive=0  # Forces memory cleanup
            )
            
            # 4. Save incrementally to the text file
            with open(txt_path, "a", encoding="utf-8") as f:
                f.write(f"\n--- PAGE {page_num + 1} ---\n")
                f.write(response.get('response', '') + "\n")
            
            print(" Success.")
            
            # Clear large image objects from Python memory
            pix = None
            img_data = None
            
        except Exception as page_err:
            print(f" FAILED: {page_err}")
            with open(txt_path, "a", encoding="utf-8") as f:
                f.write(f"\n--- PAGE {page_num + 1} ERROR ---\n{page_err}\n")

    doc.close()

def process_all_folders(base_path):
    print(f"--- Recursive Local DeepSeek-OCR Start ---")
    
    # Recursively walk through every subfolder
    for root, dirs, files in os.walk(base_path):
        pdf_files = [f for f in files if f.lower().endswith(".pdf")]
        
        if not pdf_files:
            continue
            
        print(f"\n[Folder] Scanning: {root}")
        
        for filename in pdf_files:
            pdf_file_path = os.path.join(root, filename)
            txt_file_path = os.path.splitext(pdf_file_path)[0] + ".txt"
            
            # Skip files already processed to save time
            if os.path.exists(txt_file_path):
                print(f"   > Skipping (already exists): {filename}")
                continue

            # Start fresh for this specific PDF file
            extract_and_save_by_page(pdf_file_path, txt_file_path)

if __name__ == "__main__":
    process_all_folders(folder_path)
    print("\n--- All local folders processed successfully ---")