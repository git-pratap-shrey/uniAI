import fitz
import subprocess
import os

def render_page_to_image(pdf_path, page_num=0, output_path="./image.png", scale=2.0):
    print(f"Loading PDF: {pdf_path}")
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_num)
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
    pix.save(output_path)
    print(f"Saved page {page_num} to {output_path}")
    doc.close()

def run_ollama_glm_ocr(image_path="./image.png"):
    model = "glm-ocr:bf16"
    
    commands = {
        "Text recognition": f"ollama run {model} Text Recognition: {image_path}",
        "Table recognition": f"ollama run {model} Table Recognition: {image_path}",
        "Figure recognition": f"ollama run {model} Figure Recognition: {image_path}"
    }
    
    for task_name, cmd in commands.items():
        print(f"\n{'='*40}")
        print(f"--- Running {task_name} ---")
        print(f"Command: {cmd}")
        print(f"{'='*40}")
        
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        print("--- Output ---")
        print(result.stdout)
        
        if result.stderr:
            print("--- Stderr ---")
            print(result.stderr)
            
if __name__ == "__main__":
    PDF_PATH = "/home/anon/PROJECTS/uniAI/source_code/data/year_2/DIGITAL_ELECTRONICS/notes/unit2/unit2.pdf"
    IMAGE_PATH = "./image.png"
    
    if not os.path.exists(PDF_PATH):
        print(f"Error: Hardcoded PDF not found at {PDF_PATH}")
    else:
        # Convert third page to image
        render_page_to_image(PDF_PATH, page_num=2, output_path=IMAGE_PATH)
        
        # Run hardcoded extractions
        run_ollama_glm_ocr(IMAGE_PATH)
