import os
import shutil
import sys
from pathlib import Path

# --- Ensure imports work regardless of working directory ---
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

import config

def cleanup_data():
    print("--- 🧹 Starting Cleanup Protocol ---")
    
    # 1. Delete ChromaDB
    chroma_path = Path(config.CHROMA_DB_PATH)
    if chroma_path.exists():
        print(f"🗑️ Deleting ChromaDB at: {chroma_path}")
        try:
            shutil.rmtree(chroma_path)
            print("   ✅ ChromaDB deleted.")
        except Exception as e:
            print(f"   ❌ Failed to delete ChromaDB: {e}")
    else:
        print(f"ℹ️ ChromaDB not found at: {chroma_path}")

    # 2. Delete non-PDF files in Data Directory
    data_path = Path(config.BASE_DATA_DIR)
    if not data_path.exists():
        print(f"❌ Data directory not found at: {data_path}")
        return

    print(f"📂 Scanning data directory: {data_path}")
    deleted_count = 0
    kept_count = 0

    for root, dirs, files in os.walk(data_path):
        for file in files:
            file_path = Path(root) / file
            
            # Check if file extension is NOT .pdf (case insensitive)
            if file_path.suffix.lower() != ".pdf":
                try:
                    os.remove(file_path)
                    print(f"   🗑️ Deleted: {file}")
                    deleted_count += 1
                except Exception as e:
                    print(f"   ❌ Failed to delete {file}: {e}")
            else:
                kept_count += 1

    print("-" * 30)
    print(f"✅ Cleanup Complete.")
    print(f"   - Files Deleted: {deleted_count}")
    print(f"   - PDFs Kept: {kept_count}")
    print("-" * 30)

if __name__ == "__main__":
    if "--force" in sys.argv:
        cleanup_data()
    else:
        confirmation = input("⚠️ This will delete the database and ALL non-PDF files in the data directory. Are you sure? (y/n): ")
        if confirmation.lower() == 'y':
            cleanup_data()
        else:
            print("❌ Operation cancelled.")
