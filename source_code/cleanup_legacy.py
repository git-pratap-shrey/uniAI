import os
from pathlib import Path

BASE = r"D:\CODE-workingBuild\uniAI\source_code\data\year_2"

def cleanup():
    deleted_count = 0
    print(f"Cleaning legacy artifacts in: {BASE}")

    for root, dirs, files in os.walk(BASE):
        # PROTECT THE NEW DATA: Skip any folder that is a new extraction folder
        if "_extracted" in root:
            continue

        for f in files:
            fpath = Path(root) / f
            
            # 1. Delete all .txt files (legacy OCR output)
            if f.endswith(".txt"):
                try:
                    os.remove(fpath)
                    print(f"Deleted: {f}")
                    deleted_count += 1
                except Exception as e:
                    print(f"Error deleting {f}: {e}")
                    
            # 2. Delete .json files (legacy metadata)
            # Safe to delete because new JSONs are inside "_extracted" folders which we skip above
            elif f.endswith(".json"):
                try:
                    os.remove(fpath)
                    print(f"Deleted: {f}")
                    deleted_count += 1
                except Exception as e:
                    print(f"Error deleting {f}: {e}")

    print(f"--- Cleanup complete. Deleted {deleted_count} files. ---")

if __name__ == "__main__":
    cleanup()
