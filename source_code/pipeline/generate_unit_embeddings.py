import json
import os
import sys
import pickle

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import config
from pipeline.embeddings.local_embedding import embed

def build_unit_texts():
    keywords_file = os.path.join(ROOT_DIR, "data", "subject_keywords.json")
    with open(keywords_file, "r") as f:
        data = json.load(f)
        
    unit_texts = {}
    
    for subject, entry in data.items():
        if isinstance(entry, list):
            # No unit structure
            continue
            
        units = set()
        for collection, col_val in entry.items():
            if collection == "pyq" or not isinstance(col_val, dict):
                continue
            for u in col_val.keys():
                if u not in ("unknown", "core"):
                    units.add(u)
                    
        for u in units:
            kws = []
            for col, col_val in entry.items():
                if isinstance(col_val, dict) and u in col_val:
                    kws.extend(col_val[u])
            # combine into text blob
            text = " ".join(kws)
            if text:
               unit_texts[f"{subject}_{u}"] = text
               
    return unit_texts

def main():
    print("Building unit texts from keywords...")
    unit_texts = build_unit_texts()
    
    print(f"Generating embeddings for {len(unit_texts)} units...")
    keys = list(unit_texts.keys())
    texts = [unit_texts[k] for k in keys]
    
    vectors = embed(texts)
    
    embeddings = {k: v for k, v in zip(keys, vectors)}
    
    out_path = config.UNIT_EMBEDDINGS_PATH
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    with open(out_path, "wb") as f:
        pickle.dump(embeddings, f)
        
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()
