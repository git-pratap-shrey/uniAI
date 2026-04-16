"""
test_wiki_retrieval.py
───────────────────────
Compares retrieval quality between:
  1. raw_notes (multimodal_notes)
  2. wiki_pages (multimodal_notes_wiki)

Run:
  python source_code/karpathy_llm-wiki/test_wiki_retrieval.py --query "What is CIA triad?" --subject CYBER_SECURITY
"""

import os
import sys
import argparse
from pathlib import Path

# Ensure project root is in sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from source_code.config.main import CONFIG
from source_code.rag.search import retrieve_notes, _query_collection

def test_retrieval(query: str, subject: str, unit: str = None):
    print(f"\n🔍 Testing Retrieval for: '{query}'")
    print(f"   Subject: {subject}, Unit: {unit or 'All'}\n")

    # 1. Retrieve from Raw Notes
    print("--- RAW NOTES (multimodal_notes) ---")
    raw_results = retrieve_notes(query, subject=subject, unit=unit, k=3)
    if not raw_results:
        print("   No results found.")
    for i, res in enumerate(raw_results, 1):
        text_snippet = res['text'][:200].replace('\n', ' ')
        print(f"   {i}. [{res['similarity']:.4f}] {res['metadata'].get('unit', '?')} | {text_snippet}...")

    # 2. Retrieve from Wiki
    print("\n--- WIKI PAGES (multimodal_notes_wiki) ---")
    # We need to manually query the wiki collection since it's not in the standard search.py aliases
    # But search.py's _query_collection can be used if we temporarily register it or call it directly
    
    # We'll use the internal name from ingest_wiki.py logic
    wiki_collection_name = CONFIG["paths"]["collections"]["notes"] + "_wiki"
    
    # We need to construct the 'where' clause manually or use search.py's helper
    from source_code.rag.search import _build_where, _query_collection
    
    # Note: ingest_wiki uses 'unit' as metadata, but search.py's normalize_unit 
    # might expect numeric. Let's see how ingest_wiki stored it.
    # Ingest wiki stored unit as 'unit1', 'unit2' etc.
    # Search.py's _build_where/normalize_unit would turn 'unit1' -> '1'.
    
    # Let's check how ingest_wiki stored it: unit = rel.parts[0] -> 'unit1'
    # So we should pass the raw unit string if provided, or handle the filtering carefully.
    
    # For this test, let's just use subject filter to be safe.
    where = {"subject": subject.upper()}
    if unit:
        # If user passed 'unit1', keep it. If '1', turn to 'unit1'.
        u = unit.lower()
        if u.isdigit(): u = f"unit{u}"
        where = {"$and": [{"subject": subject.upper()}, {"unit": u}]}

    # Since _query_collection uses _get(alias), and 'wiki' isn't a standard alias,
    # we'll reach into the client directly.
    import chromadb
    from source_code.rag.search import _client, embed
    
    try:
        wiki_coll = _client.get_collection(wiki_collection_name)
        query_vector = embed([query])[0]
        wiki_res = wiki_coll.query(
            query_embeddings=[query_vector],
            n_results=3,
            where=where,
            include=["documents", "metadatas", "distances"]
        )
        
        if wiki_res and wiki_res["documents"]:
            docs = wiki_res["documents"][0]
            metas = wiki_res["metadatas"][0]
            dists = wiki_res["distances"][0]
            
            for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists), 1):
                sim = 1.0 - dist
                topic = meta.get("topic", "Unknown")
                text_snippet = doc[:200].replace('\n', ' ')
                print(f"   {i}. [{sim:.4f}] {meta.get('unit', '?')} | Topic: {topic}")
                print(f"      Content: {text_snippet}...")
        else:
            print("   No wiki results found.")
            
    except Exception as e:
        print(f"   Error querying wiki: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--unit", default=None)
    args = parser.parse_args()
    
    test_retrieval(args.query, args.subject, args.unit)
