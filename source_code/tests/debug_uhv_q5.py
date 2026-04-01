"""
Debug script for investigating UHV Q5 cross-encoder scoring issue.

This script:
1. Runs the full RAG pipeline for UHV Q5
2. Logs the query, retrieved chunks, and cross-encoder scores
3. Helps identify why the cross-encoder returns a score of 0
"""

import os
import sys

# Add project root to sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from source_code.rag.rag_pipeline import answer_query
from source_code.rag.hybrid_router import route
from source_code.rag.search import retrieve_notes, retrieve_syllabus
from source_code.rag.cross_encoder import rerank_cross_encoder
from source_code.rag.query_expander import expand_query
from source_code.config import CONFIG

# UHV Q5 from questions.txt
UHV_Q5 = "Define 'Prosperity'. How is it different from 'Wealth'?"

print("=" * 80)
print("UHV Q5 Cross-Encoder Investigation")
print("=" * 80)
print(f"\nQuery: {UHV_Q5}\n")

# Step 1: Test routing
print("Step 1: Testing routing...")
route_result = route(UHV_Q5)
print(f"  Subject: {route_result.subject}")
print(f"  Unit: {route_result.unit}")
print(f"  Method: {route_result.method}")

# Step 2: Expand query
print("\nStep 2: Expanding query...")
expanded_query = expand_query(UHV_Q5)
print(f"  Expanded: {expanded_query}")

# Step 3: Retrieve chunks
print("\nStep 3: Retrieving chunks...")
note_chunks = retrieve_notes(
    expanded_query,
    subject=route_result.subject,
    unit=route_result.unit,
    k=CONFIG["rag"]["notes_k"]
)
print(f"  Note chunks retrieved: {len(note_chunks)}")

syllabus_chunks = retrieve_syllabus(
    expanded_query,
    subject=route_result.subject,
    unit=route_result.unit,
    k=CONFIG["rag"]["syllabus_k"]
)
print(f"  Syllabus chunks retrieved: {len(syllabus_chunks)}")

all_chunks = note_chunks + syllabus_chunks
print(f"  Total chunks: {len(all_chunks)}")

# Step 4: Examine chunk content
print("\nStep 4: Examining chunk content...")
if all_chunks:
    for i, chunk in enumerate(all_chunks[:3]):  # Show first 3
        print(f"\n  Chunk {i+1}:")
        print(f"    Source: {chunk.get('source', 'N/A')}")
        print(f"    Similarity: {chunk.get('similarity', 0):.4f}")
        text = chunk.get('text', '')
        print(f"    Text length: {len(text)} chars")
        print(f"    Text preview: {text[:200]}...")
        print(f"    Text is empty: {not text or text.strip() == ''}")
else:
    print("  WARNING: No chunks retrieved!")

# Step 5: Test cross-encoder
print("\nStep 5: Testing cross-encoder...")
if all_chunks:
    ranked = rerank_cross_encoder(
        expanded_query,
        all_chunks,
        top_n=CONFIG["rag"]["cross_encoder"]["pipeline_top_n"],
        candidates=CONFIG["rag"]["cross_encoder"]["candidates"],
    )
    
    print(f"  Ranked chunks: {len(ranked)}")
    if ranked:
        print(f"\n  Top ranked chunks:")
        for i, chunk in enumerate(ranked):
            print(f"\n    Rank {i+1}:")
            print(f"      Source: {chunk.get('source', 'N/A')}")
            print(f"      Final score: {chunk.get('final_score', 0):.4f}")
            print(f"      Original similarity: {chunk.get('similarity', 0):.4f}")
            text = chunk.get('text', '')
            print(f"      Text preview: {text[:150]}...")
        
        # Check if top score is 0
        top_score = ranked[0].get('final_score', 0)
        if top_score == 0:
            print("\n  ⚠️  BUG CONFIRMED: Top cross-encoder score is 0!")
        else:
            print(f"\n  ✓ Top cross-encoder score is non-zero: {top_score:.4f}")
    else:
        print("  WARNING: No ranked chunks returned!")
else:
    print("  Skipping cross-encoder test (no chunks)")

# Step 6: Test full pipeline
print("\nStep 6: Testing full RAG pipeline...")
result = answer_query(UHV_Q5)
print(f"  Subject: {result['subject']}")
print(f"  Unit: {result['unit']}")
print(f"  Mode: {result['mode']}")
print(f"  Chunks used: {len(result['chunks'])}")
if result['chunks']:
    print(f"  Top chunk score: {result['chunks'][0].get('final_score', 0):.4f}")
print(f"\n  Answer preview: {result['answer'][:200]}...")

print("\n" + "=" * 80)
print("Investigation complete!")
print("=" * 80)
