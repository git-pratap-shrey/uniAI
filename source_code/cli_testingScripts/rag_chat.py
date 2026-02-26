import os, sys
import json
import chromadb
import ollama
import re

# --- Ensure imports work regardless of working directory ---
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

import config
import prompts

from pipeline.embeddings.local_embedding import embed
from pipeline.retrieval_utils import retrieve_with_threshold

CHROMA_PATH = config.CHROMA_DB_PATH
MODEL = config.MODEL_CHAT  # main model for generation
ROUTER_MODEL = config.MODEL_ROUTER # fast model for routing
KEYWORDS_FILE = os.path.join(ROOT_DIR, "data", "subject_keywords.json")

# Load Keyword Map
SUBJECT_KEYWORD_MAP = {}
if os.path.exists(KEYWORDS_FILE):
    with open(KEYWORDS_FILE, "r") as f:
        SUBJECT_KEYWORD_MAP = json.load(f)
else:
    print(f"Warning: Keyword map not found at {KEYWORDS_FILE}. Run generate_keyword_map.py first.")

# Initialize DB + Collection
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_collection(config.CHROMA_COLLECTION_NAME)
try:
    pyq_collection = client.get_collection(config.CHROMA_PYQ_COLLECTION_NAME)
except chromadb.errors.NotFoundError:
    print(f"Warning: PYQ Collection '{config.CHROMA_PYQ_COLLECTION_NAME}' not found.")
    pyq_collection = None


def retrieve(query, active_subject=None, n_initial=5, similarity_threshold=0.3):
    """Fetch top chunks using the configured embeddings, filtered by cosine similarity threshold and subject metadata."""
    metadata_filter = {"subject": active_subject} if active_subject else None
    
    return retrieve_with_threshold(
        collection=collection,
        query=query,
        n_initial=n_initial,
        similarity_threshold=similarity_threshold,
        metadata_filter=metadata_filter
    )


def format_context(results):
    """Convert chunks into the prompt-friendly format."""
    ctx = ""
    for i in range(len(results["documents"][0])):
        ctx += f"\n### Source {i+1}:\n{results['documents'][0][i]}\n"
    return ctx

def detect_subject(query):
    """
    Score the query against keywords. 
    If unambiguous win -> return subject.
    If ambiguous -> prompt LLM.
    """
    if not SUBJECT_KEYWORD_MAP:
        return None

    query_lower = query.lower()
    scores = {subject: 0 for subject in SUBJECT_KEYWORD_MAP.keys()}
    
    for subject, keywords in SUBJECT_KEYWORD_MAP.items():
        for kw in keywords:
            if kw in query_lower:
                scores[subject] += 1
                
    # Find max score
    max_score = max(scores.values())
    
    if max_score > 0:
        # Check if there's a tie for the top score
        top_subjects = [s for s, score in scores.items() if score == max_score]
        if len(top_subjects) == 1:
            return top_subjects[0] # Clear winner!
            
    # Fallback to LLM if ambiguous (0 score or tied)
    print("   [Routing] Ambiguous query. Asking LLM for subject classification...", end="", flush=True)
    subjects_list = ", ".join(SUBJECT_KEYWORD_MAP.keys())
    prompt = prompts.subject_router(query=query, subjects_list=subjects_list)
    client = ollama.Client(host=config.OLLAMA_LOCAL_URL)
    response = client.chat(model=ROUTER_MODEL, messages=[{"role": "user", "content": prompt}])
    llm_choice = response["message"]["content"].strip()
    print(f"\r   [Routing] Classified as: {llm_choice}")
    
    for valid_subject in SUBJECT_KEYWORD_MAP.keys():
        if valid_subject.lower() in llm_choice.lower():
            return valid_subject
            
    return None

def answer(query, active_subject, conversation_history=""):
    """Run full RAG pipeline: retrieve → prompt LLM → return answer."""
    results = retrieve(query, active_subject)
    context = format_context(results)

    # 1. Detect Unit
    match = re.search(r"unit\s*(\d+)", query.lower())
    predicted_unit = match.group(1) if match else None

    # 2. Retrieve PYQs
    pyq_context = ""
    pyq_count = 0
    filtered_pyqs = []
    
    if pyq_collection:
        where_clause = {}
        if active_subject:
            where_clause["subject"] = active_subject
        if predicted_unit:
            where_clause["unit"] = str(predicted_unit)
        
        if len(where_clause) > 1:
            where_clause = {"$and": [{"subject": active_subject}, {"unit": str(predicted_unit)}]}
        elif not where_clause:
            where_clause = None

        pyq_results = pyq_collection.query(
            query_texts=[query],
            where=where_clause,
            n_results=5
        )

        # 3. Apply Threshold
        SIM_THRESHOLD = 0.72
        
        if pyq_results and pyq_results.get("distances") and pyq_results["distances"][0]:
            for i, dist in enumerate(pyq_results["distances"][0]):
                similarity = 1.0 - dist
                if similarity >= SIM_THRESHOLD:
                    filtered_pyqs.append({
                        "text": pyq_results["documents"][0][i],
                        "metadata": pyq_results["metadatas"][0][i]
                    })
        
        # Limit to top 2-3
        filtered_pyqs = filtered_pyqs[:3]
        pyq_count = len(filtered_pyqs)

        # 4. Build PYQ Context Block
        if pyq_count > 0:
            lines = []
            for p in filtered_pyqs:
                m = p["metadata"]
                line = f"({m.get('year', 'Unknown Year')} {m.get('exam_type', 'Unknown Exam')} – {m.get('marks', '?')}M) {p['text']}"
                lines.append(line)
            pyq_context = "\n".join(lines)

    prompt = prompts.rag_answer(
        query=query, 
        context=context, 
        conversation_history=conversation_history,
        pyq_context=pyq_context,
        pyq_count=pyq_count
    )

    # Initialize client explicitly with config to avoid default host issues
    client_llm = ollama.Client(host=config.OLLAMA_BASE_URL)

    print("   Thinking...", end="", flush=True)
    # Provide a generous context window (e.g. 8192) to avoid "prompt too long" errors with chunky RAG contexts
    response = client_llm.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"num_ctx": 8192}
    )
    print("\r", end="") # Clear "Thinking..."

    final_answer = response["message"]["content"]
    
    # 6. Append Similar PYQs section
    if pyq_count > 0:
        final_answer += "\n\n📝 **Similar PYQs:**\n"
        for p in filtered_pyqs:
            m = p["metadata"]
            # To ensure standard UI output:
            final_answer += f"- ({m.get('year', 'Unknown Year')} {m.get('exam_type', 'Unknown Exam')} – {m.get('marks', '?')}M) {p['text']}\n"

    return final_answer, results


def chat():
    print("🎓 UniAI RAG Chat — Ask academic questions (type 'exit' to quit)")
    conversation = ""
    session_subject = None

    while True:
        query = input("\n🧠 You: ")
        if query.lower() in ["exit", "quit"]:
            break

        # Subject Routing Logic (Run only if we don't have a subject yet)
        if not session_subject:
            detected = detect_subject(query)
            if detected:
                 session_subject = detected
                 print(f"   [Session] Locked topic to: {session_subject}")
            else:
                 print("   [Session] Unrecognized topic. Searching entire database.")
        
        # Override Subject command
        if query.lower().startswith("/switch"):
            parts = query.split(" ", 1)
            if len(parts) > 1:
                session_subject = parts[1].strip()
                print(f"   [Session] Manually switched topic to: {session_subject}")
                continue
            else:
                session_subject = None
                print("   [Session] Cleared active topic.")
                continue

        response_text, results = answer(query, session_subject, conversation)
        print("\n🤖 AI:", response_text)

        # Show sources
        print("\n📚 Sources Used:")
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]
        
        if not docs:
            print("   None (No documents met the similarity threshold).")
        else:
            for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists)):
                source_file = meta.get('source', 'Unknown')
                page_start = meta.get('page_start', '?')
                # calculate and display similarity for debug/transparency
                similarity = 1.0 - dist 
                print(f"   {i+1}. {source_file} (p.{page_start}) [Similarity: {similarity:.2f}]")

        # maintain short history
        conversation += f"\nUser: {query}\nAI: {response_text}\n"


if __name__ == "__main__":
    chat()
