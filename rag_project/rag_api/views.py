import os
import re
import json

from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

import ollama
import chromadb

# --- Ensure imports work regardless of working directory ---
import sys
# Add parent directory of 'rag_project' (which is 'uniAI') to path to find 'source_code'
# Actual structure: .../uniAI/rag_project/rag_api/views.py
# We want .../uniAI to be in path so we can do 'from source_code import config'
current_dir = os.path.dirname(os.path.abspath(__file__))
# current: rag_api
# parent: rag_project
# grandparent: uniAI
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir))) 
# Wait, let's just add the uniAI root.
# The user's root is d:\CODE-workingBuild\uniAI
# And source_code is d:\CODE-workingBuild\uniAI\source_code
# rag_project is d:\CODE-workingBuild\uniAI\rag_project

# Best bet:
uni_ai_root = os.path.abspath(os.path.join(current_dir, "../../..")) 
if uni_ai_root not in sys.path:
    sys.path.append(uni_ai_root)

# Now likely we can import source_code? 
# Actually if we are in uniAI, we can import source_code.config
try:
    from source_code import config
    from source_code.pipeline.embeddings.local_mxbai import embed
except ImportError:
    # Fallback if running relative to uniAI root directly
    import config
    from pipeline.embeddings.local_mxbai import embed

# ------------------------------------------------------------------
# CONFIG & INITIALIZATION
# ------------------------------------------------------------------

# Use Config
CHROMA_PATH = config.CHROMA_DB_PATH
COLLECTION_NAME = config.CHROMA_COLLECTION_NAME
MODEL_CHAT = config.MODEL_CHAT

# ChromaDB
client = chromadb.PersistentClient(path=CHROMA_PATH)
_collection = None


def get_collection():
    global _collection
    if _collection is None:
        try:
            _collection = client.get_collection(COLLECTION_NAME)
        except chromadb.errors.NotFoundError:
            raise RuntimeError(
                f"Collection '{COLLECTION_NAME}' not found. "
                f"Run 'python source_code/ingest_multimodal.py' first."
            )
    return _collection


# ------------------------------------------------------------------
# UI VIEW
# ------------------------------------------------------------------

def chat_view(request):
    return render(request, "chat.html")


# ------------------------------------------------------------------
# MEMORY + INTENT HELPERS
# ------------------------------------------------------------------

FOLLOWUP_TRIGGERS = [
    "repeat",
    "again",
    "previous",
    "earlier",
    "summarize",
]


def is_followup(query: str) -> bool:
    q = query.strip().lower()
    words = q.split()
    if not words:
        return False
    if any(q.startswith(t) for t in FOLLOWUP_TRIGGERS):
        return True
    if len(words) <= 4 and any(t in q for t in FOLLOWUP_TRIGGERS):
        return True
    return False


MAX_HISTORY_TURNS = 4  # user+assistant pairs


def trim_history(history: list[dict]) -> list[dict]:
    if not history:
        return []
    return history[-MAX_HISTORY_TURNS * 2:]


# ------------------------------------------------------------------
# RETRIEVAL HELPERS
# ------------------------------------------------------------------

def detect_unit_query(query: str) -> str | None:
    match = re.search(r"unit\s*(\d+)", query.lower())
    return f"unit{match.group(1)}" if match else None




def retrieve_context(query: str, mode: str = "syllabus", n_results: int = 5) -> list[dict]:
    collection = get_collection()
    unit = detect_unit_query(query)

    filters = []

    if unit:
        filters.append({"unit": unit})

    # Adjust filter for multimodal if needed
    # START CHANGE: handling metadata flexibility
    # if mode == "syllabus":
    #     filters.append({"category": {"$in": ["notes", "pyq"]}})
    # END CHANGE

    where_clause = None
    if filters:
        if len(filters) == 1:
            where_clause = filters[0]
        else:
            where_clause = {"$and": filters}

    query_embedding = embed([query])[0]

    # Increase N for multimodal as we want to find the best page visuals
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3, # Fewer results because each result is now a CHUNK (5 pages)
        where=where_clause,
    )

    if not results or not results.get("documents"):
        return []

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    return [
        {
            "text": doc,  # Full OCR text + metadata
            "source": meta.get("source", "unknown"),
            "unit": meta.get("unit", "unknown"),
            "title": meta.get("title", "unknown"),
        }
        for doc, meta in zip(documents, metadatas)
    ]


# ------------------------------------------------------------------
# GENERATION
# ------------------------------------------------------------------

def generate_answer(query: str, contexts: list[dict], mode: str, history: list[dict] | None = None) -> str:
    history = history or []

    if mode == "syllabus":
        system_prompt = """
            You are uniAI, a syllabus-aware exam assistant.

            You will be given OCR-extracted text from course notes along with a user question.

            Rules:
            - Answer from the provided notes or previous conversation.
            - Use definitions and exam keywords from the notes.
            - Write in a "what to write in exam" tone.
            - If the answer spans multiple chunks, synthesize them.
            - Use clear headings and structure.
            - If the question refers to a previous explanation, repeat or rephrase it.
            - If something is outside the provided notes, say so and answer it based on your knowledge.
        """
    else:
        system_prompt = """
            [GENERIC AI TUTOR MODE]

            This question is outside the syllabus.
            You may use general knowledge.
            Mention clearly that this is not syllabus-bound.
        """

    # Build Context
    memory_block = ""
    if history:
        memory_block = "\nPrevious conversation:\n"
        for h in history:
            role = h.get('role', 'user').upper()
            content = h.get('content', '')
            memory_block += f"{role}: {content}\n"

    context_text_block = ""
    if contexts:
        context_text_block = "\nRelevant notes (OCR-extracted text):\n"
        for c in contexts:
            context_text_block += f"\n[Source: {c['source']} | {c['unit']} | {c.get('title', '')}]\n"
            context_text_block += f"{c['text']}\n"

    # Construct the final prompt
    full_prompt = f"""
{system_prompt}

{memory_block}

{context_text_block}

User question:
{query}
"""

    try:
        # Check against config to see if we use Gemini or Ollama
        if MODEL_CHAT.startswith("gemini"):
            import google.generativeai as genai
            
            api_key = config.GEMINI_API_KEY
            if not api_key:
                return "⚠ Configuration Error: Gemini Model selected but GEMINI_API_KEY is missing."
            
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(MODEL_CHAT)
            
            response = model.generate_content(full_prompt)
            if not response or not response.text:
                 return "⚠ I couldn't generate a response (Empty from Gemini)."
            return response.text

        else:
            # Default to Ollama
            # Initialize client explicitly with config to avoid default host issues
            client = ollama.Client(host=config.OLLAMA_BASE_URL)
            
            response = client.chat(
                model=MODEL_CHAT,
                messages=[{"role": "user", "content": full_prompt}]
            )
            return response["message"]["content"]

    except Exception as e:
        return f"Error generating answer: {e}"



# ------------------------------------------------------------------
# API VIEWS
# ------------------------------------------------------------------

@csrf_exempt  # DEV ONLY
@require_http_methods(["POST"])
def query_view(request):
    try:
        data = json.loads(request.body)
        query = data.get("query", "").strip()

        print("\n--- NEW REQUEST ---")
        print("QUERY:", query)



        if not query:
            return JsonResponse({"answer": "Please enter a question."})

        history = trim_history(data.get("history", []))

        GENERIC_TRIGGERS = [
            "explain in detail",
            "implementation",
            "code",
            "algorithm",
            "beyond syllabus",
            "why does",
            "how does",
        ]

        q_lower = query.lower()
        mode = "generic" if any(t in q_lower for t in GENERIC_TRIGGERS) else "syllabus"

        followup = is_followup(query)

        if followup and history:
            contexts = []
        else:
            contexts = retrieve_context(query, mode=mode)

        if not contexts and not history:
            return JsonResponse({
                "query": query,
                "answer": (
                    "I couldn't find relevant notes or previous context. "
                    "Try asking like: 'Explain unit 2 file handling'."
                ),
                "mode": mode,
                "sources": [],
            })

        answer = generate_answer(
            query=query,
            contexts=contexts,
            mode=mode,
            history=history
        )

        return JsonResponse({
            "query": query,
            "answer": answer,
            "mode": mode,
            "sources": [
                {"source": c["source"], "unit": c["unit"]}
                for c in contexts[:3]
            ],
        })

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


@require_http_methods(["GET"])
def health_view(request):
    try:
        get_collection()
        status = "healthy"
    except Exception as e:
        status = f"unhealthy: {str(e)}"

    return JsonResponse({
        "status": status,
        "model": MODEL_CHAT,
        "chroma_path": CHROMA_PATH,
    })
