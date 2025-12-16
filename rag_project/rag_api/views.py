import os
import re
import json

from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

import google.generativeai as genai
import chromadb

from pipeline.embeddings.local_mxbai import embed  # type: ignore

# ------------------------------------------------------------------
# CONFIG & INITIALIZATION
# ------------------------------------------------------------------

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
CHROMA_PATH = os.getenv("CHROMA_PATH")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

# Validate required env vars
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set")
if not CHROMA_PATH:
    raise ValueError("CHROMA_PATH environment variable not set")
if not COLLECTION_NAME:
    raise ValueError("COLLECTION_NAME environment variable not set")

# Gemini
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")

# ChromaDB - initialize client only, NOT collection (lazy load)
client = chromadb.PersistentClient(path=CHROMA_PATH)
_collection = None  # Will be loaded on first use


def get_collection():
    """Lazy-load the collection on first use."""
    global _collection
    if _collection is None:
        try:
            _collection = client.get_collection(COLLECTION_NAME)
        except chromadb.errors.NotFoundError:
            raise RuntimeError(
                f"Collection '{COLLECTION_NAME}' not found. "
                f"Run 'python ingest_python.py' first to create it."
            )
    return _collection


# ------------------------------------------------------------------
# UI VIEW
# ------------------------------------------------------------------

def chat_view(request):
    """Render chat interface."""
    return render(request, "chat.html")


# ------------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------------

def detect_unit_query(query: str) -> str | None:
    """
    Detect unit references like:
    - 'unit4'
    - 'unit 4'
    """
    match = re.search(r"unit\s*([1-9])", query.lower())
    return f"unit{match.group(1)}" if match else None


def retrieve_context(query: str, mode: str = "syllabus", n_results: int = 5) -> list[dict]:
    """
    Retrieve relevant chunks from ChromaDB.
    - Uses metadata filtering for unit-specific queries
    - Uses semantic search otherwise
    """
    collection = get_collection()
    unit = detect_unit_query(query)

    where_clause = {}

    if unit:
        where_clause["unit"] = unit

    if mode == "syllabus":
        where_clause["type"] = {"$in": ["notes", "pyq"]}

    # In generic mode → no type restriction

    query_embedding = embed([query])[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=20 if unit else n_results,
        where=where_clause if where_clause else None,
    )


    if not results or not results.get("documents"):
        return []

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    return [
        {
            "text": doc,
            "source": meta.get("source", "unknown"),
            "unit": meta.get("unit", "unknown"),
        }
        for doc, meta in zip(documents, metadatas)
    ]


def generate_answer(query: str, contexts: list[dict], mode: str) -> str:
    """
    Generate an answer using Gemini constrained to retrieved context.
    """
    if not contexts:
        return (
            "I couldn't find relevant information in the notes. "
            "Could you rephrase your question?"
        )

    context_block = "\n\n".join(
        f"[Source: {c['source']} - {c['unit']}]\n{c['text']}"
        for c in contexts
    )
    if mode == "syllabus":
        system_prompt = """
    You are a syllabus-aware exam assistant.

    Rules:
    - Answer ONLY from the provided notes and PYQs.
    - Use definitions and exam keywords.
    - Write in a "what to write in exam" tone.
    - Do NOT add extra theory.
    - If something is outside the syllabus, clearly say so.
    """
    else:
        system_prompt = """
    The following question is outside the syllabus.

    You are now acting as a generic AI tutor.
    You may use general knowledge.
    Explain clearly, but mention that this is not syllabus-bound.
    """


    prompt = f"""
                {system_prompt}

                Context:
                {context_block}

                Question:
                {query}
            """


    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating answer: {e}"


# ------------------------------------------------------------------
# API VIEWS
# ------------------------------------------------------------------

@csrf_exempt  # Development only – enable CSRF in production
@require_http_methods(["POST"])
def query_view(request):
    """Main RAG query endpoint."""
    try:
        data = json.loads(request.body)
        query = data.get("query", "").strip()
        
        GENERIC_TRIGGERS = [
            "explain in detail",
            "implementation",
            "code",
            "algorithm",
            "beyond syllabus",
            "why does",
            "how does",
        ]

        mode = "syllabus"
        if any(t in query for t in GENERIC_TRIGGERS):
            mode = "generic"
        
        print("MODE:", mode)

        if not query:
            return JsonResponse({"error": "No query provided"}, status=400)

        contexts = retrieve_context(query, mode=mode)
        answer = generate_answer(query, contexts, mode)

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
    """Basic health check endpoint."""
    try:
        get_collection()  # Verify collection exists
        status = "healthy"
    except Exception as e:
        status = f"unhealthy: {str(e)}"

    return JsonResponse({
        "status": status,
        "model": "gemini-2.5-flash",
        "chroma_path": CHROMA_PATH,
    })