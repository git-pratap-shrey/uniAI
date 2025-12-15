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


def retrieve_context(query: str, n_results: int = 5) -> list[dict]:
    """
    Retrieve relevant chunks from ChromaDB.
    - Uses metadata filtering for unit-specific queries
    - Uses semantic search otherwise
    """
    collection = get_collection()
    unit = detect_unit_query(query)

    if unit:
        # Metadata filtering - search within unit
        results = collection.query(
            query_embeddings=[embed([query])[0]],  # Embed the query properly
            n_results=20,
            where={"unit": unit},
        )
    else:
        query_embedding = embed([query])
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
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


def generate_answer(query: str, contexts: list[dict]) -> str:
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

    prompt = f"""
You are a helpful AI assistant for a student studying Python programming.

Answer the question using ONLY the provided course notes.
If the context is insufficient, clearly say so.

Context:
{context_block}

Question:
{query}

Provide a clear, well-structured answer. Include code examples if relevant.
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

        if not query:
            return JsonResponse({"error": "No query provided"}, status=400)

        contexts = retrieve_context(query)
        answer = generate_answer(query, contexts)

        return JsonResponse({
            "query": query,
            "answer": answer,
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