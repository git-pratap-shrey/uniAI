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

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set")
if not CHROMA_PATH:
    raise ValueError("CHROMA_PATH environment variable not set")
if not COLLECTION_NAME:
    raise ValueError("COLLECTION_NAME environment variable not set")

# Gemini
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")

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
                f"Run 'python ingest_python.py' first."
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

    if mode == "syllabus":
        filters.append({"category": {"$in": ["notes", "pyq"]}})

    where_clause = None
    if filters:
        if len(filters) == 1:
            where_clause = filters[0]
        else:
            where_clause = {"$and": filters}

    query_embedding = embed([query])[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=20 if unit else n_results,
        where=where_clause,
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



# ------------------------------------------------------------------
# GENERATION
# ------------------------------------------------------------------

def generate_answer(query: str, contexts: list[dict], mode: str, history: list[dict] | None = None) -> str:
    history = history or []

    if mode == "syllabus":
        system_prompt = """
You are uniAI, a syllabus-aware exam assistant.

Rules:
- Answer ONLY from provided notes or previous assistant responses.
- Use definitions and exam keywords.
- Write in a "what to write in exam" tone.
- If the question refers to previous explanation, repeat or rephrase it.
- If something is outside the syllabus, clearly say so.
"""
    else:
        system_prompt = """
[GENERIC AI TUTOR MODE]

This question is outside the syllabus.
You may use general knowledge.
Mention clearly that this is not syllabus-bound.
"""

    memory_block = ""
    if history:
        memory_block = "\nPrevious conversation:\n"
        for h in history:
            memory_block += f"{h['role'].upper()}: {h['content']}\n"

    context_block = ""
    if contexts:
        context_block = "\nSyllabus context:\n"
        for c in contexts:
            context_block += f"[{c['source']} - {c['unit']}]\n{c['text']}\n\n"

    prompt = f"""
{system_prompt}

{memory_block}

{context_block}

User question:
{query}
"""

    try:
        response = model.generate_content(prompt)
        if not response or not response.text:
            return "⚠ I couldn't generate a response. Please try again."
        return response.text
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
        "model": "gemini-2.5-flash",
        "chroma_path": CHROMA_PATH,
    })
