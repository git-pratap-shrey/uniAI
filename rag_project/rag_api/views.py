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
# Actual structure: .../uniAI/rag_project/rag_api/views.py
current_dir = os.path.dirname(os.path.abspath(__file__))
uni_ai_root = os.path.abspath(os.path.join(current_dir, "../../.."))
if uni_ai_root not in sys.path:
    sys.path.append(uni_ai_root)

try:
    from source_code import config
    from source_code.pipeline.embeddings.local_embedding import embed
    from source_code.pipeline.retrieval_utils import retrieve_with_threshold
except ImportError:
    import config
    from pipeline.embeddings.local_embedding import embed
    from pipeline.retrieval_utils import retrieve_with_threshold

# ------------------------------------------------------------------
# CONFIG & INITIALIZATION
# ------------------------------------------------------------------

CHROMA_PATH = config.CHROMA_DB_PATH
COLLECTION_NAME = config.CHROMA_COLLECTION_NAME
MODEL_CHAT = config.MODEL_CHAT
ROUTER_MODEL = config.MODEL_ROUTER

KEYWORDS_FILE = os.path.join(uni_ai_root, "source_code", "data", "subject_keywords.json")

SUBJECT_KEYWORD_MAP = {}
if os.path.exists(KEYWORDS_FILE):
    with open(KEYWORDS_FILE, "r") as f:
        SUBJECT_KEYWORD_MAP = json.load(f)

# FIX: Renamed from `client` → `chroma_client` to prevent shadowing by
# ollama.Client() calls inside detect_subject() and generate_answer().
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
_collection = None


def get_collection():
    global _collection
    if _collection is None:
        try:
            # FIX: Uses renamed chroma_client
            _collection = chroma_client.get_collection(COLLECTION_NAME)
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


def detect_subject(query: str) -> str | None:
    if not SUBJECT_KEYWORD_MAP:
        return None

    query_lower = query.lower()
    scores = {subject: 0 for subject in SUBJECT_KEYWORD_MAP.keys()}

    for subject, keywords in SUBJECT_KEYWORD_MAP.items():
        for kw in keywords:
            if kw in query_lower:
                scores[subject] += 1

    max_score = max(scores.values())

    if max_score > 0:
        top_subjects = [s for s, score in scores.items() if score == max_score]
        if len(top_subjects) == 1:
            return top_subjects[0]

    # Fallback to LLM router
    subjects_list = ", ".join(SUBJECT_KEYWORD_MAP.keys())
    prompt = f"""
You are a routing agent. The user is asking a question about university coursework.
Known Subjects: {subjects_list}

User Query: "{query}"

Which of the Known Subjects is this query about? 
Reply ONLY with the exact Subject name. If it does not match any, reply NONE.
"""
    # FIX: Renamed to ollama_router_client to avoid shadowing chroma_client
    ollama_router_client = ollama.Client(host=config.OLLAMA_LOCAL_URL)
    response = ollama_router_client.chat(model=ROUTER_MODEL, messages=[{"role": "user", "content": prompt}])
    llm_choice = response["message"]["content"].strip()

    for valid_subject in SUBJECT_KEYWORD_MAP.keys():
        if valid_subject.lower() in llm_choice.lower():
            return valid_subject

    return None


def retrieve_context(query: str, active_subject: str = None, n_results: int = 5) -> list[dict]:
    # FIX: Removed unused `mode` parameter.
    # FIX: `n_results` now actually passed through to retrieve_with_threshold.
    collection = get_collection()
    unit = detect_unit_query(query)

    filters = []
    if unit:
        filters.append({"unit": unit})
    if active_subject:
        filters.append({"subject": active_subject})

    where_clause = None
    if filters:
        if len(filters) == 1:
            where_clause = filters[0]
        else:
            where_clause = {"$and": filters}

    results = retrieve_with_threshold(
        collection=collection,
        query=query,
        n_initial=n_results,  # FIX: was hardcoded to 5, now uses the parameter
        similarity_threshold=0.3,
        metadata_filter=where_clause
    )

    if not results or not results.get("documents") or not results["documents"][0]:
        return []

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    return [
        {
            "text": doc,
            "source": meta.get("source", "unknown"),
            "unit": meta.get("unit", "unknown"),
            "title": meta.get("title", "unknown"),
            "page_start": meta.get("page_start", "?"),
        }
        for doc, meta in zip(documents, metadatas)
    ]


# ------------------------------------------------------------------
# GENERATION
# ------------------------------------------------------------------

def generate_answer(query: str, contexts: list[dict], mode: str, history: list[dict] | None = None) -> str:
    history = history or []

    # Build conversation history block
    conversation_history = ""
    if history:
        for h in history:
            role = "User" if h.get('role', 'user').lower() == 'user' else "AI"
            content = h.get('content', '')
            conversation_history += f"{role}: {content}\n"

    # Build context block
    context = ""
    if contexts:
        for i, c in enumerate(contexts):
            context += f"\n### Source {i+1}:\n{c['text']}\n"

    prompt = f"""
You are a university assistant trained on real syllabus, notes, and PYQs.

Answer the user based on the provided context and your own knowledge.
Your goal is to provide answers relevant to the students' academic needs, for exams, assignments, and projects.

If the answer is not in the context, say: "Here is some information about the topic from my knowledge: ".

Conversation so far:
{conversation_history}

User question:
{query}

Relevant context:
{context}

Answer clearly with bullet points and examples. Do NOT mention the context explicitly.
"""

    try:
        ollama_chat_client = ollama.Client(host=config.OLLAMA_BASE_URL)
        response = ollama_chat_client.chat(
            model=MODEL_CHAT,
            messages=[{"role": "user", "content": prompt}],
            options={"num_ctx": 8192}
        )
        return response["message"]["content"]

    except Exception as e:
        return f"Error generating answer: {e}"


# ------------------------------------------------------------------
# API VIEWS
# ------------------------------------------------------------------

# TODO: Remove @csrf_exempt before deploying to production.
# Options: (a) send CSRF token from frontend, or
#          (b) add token auth: check request.headers.get("X-API-Key") == settings.API_SECRET
@csrf_exempt
@require_http_methods(["POST"])
def query_view(request):
    # FIX: JSON decode errors now caught separately and return a clean 400
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON in request body."}, status=400)

    try:
        query = data.get("query", "").strip()

        print("\n--- NEW REQUEST ---")
        print("QUERY:", query)

        if not query:
            return JsonResponse({"answer": "Please enter a question."})

        # FIX: Guard against oversized inputs to prevent prompt bloat / slow LLM calls
        MAX_QUERY_LENGTH = 1000
        if len(query) > MAX_QUERY_LENGTH:
            return JsonResponse({
                "answer": f"Your question is too long. Please keep it under {MAX_QUERY_LENGTH} characters."
            })

        history = trim_history(data.get("history", []))

        # FIX: Removed "how does" and "why does" — these fired on almost every
        # valid syllabus question (e.g. "how does TCP work?") and incorrectly
        # bypassed note retrieval. Remaining triggers reliably signal out-of-syllabus intent.
        GENERIC_TRIGGERS = [
            "explain in detail",
            "implementation",
            "code",
            "algorithm",
            "beyond syllabus",
        ]

        q_lower = query.lower()
        mode = "generic" if any(t in q_lower for t in GENERIC_TRIGGERS) else "syllabus"

        followup = is_followup(query)
        active_subject = detect_subject(query)
        print(f"ROUTING => Active Subject: {active_subject}")

        if followup and history:
            contexts = []
        else:
            # FIX: No longer passes unused `mode` argument to retrieve_context
            contexts = retrieve_context(query, active_subject=active_subject)

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
                {"source": c["source"], "unit": c["unit"], "page_start": c.get("page_start", "?")}
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