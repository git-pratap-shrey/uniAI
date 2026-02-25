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
    from source_code.pipeline.embeddings.local_embedding import embed
    from source_code.pipeline.retrieval_utils import retrieve_with_threshold
except ImportError:
    # Fallback if running relative to uniAI root directly
    import config
    from pipeline.embeddings.local_embedding import embed
    from pipeline.retrieval_utils import retrieve_with_threshold

# ------------------------------------------------------------------
# CONFIG & INITIALIZATION
# ------------------------------------------------------------------

# Use Config
CHROMA_PATH = config.CHROMA_DB_PATH
COLLECTION_NAME = config.CHROMA_COLLECTION_NAME
MODEL_CHAT = config.MODEL_CHAT
ROUTER_MODEL = config.MODEL_ROUTER

KEYWORDS_FILE = os.path.join(uni_ai_root, "source_code", "data", "subject_keywords.json")

# Load Keyword Map
SUBJECT_KEYWORD_MAP = {}
if os.path.exists(KEYWORDS_FILE):
    with open(KEYWORDS_FILE, "r") as f:
        SUBJECT_KEYWORD_MAP = json.load(f)

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
            
    # Fallback to LLM
    subjects_list = ", ".join(SUBJECT_KEYWORD_MAP.keys())
    prompt = f"""
You are a routing agent. The user is asking a question about university coursework.
Known Subjects: {subjects_list}

User Query: "{query}"

Which of the Known Subjects is this query about? 
Reply ONLY with the exact exact Subject name. If it does not match any, reply NONE.
"""
    client = ollama.Client(host=config.OLLAMA_LOCAL_URL)
    response = client.chat(model=ROUTER_MODEL, messages=[{"role": "user", "content": prompt}])
    llm_choice = response["message"]["content"].strip()
    
    for valid_subject in SUBJECT_KEYWORD_MAP.keys():
        if valid_subject.lower() in llm_choice.lower():
            return valid_subject
            
    return None
def retrieve_context(query: str, active_subject: str = None, mode: str = "syllabus", n_results: int = 5) -> list[dict]:
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

    # Use retrieve_with_threshold from pipeline.retrieval_utils
    results = retrieve_with_threshold(
        collection=collection,
        query=query,
        n_initial=5,
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

    # Build Context
    conversation_history = ""
    if history:
        for h in history:
            role = "User" if h.get('role', 'user').lower() == 'user' else "AI"
            content = h.get('content', '')
            conversation_history += f"{role}: {content}\n"

    context = ""
    if contexts:
        for i, c in enumerate(contexts):
            context += f"\n### Source {i+1}:\n{c['text']}\n"

    # Construct the final prompt
    prompt = f"""
You are a university assistant trained on real syllabus, notes, and PYQs.

Answer the user based on the provided context, add that to your own knowledge, and information available on the internet,

your goal is therefore to provide answers that are relevant to the students' academic needs, for exams, assignments, and projects.

If the answer is not in the context, say: "here is some information about the topic from my knowledge: ".

Conversation so far:
{conversation_history}

User question:
{query}

Relevant context:
{context}

Answer clearly with bullet points and examples. Do NOT mention the context explicitly.
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
            
            response = model.generate_content(prompt)
            if not response or not response.text:
                 return "⚠ I couldn't generate a response (Empty from Gemini)."
            return response.text

        else:
            # Default to Ollama
            # Initialize client explicitly with config to avoid default host issues
            client = ollama.Client(host=config.OLLAMA_BASE_URL)
            
            response = client.chat(
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
        # Attempt subject routing
        active_subject = detect_subject(query)
        print(f"ROUTING => Active Subject: {active_subject}")

        if followup and history:
            contexts = []
        else:
            contexts = retrieve_context(query, active_subject=active_subject, mode=mode)

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
