import os
import re
import json

from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

# --- Ensure imports work regardless of working directory ---
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
uni_ai_root = os.path.abspath(os.path.join(current_dir, "../../.."))
if uni_ai_root not in sys.path:
    sys.path.append(uni_ai_root)

try:
    from source_code import config
    from source_code.rag.rag_pipeline import answer_query
    from source_code.rag.search import collection_exists
except ImportError:
    import config
    from rag.rag_pipeline import answer_query
    from rag.search import collection_exists


# ------------------------------------------------------------------
# UI VIEW
# ------------------------------------------------------------------

def chat_view(request):
    return render(request, "chat.html")


# ------------------------------------------------------------------
# API VIEWS
# ------------------------------------------------------------------

# TODO: Remove @csrf_exempt before deploying to production.
@csrf_exempt
@require_http_methods(["POST"])
def query_view(request):
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

        # Guard against oversized inputs to prevent prompt bloat / slow LLM calls
        MAX_QUERY_LENGTH = 1000
        if len(query) > MAX_QUERY_LENGTH:
            return JsonResponse({
                "answer": f"Your question is too long. Please keep it under {MAX_QUERY_LENGTH} characters."
            })

        history = data.get("history", [])

        # The frontend isn't currently sending a locked session_subject, 
        # but if it does in the future, we can extract it here.
        session_subject = data.get("subject", None)

        print(f"ROUTING => Provided Subject: {session_subject}")

        # Run the full RAG pipeline
        result = answer_query(
            query=query,
            history=history,
            session_subject=session_subject
        )

        # Build frontend-compatible sources directly from chunks
        sources = [
            {
                "source": chunk.get("metadata", {}).get("source", "unknown"),
                "unit": chunk.get("metadata", {}).get("unit", "?"),
                "page_start": chunk.get("metadata", {}).get("page_start", "?")
            }
            for chunk in result.get("chunks", [])[:3]
        ]

        return JsonResponse({
            "query": query,
            "answer": result["answer"],
            "mode": result["mode"],
            "sources": sources,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JsonResponse({"error": str(e)}, status=500)


@require_http_methods(["GET"])
def health_view(request):
    try:
        # Check if the primary notes collection is accessible
        if collection_exists("notes"):
            status = "healthy"
        else:
            status = "unhealthy: Notes collection not found."
    except Exception as e:
        status = f"unhealthy: {str(e)}"

    return JsonResponse({
        "status": status,
        "model": config.MODEL_CHAT,
        "chroma_path": config.CHROMA_DB_PATH,
    })