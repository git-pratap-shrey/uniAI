import os
import json

from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

# --- Ensure imports work regardless of working directory ---
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
uni_ai_root = os.path.abspath(os.path.join(current_dir, "../.."))
if uni_ai_root not in sys.path:
    sys.path.insert(0, uni_ai_root)

from source_code.config.main import CONFIG
from source_code.rag.rag_pipeline import answer_query
from source_code.rag.search import collection_exists
from source_code.rag.router import list_subjects


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

        MAX_QUERY_LENGTH = 1000
        if len(query) > MAX_QUERY_LENGTH:
            return JsonResponse({
                "answer": f"Your question is too long. Please keep it under {MAX_QUERY_LENGTH} characters."
            })

        history = data.get("history", [])
        session_subject = data.get("subject", None)

        print(f"ROUTING => Provided Subject: {session_subject}")

        result = answer_query(
            query=query,
            history=history,
            session_subject=session_subject,
        )

        # Build frontend-compatible sources (mirrors CLI _print_answer)
        sources = []
        for chunk in result.get("chunks", []):
            meta = chunk.get("metadata", {})
            src = meta.get("source", "unknown")
            page = meta.get("page_start", "?")
            unit = meta.get("unit", "?")
            score = chunk.get("final_score", chunk.get("similarity", 0))
            sources.append({
                "source": src,
                "page": page,
                "unit": unit,
                "score": round(score, 2),
            })

        return JsonResponse({
            "query": query,
            "answer": result["answer"],
            "mode": result["mode"],
            "subject": result.get("subject"),
            "unit": result.get("unit"),
            "expanded_query": result.get("expanded_query", query),
            "sources": sources,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JsonResponse({"error": str(e)}, status=500)


@require_http_methods(["GET"])
def subjects_view(request):
    """Return known subjects — mirrors the CLI /subjects command."""
    subjects = list_subjects()
    return JsonResponse({"subjects": subjects})


@require_http_methods(["GET"])
def health_view(request):
    try:
        status = "healthy" if collection_exists("notes") else "unhealthy: Notes collection not found."
    except Exception as e:
        status = f"unhealthy: {str(e)}"

    return JsonResponse({
        "status": status,
        "model": CONFIG["model"].get("model", "unknown"),
        "chroma_path": str(CONFIG["paths"].get("chroma", "unknown")),
    })