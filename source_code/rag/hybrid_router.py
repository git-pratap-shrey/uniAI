import os
import sys
from dataclasses import dataclass

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import config
import ollama
from prompts import subject_unit_router
from rag.router import detect_subject, _keyword_map
from rag.unit_router import detect_unit
from rag.embedding_router import route as embedding_route

@dataclass
class RouteResult:
    subject: str | None
    unit: str | None
    method: str  # "keyword" | "embedding" | "llm" | "none"

def _llm_classify_subject_unit(query: str) -> RouteResult:
    """Fallback router using LLM to classify both subject and unit."""
    if not _keyword_map:
        return RouteResult(None, None, "none")

    # Build subjects_units_list
    subject_units = []
    for subject, entry in _keyword_map.items():
        if isinstance(entry, list):
            subject_units.append(subject)
            continue
        # Find units
        units = set()
        for col_val in entry.values():
            if isinstance(col_val, dict):
                for u in col_val.keys():
                    if u not in ("unknown", "core"):
                        units.add(u)
        
        if units:
            for u in sorted(units):
                subject_units.append(f"{subject}_{u}")
        else:
            subject_units.append(subject)

    subjects_units_str = ", ".join(subject_units)
    prompt = subject_unit_router(query=query, subjects_units_list=subjects_units_str)

    try:
        client = ollama.Client(host=config.OLLAMA_LOCAL_URL)
        response = client.chat(
            model=config.MODEL_ROUTER,
            messages=[
                {"role": "system", "content": "You are a helpful assistant. You must respond directly without internal reasoning or <think> tags."},
                {"role": "user",   "content": f"{prompt} /no_think"}
            ],
            think=False,
            options={
                "temperature": config.OLLAMA_ROUTER_TEMPERATURE,
                "num_predict": config.OLLAMA_ROUTER_NUM_PREDICT,
            },
        )
        
        llm_choice = response["message"]["content"].strip().upper().replace(" ", "_")
        
        for su in subject_units:
            if su.upper() == llm_choice:
                parts = su.rsplit("_", 1)
                # Ensure it has a unit part and it's something like "3", "4"
                if len(parts) == 2 and parts[1].isdigit():
                    return RouteResult(parts[0], parts[1], "llm")
                else:
                    return RouteResult(su, None, "llm")
                    
    except Exception as e:
        print(f"[hybrid_router] LLM classification failed: {e}")

    return RouteResult(None, None, "none")


def route(query: str, session_subject: str | None = None) -> RouteResult:
    # 1. Explicit unit via regex
    explicit_unit = detect_unit(query)
    
    # 2. Keyword Router (subject level, and optionally unit)
    subj, unit, used_llm = detect_subject(query, debug=True)
    if subj and not used_llm:
        # If user explicitly specified a unit in the query, it overrides the keyword unit
        final_unit = explicit_unit or unit
        # session_subject overrides keyword subject if provided
        final_subj = session_subject or subj
        return RouteResult(final_subj, final_unit, "keyword")
        
    # 3. Embedding Router
    emb_subj, emb_unit, emb_score = embedding_route(query)
    if emb_subj:
        final_unit = explicit_unit or emb_unit
        final_subj = session_subject or emb_subj
        return RouteResult(final_subj, final_unit, "embedding")
        
    # 4. LLM Router
    llm_res = _llm_classify_subject_unit(query)
    if llm_res.subject:
        final_unit = explicit_unit or llm_res.unit
        final_subj = session_subject or llm_res.subject
        return RouteResult(final_subj, final_unit, "llm")
        
    # 5. Fallback
    return RouteResult(session_subject, explicit_unit, "none")
