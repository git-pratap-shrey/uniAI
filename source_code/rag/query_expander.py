import json
import os
import re

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ALIASES_FILE = os.path.join(ROOT_DIR, "data", "subject_aliases.json")

# Mapping of specific topics to related syllabus terms
TOPIC_EXPANSION = {
    "tabular method": ["quine mccluskey", "boolean minimization"],
    "k map": ["karnaugh map"],
    "k-map": ["karnaugh map"],
    "phishing": ["social engineering attack"],
    "dos": ["denial of service"],
    "ddos": ["distributed denial of service"]
}

_subject_aliases: dict = {}

if os.path.exists(ALIASES_FILE):
    try:
        with open(ALIASES_FILE, "r", encoding="utf-8") as f:
            _subject_aliases = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"[query_expander] WARNING: Could not load aliases map: {e}")
else:
    print(f"[query_expander] WARNING: Aliases map not found at {ALIASES_FILE}")

def expand_query(user_query: str) -> str:
    """Expand the user query with subject aliases and syllabus keywords."""
    normalized_query = re.sub(r'\s+', ' ', user_query.lower()).strip()
    
    expansions = set()
    
    # 1. Alias Expansion
    for subject, aliases in _subject_aliases.items():
        for alias in aliases:
            # Check with word boundaries to avoid partial matches (e.g. "de" in "under")
            if re.search(rf'\b{re.escape(alias)}\b', normalized_query):
                # Append canonical subject parts
                expansions.add(subject.replace("_", " ").lower())
                break
    
    # 2. Syllabus Keyword Expansion
    for topic, terms in TOPIC_EXPANSION.items():
        if re.search(rf'\b{re.escape(topic)}\b', normalized_query):
            expansions.update(terms)
                
    # Combine original query with expansions
    if expansions:
        expanded_parts = list(expansions)
        return f"{normalized_query} " + " ".join(expanded_parts)
    
    return normalized_query

