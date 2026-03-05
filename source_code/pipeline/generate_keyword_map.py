import os
import sys
import json
import re
from collections import Counter
import chromadb
import ollama

# --- Ensure imports work regardless of working directory ---
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import config
import prompts

# -----------------------------------------------------------------
# Paths & constants
# -----------------------------------------------------------------

CHROMA_PATH = config.CHROMA_DB_PATH
MODEL       = config.MODEL_ROUTER
OUTPUT_FILE = os.path.join(ROOT_DIR, "data", "subject_keywords.json")

COLLECTIONS = {
    "notes":    config.CHROMA_COLLECTION_NAME,
    "syllabus": config.CHROMA_SYLLABUS_COLLECTION_NAME,
    "pyq":      config.CHROMA_PYQ_COLLECTION_NAME,
}

STOP_WORDS = {
    "unit", "none", "unknown", "introduction", "overview", "basics",
    "chapter", "section", "topic", "part", "module", "lecture", "notes",
    "concepts", "fundamentals", "course", "subject",
}

MAX_ITEMS_PER_UNIT    = 30
MAX_ITEMS_PER_SUBJECT = 50
MAX_KEYWORD_WORDS     = 5   # reject keywords longer than this many words

# Regex patterns to strip from keyword lists
_UNIT_LABEL_RE = re.compile(r'^unit\s*\d*$')   # "unit 1", "unit", …


# -----------------------------------------------------------------
# LLM output cleaning
# -----------------------------------------------------------------

def clean_llm_output(raw_output: str) -> list[str]:
    """
    Parse LLM output into a clean list of atomic keyword strings.
    - Strips markdown, numbered lists, unit labels, subject metadata
    - Rejects multi-clause phrases (> MAX_KEYWORD_WORDS words)
    """
    cleaned  = raw_output.replace("\n", ",").replace("**", "").replace("*", "")
    cleaned  = re.sub(r'\d+[\.\)]\s*', '', cleaned)
    keywords = [k.strip().lower() for k in cleaned.split(",") if k.strip()]
    keywords = [
        k for k in keywords
        if 3 < len(k) < 60
        and k not in STOP_WORDS
        and not k.isdigit()
        and not _UNIT_LABEL_RE.match(k)
        and len(k.split()) <= MAX_KEYWORD_WORDS   # reject sentence-style keywords
    ]
    return keywords


# -----------------------------------------------------------------
# Core vs unit-specific separation
# -----------------------------------------------------------------

def split_core_and_specific(unit_kws: dict[str, list[str]]) -> dict[str, list[str]]:
    """
    Promote keywords appearing in ≥2 units to a "core" bucket.
    Unit-specific entries have core keywords removed.
    "unknown" unit is excluded from scoring (kept but not promoted to core).
    """
    # Only count non-unknown units toward core promotion
    counter: Counter = Counter()
    for label, kws in unit_kws.items():
        if label != "unknown":
            counter.update(kws)

    core_set = {kw for kw, count in counter.items() if count >= 2}
    core     = sorted(core_set)

    result: dict[str, list[str]] = {"core": core}
    for unit_label, kws in sorted(unit_kws.items()):
        if unit_label == "unknown":
            result["unknown"] = []      # keep key but empty — router gives it weight 0
        else:
            result[unit_label] = [kw for kw in kws if kw not in core_set]

    return result


# -----------------------------------------------------------------
# Checkpoint helpers
# -----------------------------------------------------------------

def load_checkpoint() -> dict:
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                print(f"Checkpoint found — {len(data)} subjects already processed. Resuming...")
                return data
        except (json.JSONDecodeError, IOError):
            print("Checkpoint file found but unreadable. Starting fresh.")
    return {}


def save_checkpoint(final_map: dict):
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(final_map, f, indent=4)


# -----------------------------------------------------------------
# Metadata gathering
# -----------------------------------------------------------------

def fetch_collection(
    client: chromadb.PersistentClient,
    collection_name: str,
    include: list[str],
) -> dict:
    """Return collection.get() result or empty stub on failure."""
    try:
        collection = client.get_collection(collection_name)
        return collection.get(include=include) or {}
    except Exception as e:
        print(f"  [WARN] Could not open collection '{collection_name}': {e}")
        return {}


def collect_notes_syllabus(metadatas: list[dict]) -> dict[str, dict[str, set]]:
    """Group notes metadata as: subject → unit_label → {titles}."""
    grouped: dict[str, dict[str, set]] = {}
    for meta in metadatas:
        subject = meta.get("subject", "").strip()
        if not subject or subject.lower() in ("", "unknown", "none"):
            continue

        raw_unit   = str(meta.get("unit", "")).strip()
        unit_label = re.sub(r'(?i)^unit\s*', '', raw_unit).strip() or "unknown"

        title = meta.get("title", "").strip()
        if title and title.lower() not in ("unknown", "none", ""):
            grouped.setdefault(subject, {}).setdefault(unit_label, set()).add(title)

    return grouped


def collect_syllabus(metadatas: list[dict], documents: list[str]) -> dict[str, dict[str, set]]:
    """
    Group syllabus data as: subject → unit_label → {topic_snippets}.

    Uses the full embedded document text (which contains all topics) rather
    than just the unit_title from metadata. This prevents LLM acronym confusion
    (e.g. UHV = Ultra High Voltage vs Universal Human Values) by giving the
    model rich topic context that clearly signals the correct domain.
    """
    grouped: dict[str, dict[str, set]] = {}
    for meta, doc_text in zip(metadatas, documents):
        subject = meta.get("subject", "").strip()
        if not subject or subject.lower() in ("", "unknown", "none"):
            continue

        raw_unit   = str(meta.get("unit", "")).strip()
        unit_label = re.sub(r'(?i)^unit\s*', '', raw_unit).strip() or "unknown"

        # Extract the Topics section from the embedded document text
        # Format: "Subject: X | Unit: Y | Title: Z | Topics: t1, t2 ...\n\n<full_text>"
        topics_str = ""
        if "Topics:" in doc_text:
            topics_str = doc_text.split("Topics:", 1)[-1].split("\n")[0].strip()
        elif "Title:" in doc_text:
            topics_str = doc_text.split("Title:", 1)[-1].split("\n")[0].strip()

        # Fall back to the unit title from metadata
        if not topics_str:
            topics_str = meta.get("title", "").strip()

        # Split comma-separated topics and add each as a seed item
        for topic in topics_str.split(","):
            topic = topic.strip()
            if topic and len(topic) > 5 and topic.lower() not in ("unknown", "none"):
                grouped.setdefault(subject, {}).setdefault(unit_label, set()).add(topic)

    return grouped


def collect_pyq(metadatas: list[dict], documents: list[str]) -> dict[str, set]:
    """
    Group PYQ by subject, using actual question text (from ChromaDB documents field)
    as the semantic seed — NOT unit labels, which cause LLM to hallucinate wrong subjects.
    """
    grouped: dict[str, set] = {}
    for meta, doc_text in zip(metadatas, documents):
        subject = meta.get("subject", "").strip()
        if not subject or subject.lower() in ("", "unknown", "none"):
            continue

        # Extract the "Question:" block from the embedded doc text
        # doc_text format: "Subject: X | Unit: Y\n\nQuestion:\n<question text>"
        q_text = ""
        if "Question:" in doc_text:
            q_text = doc_text.split("Question:", 1)[-1].strip()
        elif doc_text.strip():
            q_text = doc_text.strip()

        # Only add meaningful question snippets (first 80 chars as a title seed)
        snippet = q_text[:80].strip()
        if snippet and len(snippet) > 10:
            grouped.setdefault(subject, set()).add(snippet)

    return grouped


# -----------------------------------------------------------------
# LLM keyword extraction
# -----------------------------------------------------------------

def extract_keywords_for_unit(
    ollama_client: ollama.Client,
    subject: str,
    items: set,
    unit: str | None,
    max_items: int,
) -> list[str]:
    """Extract keywords for one unit (or full subject if unit=None)."""
    capped    = list(items)[:max_items]
    items_str = ", ".join(capped)
    prompt    = prompts.keyword_extraction(subject=subject, items_list=items_str, unit=unit)

    try:
        response = ollama_client.chat(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            think=False,
            options={"num_predict": 150, "temperature": 0.1},
        )
        return clean_llm_output(response.message.content.strip())
    except Exception as e:
        print(f"    [ERROR] LLM failed: {e} — using raw items as fallback")
        return clean_llm_output(", ".join(str(i) for i in capped))


def dedupe(lst: list[str]) -> list[str]:
    return list(dict.fromkeys(lst))


# -----------------------------------------------------------------
# Main
# -----------------------------------------------------------------

def generate_keyword_map():
    print(f"Connecting to ChromaDB at {CHROMA_PATH}...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Fetch metadata AND documents for syllabus and PYQ (to get actual topic/question text)
    print("\nFetching metadata from all collections...")
    notes_result    = fetch_collection(client, COLLECTIONS["notes"],    include=["metadatas"])
    syllabus_result = fetch_collection(client, COLLECTIONS["syllabus"], include=["metadatas", "documents"])
    pyq_result      = fetch_collection(client, COLLECTIONS["pyq"],      include=["metadatas", "documents"])

    notes_meta    = notes_result.get("metadatas") or []
    syllabus_meta = syllabus_result.get("metadatas") or []
    syllabus_docs = syllabus_result.get("documents") or []
    pyq_meta      = pyq_result.get("metadatas") or []
    pyq_docs      = pyq_result.get("documents") or []

    print(f"  notes    : {len(notes_meta)} documents")
    print(f"  syllabus : {len(syllabus_meta)} documents")
    print(f"  pyq      : {len(pyq_meta)} documents")

    notes_grouped    = collect_notes_syllabus(notes_meta)
    syllabus_grouped = collect_syllabus(syllabus_meta, syllabus_docs)
    pyq_grouped      = collect_pyq(pyq_meta, pyq_docs)

    all_subjects = sorted(
        set(notes_grouped) | set(syllabus_grouped) | set(pyq_grouped)
    )
    print(f"\nFound {len(all_subjects)} unique subjects across all collections.")

    ollama_client = ollama.Client(host=config.OLLAMA_LOCAL_URL, timeout=90)
    final_map     = load_checkpoint()

    for subject in all_subjects:
        if subject in final_map:
            print(f"Skipping '{subject}' (already in checkpoint).")
            continue

        print(f"\n── Subject: {subject} ──────────────────────────────────")
        subject_entry: dict = {}

        # ── Notes: per-unit → core split ─────────────────────────
        if subject in notes_grouped:
            units = notes_grouped[subject]
            print(f"  notes: {len(units)} unit(s)")
            raw_unit_kws: dict[str, list[str]] = {}
            for unit_label, titles in sorted(units.items()):
                print(f"    unit {unit_label}: {len(titles)} titles")
                kws = extract_keywords_for_unit(
                    ollama_client, subject, titles, unit=unit_label, max_items=MAX_ITEMS_PER_UNIT
                )
                raw_unit_kws[unit_label] = dedupe(kws)
                print(f"      → {len(raw_unit_kws[unit_label])}: {raw_unit_kws[unit_label][:4]}...")
            subject_entry["notes"] = split_core_and_specific(raw_unit_kws)
            print(f"    core ({len(subject_entry['notes']['core'])}): {subject_entry['notes']['core'][:5]}")

        # ── Syllabus: per-unit → core split ──────────────────────
        if subject in syllabus_grouped:
            units = syllabus_grouped[subject]
            print(f"  syllabus: {len(units)} unit(s)")
            raw_unit_kws = {}
            for unit_label, titles in sorted(units.items()):
                print(f"    unit {unit_label}: {len(titles)} titles")
                kws = extract_keywords_for_unit(
                    ollama_client, subject, titles, unit=unit_label, max_items=MAX_ITEMS_PER_UNIT
                )
                raw_unit_kws[unit_label] = dedupe(kws)
                print(f"      → {len(raw_unit_kws[unit_label])}: {raw_unit_kws[unit_label][:4]}...")
            subject_entry["syllabus"] = split_core_and_specific(raw_unit_kws)
            print(f"    core ({len(subject_entry['syllabus']['core'])}): {subject_entry['syllabus']['core'][:5]}")

        # ── PYQ: flat subject-level keywords ─────────────────────
        if subject in pyq_grouped:
            items = pyq_grouped[subject]
            print(f"  pyq: {len(items)} question snippets")
            kws = extract_keywords_for_unit(
                ollama_client, subject, items, unit=None, max_items=MAX_ITEMS_PER_SUBJECT
            )
            subject_entry["pyq"] = dedupe(kws)
            print(f"    → {len(subject_entry['pyq'])}: {subject_entry['pyq'][:4]}...")

        final_map[subject] = subject_entry
        save_checkpoint(final_map)

    print(f"\n✅ Keyword map complete. {len(final_map)} subjects saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    generate_keyword_map()