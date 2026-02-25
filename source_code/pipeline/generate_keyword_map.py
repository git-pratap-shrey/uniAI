import os
import sys
import json
import re
import chromadb
import ollama

# --- Ensure imports work regardless of working directory ---
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import config

CHROMA_PATH = config.CHROMA_DB_PATH
MODEL = config.MODEL_ROUTER
COLLECTION_NAME = config.CHROMA_COLLECTION_NAME
OUTPUT_FILE = os.path.join(ROOT_DIR, "data", "subject_keywords.json")

STOP_WORDS = {
    "unit", "none", "unknown", "introduction", "overview", "basics",
    "chapter", "section", "topic", "part", "module", "lecture", "notes"
}

MAX_ITEMS_PER_SUBJECT = 50  # Cap to avoid context overflow


def clean_llm_output(raw_output: str) -> list[str]:
    """
    Robustly parse LLM output into a clean list of keyword strings.
    Handles numbered lists, markdown, newlines, and extra whitespace.
    """
    # Normalize newlines and markdown artifacts to commas
    cleaned = raw_output.replace("\n", ",").replace("**", "").replace("*", "")

    # Remove numbered list prefixes like "1. " or "1) "
    cleaned = re.sub(r'\d+[\.\)]\s*', '', cleaned)

    # Split on commas, strip whitespace, lowercase
    keywords = [k.strip().lower() for k in cleaned.split(",") if k.strip()]

    # Filter: remove too-short, too-long, and stop-word entries
    keywords = [
        k for k in keywords
        if len(k) > 3
        and len(k) < 60
        and k not in STOP_WORDS
        and not k.isdigit()
    ]

    return keywords


def load_checkpoint() -> dict:
    """Load existing keyword map if available, to allow resuming interrupted runs."""
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
    """Save current state to disk after each subject."""
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(final_map, f, indent=4)


def generate_keyword_map():
    print(f"Connecting to ChromaDB at {CHROMA_PATH}...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception as e:
        print(f"Error getting collection '{COLLECTION_NAME}': {e}")
        return

    # Fetch all metadata
    print("Fetching all document metadata...")
    results = collection.get(include=["metadatas"])

    if not results or not results.get("metadatas"):
        print("No metadata found in the collection.")
        return

    # Group topics/titles by subject
    subject_raw_data: dict[str, set] = {}
    for meta in results["metadatas"]:
        subject = meta.get("subject")

        if not subject or str(subject).strip().lower() in ["", "unknown", "none"]:
            continue

        if subject not in subject_raw_data:
            subject_raw_data[subject] = set()

        title = meta.get("title", "")
        unit = meta.get("unit", "")

        if title and title.lower() not in ["unknown", "none", ""]:
            subject_raw_data[subject].add(title)
        if unit and unit.lower() not in ["unknown", "none", ""]:
            subject_raw_data[subject].add(f"Unit {unit}")

    if not subject_raw_data:
        print("Could not find any subjects in the database metadata.")
        return

    print(f"Found {len(subject_raw_data)} active subjects. Extracting keywords via LLM...")

    ollama_client = ollama.Client(host=config.OLLAMA_LOCAL_URL)

    # Load checkpoint so interrupted runs can resume
    final_map = load_checkpoint()

    for subject, raw_items in subject_raw_data.items():

        # Skip already-processed subjects
        if subject in final_map:
            print(f"Skipping '{subject}' (already in checkpoint).")
            continue

        if not raw_items:
            final_map[subject] = []
            save_checkpoint(final_map)
            continue

        # Cap items to avoid context window overflow
        capped_items = list(raw_items)[:MAX_ITEMS_PER_SUBJECT]
        items_list = ", ".join(capped_items)

        print(f"\nProcessing '{subject}' ({len(raw_items)} items, using up to {MAX_ITEMS_PER_SUBJECT})...")

        prompt = prompts.keyword_extraction(subject=subject, items_list=items_list)

        try:
            response = ollama_client.chat(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}]
            )

            # Access as object attribute (Ollama Python client returns an object, not a dict)
            raw_output = response.message.content.strip()
            keywords = clean_llm_output(raw_output)

            # Append unit names from raw metadata as reliable anchors
            for item in raw_items:
                if str(item).lower().startswith("unit"):
                    keywords.append(str(item).lower())

            # Always include the subject name itself
            keywords.append(subject.lower())

            # Deduplicate while preserving some order (dict trick)
            keywords = list(dict.fromkeys(keywords))

            final_map[subject] = keywords
            print(f"  -> Extracted {len(keywords)} keywords: {keywords[:5]}...")

        except Exception as e:
            print(f"  [ERROR] LLM query failed for '{subject}': {e}")
            print(f"  -> Falling back to raw metadata items.")
            fallback = [str(i).lower() for i in raw_items]
            fallback.append(subject.lower())
            final_map[subject] = list(dict.fromkeys(fallback))

        # Save after every subject so progress is never lost
        save_checkpoint(final_map)

    print(f"\n✅ Keyword map complete. {len(final_map)} subjects saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    generate_keyword_map()