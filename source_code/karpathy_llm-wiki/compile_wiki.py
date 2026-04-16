"""
compile_wiki.py
───────────────
Reads all unit-specific JSON files for a subject, groups them by unit,
and uses Gemini to compile structured wiki pages per topic.

Output: source_code/data/year_2/<SUBJECT>/wiki/<unit>/<topic>.md

Run:
  python source_code/karpathy_llm-wiki/compile_wiki.py --subject CYBER_SECURITY
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from collections import defaultdict
from dotenv import load_dotenv

# ── path setup ────────────────────────────────────────────────────────────────
# Ensure project root is in sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from source_code.config.main import CONFIG
import source_code.config.models as model_config
from source_code import models

load_dotenv()

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────

BASE_PATH   = CONFIG["paths"]["base_data"]      # e.g. .../year_2
MODEL_NAME  = model_config.MODEL_CONFIGS["gemini"]["model"] # e.g. gemini-3.1-flash-lite-preview

MAX_CHARS_PER_UNIT = 80_000   # stays well within Gemini's context window
MAX_RETRIES        = 3
RETRY_DELAY        = 15       # seconds between retries

# ──────────────────────────────────────────────────────────────────────────────
# PROMPT
# ──────────────────────────────────────────────────────────────────────────────

WIKI_PROMPT = """\
You are a university study assistant. I will give you:
1. All the raw OCR text extracted from a student's course notes for ONE unit.
2. The official SYLLABUS topics for this unit.
3. Actual PAST YEAR QUESTIONS (PYQs) for this unit.

Your job is to compile the notes into a set of structured wiki pages
that are optimized for exam preparation — NOT for general learning.

## Output format

Return a JSON array. Each element is one wiki page:

[
  {{
    "topic": "Short topic title, e.g. 'File Handling'",
    "filename": "snake_case_filename_no_spaces.md",
    "content": "Full markdown content of the page (see format below)"
  }},
  ...
]

## Wiki page markdown format

Each page must follow this exact structure:

# <Topic Title>

**Subject**: {subject} | **Unit**: {unit}

## Definition
One clear exam-ready definition. Use the exact wording from the notes.

## Key points
- Bullet points covering what students need to write in an exam answer.
- Include syntax, modes, keywords, or steps as relevant.
- Each point should be self-contained and exam-relevant.

## Important terms
| Term | Meaning |
|------|---------|
| term | one-line definition |

## Actual & Predicted Exam Questions
- List 2-3 ACTUAL past year questions from the provided list if they relate to this topic.
- If no actual questions apply, list 1-2 highly probable predicted questions based on the notes.
- Format: "[Actual 2021] Question text..." or "[Predicted] Question text..."

## See also
- [[Related topic 1]]
- [[Related topic 2]]

## Rules

- Extract 4-8 pages per unit, one per distinct topic found in the notes.
- Every page must have all sections above, even if brief.
- Use the official syllabus topics to guide the organization and naming.
- "Key points" must use the language of the notes — not generic textbook language.
- Do NOT invent content not present in the notes.
- Do NOT merge unrelated topics into one page.
- Output ONLY the raw JSON array. No markdown fences, no extra text.
"""


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def load_chunks_for_subject(subject: str) -> dict[str, list[dict]]:
    """
    Scans BASE_PATH/<subject>/notes/ for all *.json files.
    Returns a dict keyed by unit string, each value a list of chunk dicts.
    """
    subject_path = Path(BASE_PATH) / subject
    notes_path = subject_path / "notes"
    if not notes_path.exists():
        print(f"Notes folder not found: {notes_path}")
        # Try without 'notes' subfolder if it doesn't exist (legacy/alternative structure)
        notes_path = subject_path
        if not notes_path.exists():
            raise FileNotFoundError(f"Subject folder not found: {subject_path}")

    json_files = sorted(notes_path.rglob("*.json"))
    print(f"Found {len(json_files)} chunk JSON files for {subject}.")

    by_unit: dict[str, list[dict]] = defaultdict(list)

    for jf in json_files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"  ⚠ Skipping {jf.name}: {e}")
            continue

        meta = data.get("extracted_metadata", {})
        full_text = meta.get("full_text", "").strip()
        if not full_text:
            continue

        # Prefer unit from top-level key; fall back to extracted_metadata
        unit_val = data.get("unit") or meta.get("unit") or "unknown"
        unit = str(unit_val).strip().lower()
        if unit in ("none", "null", ""):
            unit = "unknown"
        
        # Normalize unit string (ensure consistency like 'unit1' vs '1')
        if unit.isdigit():
            unit = f"unit{unit}"

        by_unit[unit].append({
            "file": jf.name,
            "unit": unit,
            "subject": data.get("subject", subject),
            "title": meta.get("title", ""),
            "full_text": full_text,
            "page_start": data.get("page_start", 0),
        })

    return dict(by_unit)


def load_syllabus_for_unit(subject: str, unit: str) -> dict | None:
    """Loads topics and unit title from syllabus_unit_<N>.json."""
    unit_num = "".join(filter(str.isdigit, unit))
    if not unit_num:
        return None
    
    syllabus_path = Path(BASE_PATH) / subject / "syllabus" / f"syllabus_unit_{unit_num}.json"
    if not syllabus_path.exists():
        return None
        
    try:
        with open(syllabus_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"  ⚠ Error loading syllabus for {unit}: {e}")
        return None


def load_pyqs_for_unit(subject: str, unit: str) -> list[dict]:
    """Loads all processed PYQs for a subject and filters by unit."""
    pyqs_dir = Path(BASE_PATH) / subject / "pyqs" / "pyqs_processed"
    if not pyqs_dir.exists():
        return []
        
    unit_num = "".join(filter(str.isdigit, unit))
    if not unit_num:
        return []
    
    try:
        target_unit = int(unit_num)
    except ValueError:
        return []
    
    matched_pyqs = []
    for jf in pyqs_dir.glob("*.json"):
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    for q in data:
                        if q.get("unit") == target_unit:
                            matched_pyqs.append(q)
        except Exception as e:
            print(f"  ⚠ Error loading PYQ {jf.name}: {e}")
            continue
            
    return matched_pyqs


def build_unit_context(chunks: list[dict]) -> str:
    """Concatenate full_text from all chunks for a unit, up to MAX_CHARS_PER_UNIT."""
    parts = []
    total = 0

    # Sort by page order
    for chunk in sorted(chunks, key=lambda c: c.get("page_start", 0)):
        header = f"\n\n--- {chunk['file']} (p.{chunk['page_start']}) ---\n"
        body   = chunk["full_text"]
        addition = header + body

        if total + len(addition) > MAX_CHARS_PER_UNIT:
            remaining = MAX_CHARS_PER_UNIT - total
            if remaining > 200:
                parts.append(addition[:remaining] + "\n[...truncated]")
            break

        parts.append(addition)
        total += len(addition)

    return "".join(parts)


def call_gemini(prompt: str) -> list[dict] | None:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # Use unified models.chat() instead of direct genai calls
            response_text = models.chat(
                prompt=prompt,
                model=MODEL_NAME,
                provider="gemini",
                temperature=0.1,
                max_tokens=8192
            )
            
            raw = response_text.strip()

            if "⚠ Gemini Error" in raw:
                print(f"  ⚠ Attempt {attempt}: {raw}")
                time.sleep(RETRY_DELAY)
                continue

            # Strip markdown fences if the model added them
            if raw.startswith("```"):
                if raw.startswith("```json"):
                    raw = raw.split("\n", 1)[1] if "\n" in raw else raw[7:]
                else:
                    raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
                raw = raw.rsplit("```", 1)[0].strip()

            pages = json.loads(raw)
            if isinstance(pages, list):
                return pages

            print(f"  ⚠ Attempt {attempt}: Gemini returned non-list JSON.")
            print(f"  Raw output sample: {raw[:200]}...")
        except json.JSONDecodeError as e:
            print(f"  ⚠ Attempt {attempt}: JSON parse error — {e}")
            print(f"  Raw output sample: {raw[:200]}...")
        except Exception as e:
            err = str(e)[:120]
            print(f"  ⚠ Attempt {attempt}: {err}")

        if attempt < MAX_RETRIES:
            print(f"  Retrying in {RETRY_DELAY}s...")
            time.sleep(RETRY_DELAY)

    return None


def write_wiki_pages(pages: list[dict], subject: str, unit: str, output_dir: Path, force: bool):
    written = 0
    for page in pages:
        topic    = page.get("topic", "unknown")
        filename = page.get("filename", "")
        content  = page.get("content", "").strip()

        if not filename or not content:
            print(f"    ⚠ Skipping page '{topic}' — missing filename or content.")
            continue

        # Ensure .md extension
        if not filename.endswith(".md"):
            filename += ".md"

        out_path = output_dir / filename

        if out_path.exists() and not force:
            # print(f"    -> {filename} exists, skipping.")
            continue

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(content + "\n")

        print(f"    ✅ {filename}")
        written += 1

    return written


def update_index(wiki_dir: Path, subject: str):
    """Write/update a wiki/index.md that catalogs all pages."""
    all_pages = sorted(wiki_dir.rglob("*.md"))
    all_pages = [p for p in all_pages if p.name != "index.md"]

    lines = [
        f"# {subject} wiki index\n",
        f"_{len(all_pages)} pages · auto-generated by compile_wiki.py_\n\n",
    ]

    # Group by unit subfolder
    by_unit: dict[str, list[Path]] = defaultdict(list)
    for p in all_pages:
        rel = p.relative_to(wiki_dir)
        unit_folder = rel.parts[0] if len(rel.parts) > 1 else "misc"
        by_unit[unit_folder].append(p)

    for unit_folder, pages in sorted(by_unit.items()):
        lines.append(f"## {unit_folder}\n\n")
        for page in sorted(pages):
            name = page.stem.replace("_", " ").title()
            rel_link = page.relative_to(wiki_dir)
            lines.append(f"- [[{rel_link}]] — {name}\n")
        lines.append("\n")

    index_path = wiki_dir / "index.md"
    with open(index_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print(f"\n📋 Updated index: {index_path}")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def compile_wiki(subject: str, force: bool = False, unit_filter: str | None = None):
    print(f"\n🧠 uniAI Wiki Compiler")
    print(f"   Subject : {subject}")
    print(f"   Model   : {MODEL_NAME}\n")

    # Load all chunks grouped by unit
    by_unit = load_chunks_for_subject(subject)

    if not by_unit:
        print("❌ No chunks found. Run extract_multimodal.py first.")
        return

    # Wiki output root: data/year_2/<SUBJECT>/wiki/
    wiki_root = Path(BASE_PATH) / subject / "wiki"
    wiki_root.mkdir(exist_ok=True, parents=True)

    total_written = 0

    # Normalize unit_filter
    if unit_filter:
        unit_filter = unit_filter.lower()
        if unit_filter.isdigit():
            unit_filter = f"unit{unit_filter}"

    for unit, chunks in sorted(by_unit.items()):

        if unit_filter and unit != unit_filter:
            continue

        print(f"\n📖 Unit: {unit}  ({len(chunks)} chunks)")

        # One subfolder per unit inside wiki/
        unit_dir = wiki_root / unit
        unit_dir.mkdir(exist_ok=True, parents=True)

        context = build_unit_context(chunks)
        print(f"   Context size: {len(context):,} chars")

        # --- Load Extra Context ---
        syllabus_data = load_syllabus_for_unit(subject, unit)
        pyqs = load_pyqs_for_unit(subject, unit)

        extra_context = ""
        if syllabus_data:
            topics_str = ", ".join(syllabus_data.get("topics", []))
            extra_context += f"\n\n## Official Syllabus Topics for {unit}\n{topics_str}\n"
        
        if pyqs:
            extra_context += f"\n\n## Actual Past Year Questions for {unit}\n"
            for q in pyqs:
                year = q.get("year", "N/A")
                marks = q.get("marks", "N/A")
                text = q.get("question_text", "")
                extra_context += f"- [Actual {year}, {marks} marks] {text}\n"

        # Build the full prompt
        prompt = WIKI_PROMPT.format(subject=subject, unit=unit)
        prompt += f"\n\n{extra_context}"
        prompt += f"\n\n## Raw course notes for {subject} — {unit}\n\n{context}"

        print(f"   Calling Gemini...", end="", flush=True)
        pages = call_gemini(prompt)

        if pages is None:
            print(f"\n   ❌ Failed after {MAX_RETRIES} attempts. Skipping unit.")
            continue

        print(f" ✅ Got {len(pages)} pages.")

        n = write_wiki_pages(pages, subject, unit, unit_dir, force)
        total_written += n

        # Respect Gemini rate limits
        time.sleep(4)

    # Write index
    update_index(wiki_root, subject)

    print(f"\n✅ Done. {total_written} wiki pages written to {wiki_root}")
    print(f"\nNext step: run ingest_wiki.py to embed the wiki pages into ChromaDB.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile OCR chunks into wiki pages.")
    parser.add_argument("--subject", required=True, help="Subject folder name, e.g. PYTHON")
    parser.add_argument("--unit",    default=None,  help="Process only this unit, e.g. unit3")
    parser.add_argument("--force",   action="store_true", help="Overwrite existing wiki pages")
    args = parser.parse_args()

    compile_wiki(
        subject=args.subject.upper(),
        force=args.force,
        unit_filter=args.unit,
    )
