"""
prompts.py
──────────
Single source of truth for every LLM prompt in the project.

Prompts are organised into four groups:
  1. EXTRACTION  — OCR/VLM prompts used by extract_multimodal*.py
  2. RAG         — Retrieval-augmented chat prompts used by views.py / rag_chat.py
  3. ROUTING     — Subject-classification prompt (router model)
  4. PIPELINE    — Offline pipeline prompts (keyword map generation)

Static prompts are plain module-level strings.
Prompts that include runtime variables are builder functions returning a str.
"""


# ══════════════════════════════════════════════════════════════════════════════
# 1. EXTRACTION PROMPTS
# ══════════════════════════════════════════════════════════════════════════════

NOTES_EXTRACTION = """\
You are an OCR + metadata extraction system for university course materials.

You will receive images of PDF pages. These may be handwritten notes, printed \
slides, question papers, or diagrams. Text CANNOT be copy-pasted from these — \
you must read them visually.

## Your Tasks

### Task 1 — Full OCR
Read ALL visible text from the images. Transcribe it faithfully:
- Preserve headings, bullet points, numbering, and structure
- For handwritten text, do your best to read it accurately
- For code snippets, preserve indentation and syntax
- Skip watermarks, page numbers, and headers/footers
- If a diagram is present, describe it briefly in [DIAGRAM: ...]

### Task 2 — Structured Metadata
Classify and tag the content you extracted.

## Output Format
Return ONLY a valid JSON object (no markdown fences, no extra text):

{
  "full_text": "The complete transcribed text from all pages, preserving structure with newlines",
  "title": "The topic title visible on the pages (e.g. 'Functions in Python', '2023 End Sem Paper')",
  "unit": "Unit number if identifiable (e.g. '1', '3'), else null",
  "document_type": "One of: question_paper, handwritten_notes, printed_notes, syllabus, lab_manual, other",
  "topics": ["List of specific subtopics covered in these pages"],
  "key_concepts": ["Important definitions, formulas, theorems, or algorithms mentioned"],
  "diagrams_present": false,
  "content_quality": "One of: clear, partially_legible, illegible",
  "confidence": 0.85
}

## Rules
- full_text must contain the ACTUAL text from the pages. This is the most important field.
- Be thorough — every readable sentence matters for search.
- Do NOT invent content that isn't visible.
- If pages are completely illegible, set confidence to 0.1 and full_text to empty string.
- confidence is a float between 0.0 and 1.0 reflecting OCR accuracy.
"""


SYLLABUS_EXTRACTION = """\
You are a precise syllabus extraction system for university course documents.

You will receive image(s) of a university course syllabus page (typically a single \
dense table). Extract ALL information and return it as a single valid JSON object \
with the structure below. Do NOT output markdown fences, comments, or any text \
outside the JSON.

Required JSON structure:

{
  "syllabus_version": "The course code printed on the syllabus, e.g. 'BCS302', 'BCC302'. \
If multiple codes appear (e.g. 'BCC302 / BCC402H'), use the primary/first one.",
  "subject_name": "Full subject title, e.g. 'Computer Organization and Architecture'",
  "units": [
    {
      "unit_number": 1,
      "unit_title": "Short title if present, e.g. 'Introduction' or 'Arithmetic and logic unit'",
      "topics": [
        "Each distinct sub-topic as a separate string. Split by commas/semicolons where appropriate.",
        "..."
      ],
      "proposed_lectures": 8,
      "full_text": "Complete verbatim text block for this unit exactly as printed"
    }
  ],
  "course_outcomes": [
    {
      "co_number": 1,
      "description": "Full CO description text",
      "blooms_level": ["K1", "K2"]
    }
  ],
  "textbooks": [
    "Full citation string for each textbook"
  ],
  "reference_books": [
    "Full citation string for each reference book (if a separate reference section exists, else empty array)"
  ]
}

Rules:
- Extract EXACTLY as many units as appear (usually 5).
- topics must be individual sub-topic strings — split compound topics at commas where sensible.
- blooms_level is an array of strings like ["K1", "K2"] or ["K3"].
- If textbooks and references are in the same list with no separation, put all in textbooks \
and leave reference_books empty.
- full_text for each unit should be the complete raw sentence(s) from the syllabus for that unit.
- Do not invent content not visible in the image.
"""


# ══════════════════════════════════════════════════════════════════════════════
# 2. RAG CHAT PROMPT
# ══════════════════════════════════════════════════════════════════════════════

def rag_answer(query: str, context: str, conversation_history: str = "", pyq_context: str = "", pyq_count: int = 0) -> str:
    """
    Build the full RAG answer prompt.

    Args:
        query:                The user's question.
        context:              Pre-formatted retrieved context string (### Source N: ...).
        conversation_history: Formatted prior turns (User: ... / AI: ...).
        pyq_context:          Pre-formatted PYQ context string.
        pyq_count:            Number of similar PYQs found.
    """
    return f"""\
You are UniAI, an exam-focused assistant.

CONTEXT FROM NOTES:
{context}

SIMILAR PREVIOUS YEAR QUESTIONS (same unit):
{pyq_context}

This topic has appeared in {pyq_count} previous exams.

INSTRUCTIONS:
1. Prioritize syllabus-aligned explanation.
2. If PYQs are closely related:
   - Match expected depth.
   - Use exam-friendly phrasing.
   - Include keywords.
3. If partially related, use only as structural guidance.
4. Do NOT hallucinate beyond provided notes.
5. Write in exam-ready format.

Conversation so far:
{conversation_history}

User question:
{query}
"""


# ══════════════════════════════════════════════════════════════════════════════
# 3. ROUTING PROMPT
# ══════════════════════════════════════════════════════════════════════════════

def subject_router(query: str, subjects_list: str) -> str:
    return (
        "/no_think\n"
        "You are a routing agent for a university study assistant.\n"
        f"Known subjects: {subjects_list}\n\n"
        f'User query: "{query}"\n\n'
        "Reply ONLY with one of these exact strings: "
        f"{subjects_list}, NONE\n"
        "No explanation. No punctuation. Just the name."
    )

# ══════════════════════════════════════════════════════════════════════════════
# 4. PIPELINE PROMPTS
# ══════════════════════════════════════════════════════════════════════════════

def keyword_extraction(subject: str, items_list: str) -> str:
    """
    Build the keyword extraction prompt for generate_keyword_map.py.

    Args:
        subject:    Subject name (e.g. 'COA', 'PYTHON').
        items_list: Comma-separated discovered content titles/topics.
    """
    return f"""\
You are a taxonomy expert extracting academic keywords.
I will give you a Subject Name and a list of internal topics/titles found in that subject's syllabus.

Subject: {subject}
Discovered Content: {items_list}

Extract a concise, comma-separated list of the 10-15 most defining key phrases, \
concepts, or terms that definitively represent this subject.
Do not include generic words like "introduction", "overview", or "unit".
ONLY output the comma-separated list of keywords. \
No introductory text. No numbering. No markdown.\
"""

# ══════════════════════════════════════════════════════════════════════════════
# 5. PYQ EXTRACTION PROMPTS
# ══════════════════════════════════════════════════════════════════════════════

def pyq_unit_classification(question: str, syllabus_units: str) -> str:
    """
    Classify a given PYQ question into one of the syllabus units based on its text.

    Args:
        question:       The text of the question.
        syllabus_units: Formatted string of syllabus units (Unit X: topic1, topic2 ...).
    """
    return f"""\
You are an expert academic classifier.
Given the following syllabus units for a course:
{syllabus_units}

And the following exam question:
"{question}"

Classify this question into the most appropriate unit number (1-5).
Analyze the key concepts in the question and match them to the topics covered in each unit.
Only output the unit number as an integer (e.g., 1, 3, 5). No other text or explanation.
"""

PYQ_VLM_TRANSCRIPTION = """\
Read the text from this image block. It is part of a university exam question paper.
Transcribe it faithfully:
- Preserve headings, bullet points, numbering, and structure
- For mathematical equations, transcribe them clearly
- If there is a diagram, describe it briefly in [DIAGRAM: ...]
- Do NOT output anything else except the transcribed text
"""
