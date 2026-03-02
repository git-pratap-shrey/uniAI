"""
prompts.py
──────────
All prompt templates for uniAI.

Centralised here so tone, rules, and structure can be tuned
without touching pipeline logic.
"""


def rag_answer(
    query: str,
    notes_context: str,
    history_block: str,
    mode: str = "syllabus",          # "syllabus" | "generic"
    subject: str | None = None,
) -> str:
    """
    Build the full prompt sent to the generation model.

    mode="syllabus"  → strict, grounded in notes, exam tone
    mode="generic"   → general knowledge, labeled clearly
    """

    if mode == "syllabus":
        subject_line = f" for {subject}" if subject else ""
        system = f"""\
You are uniAI, a syllabus-aware exam assistant{subject_line}.

You are given OCR-extracted text from real university course notes and/or syllabus.
Your job is to help students write the best possible exam answers.

Rules:
- Answer strictly from the provided notes context.
- Lead with a clear definition if the question asks "what is".
- Use bullet points, numbered steps, and bold keywords where appropriate.
- Write in "what to write in exam" tone — concise, structured, keyword-rich.
- If the answer spans multiple sources, synthesize them into one coherent answer.
- If information is missing from the notes, say: "This is not in the provided notes." 
  Then provide a brief general answer labeled [General Knowledge].
- Never fabricate syllabus content.\
"""
    else:
        system = """\
[GENERIC AI TUTOR MODE]

This question appears to be outside the available syllabus notes.
You are answering from general knowledge. Make this clear to the student.

Rules:
- Label your response clearly with [General Knowledge].
- Still use structured, exam-friendly formatting.
- Be honest if you are uncertain.\
"""

    sections = [system]

    if history_block:
        sections.append(history_block)

    if notes_context:
        sections.append(f"Relevant notes:\n\n{notes_context}")
    elif mode == "syllabus":
        sections.append("(No relevant notes were found in the database for this query.)")

    sections.append(f"Student question:\n{query}")

    return "\n\n".join(sections)


def topic_list(subject: str, unit: str) -> str:
    """Prompt for listing topics in a unit — used when the query is a unit overview."""
    return (
        f"List all the key topics covered in Unit {unit} of {subject}.\n"
        f"Format as a numbered list. Be concise."
    )
