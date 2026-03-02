"""
chat_cli.py
───────────
CLI chat loop for uniAI.

Responsibilities:
  - Session-level subject locking
  - Conversation history management
  - /commands handling
  - Display formatting
  - Calls rag_pipeline.answer_query() for all intelligence

No retrieval logic. No prompt logic.
"""

import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from rag.rag_pipeline import answer_query
from rag.router import list_subjects


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _print_header():
    print("\n" + "=" * 60)
    print("  🎓  uniAI — Syllabus-Aware Exam Assistant")
    print("=" * 60)
    print("  Commands:")
    print("    /switch <SUBJECT>  — lock to a subject (e.g. /switch COA)")
    print("    /switch            — clear subject lock")
    print("    /subject           — show current session subject")
    print("    /subjects          — list all known subjects")
    print("    /history           — show conversation history")
    print("    /clear             — clear conversation history")
    print("    exit / quit        — exit")
    print("=" * 60 + "\n")


def _print_answer(result: dict):
    print(f"\n🤖  [{result['mode'].upper()}]", end="")
    if result["subject"]:
        print(f"  Subject: {result['subject']}", end="")
    if result["unit"]:
        print(f"  Unit: {result['unit']}", end="")
    print()
    print("-" * 60)
    print(result["answer"])

    if result["sources"]:
        print("\n📚  Sources:")
        for src in result["sources"]:
            print(f"   • {src}")
    print()


def _print_history(history: list[dict]):
    if not history:
        print("  (no history yet)")
        return
    for turn in history:
        role = "You" if turn["role"] == "user" else "AI"
        content = turn["content"][:120] + ("..." if len(turn["content"]) > 120 else "")
        print(f"  [{role}] {content}")


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

def _handle_command(
    query: str,
    session_subject: str | None,
    history: list[dict],
) -> tuple[str | None, bool]:
    """
    Process a /command.

    Returns (new_session_subject, should_continue).
    should_continue=False means the command was handled and the loop
    should skip the normal query flow for this turn.
    """
    parts = query.strip().split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    if cmd == "/switch":
        if arg:
            session_subject = arg.upper()
            print(f"  ✅ Subject locked to: {session_subject}")
        else:
            session_subject = None
            print("  ✅ Subject lock cleared. Will auto-detect per query.")
        return session_subject, False

    if cmd == "/subject":
        if session_subject:
            print(f"  Current subject: {session_subject}")
        else:
            print("  No subject locked. Auto-detecting per query.")
        return session_subject, False

    if cmd == "/subjects":
        subjects = list_subjects()
        if subjects:
            print(f"  Known subjects: {', '.join(subjects)}")
        else:
            print("  No subjects loaded (keyword map missing?).")
        return session_subject, False

    if cmd == "/history":
        _print_history(history)
        return session_subject, False

    if cmd == "/clear":
        history.clear()
        print("  ✅ Conversation history cleared.")
        return session_subject, False

    print(f"  Unknown command: {cmd}")
    return session_subject, False


# ---------------------------------------------------------------------------
# Main chat loop
# ---------------------------------------------------------------------------

def chat():
    _print_header()

    history: list[dict] = []
    session_subject: str | None = None
    first_query = True

    while True:
        try:
            query = input("🧠  You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye! 👋")
            break

        if not query:
            continue

        if query.lower() in ("exit", "quit", "q"):
            print("\nGoodbye! 👋\n")
            break

        # Handle /commands
        if query.startswith("/"):
            session_subject, _ = _handle_command(query, session_subject, history)
            continue

        # Auto-detect subject on the first real query if not locked
        if first_query and not session_subject:
            print("  [Routing] Detecting subject...", end="", flush=True)

        # Run pipeline
        result = answer_query(
            query=query,
            history=history,
            session_subject=session_subject,
        )

        # Lock subject for session after first successful detection
        if first_query:
            print("\r" + " " * 40 + "\r", end="")  # clear routing line
            first_query = False
            if result["subject"] and not session_subject:
                session_subject = result["subject"]
                print(f"  [Session] Locked to subject: {session_subject}")

        _print_answer(result)

        # Update conversation history
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": result["answer"]})


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    chat()
