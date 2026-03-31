"""
parser.py
─────────
Parse questions.txt into structured Question objects.

Extracts:
- Subject headers (e.g., "SUBJECT: DIGITAL ELECTRONICS")
- Question numbers, text, and expected answers
- Maps questions to their respective subjects
"""

import re
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Question:
    """Represents a single test question with expected answer."""
    question_id: str      # e.g., "Q1", "Q2"
    subject: str          # Subject name from section header
    subject_code: str     # e.g., "BEC301/401"
    query: str            # The actual question text
    expected_answer: str  # The expected answer from file
    line_number: int      # For debugging

    def __repr__(self) -> str:
        return f"Question({self.question_id}, {self.subject}: {self.query[:50]}...)"


def parse_questions_file(filepath: str) -> List[Question]:
    """
    Parse the questions.txt file into a list of Question objects.

    The file format has:
    - File header (e.g., "AKTU_UNI_AI_TEST_SET_V1.txt")
    - Separator lines (e.g., "--------------------------------------------------")
    - Subject headers: "SUBJECT: NAME (CODE)"
    - Questions: "Q1: question text"
    - Expected answers: "EXPECTED ANSWER: answer text" (can span multiple lines)

    Args:
        filepath: Path to questions.txt

    Returns:
        List of Question objects
    """
    questions = []
    current_subject = ""
    current_subject_code = ""

    # Current question being built
    current_q_id = ""
    current_q_text = ""
    current_q_expected = ""

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    lines = content.split('\n')
    i = 0

    def save_question():
        """Save the current question if complete."""
        nonlocal current_q_id, current_q_text, current_q_expected
        if current_q_id and current_q_text and current_q_expected:
            questions.append(Question(
                question_id=current_q_id,
                subject=current_subject,
                subject_code=current_subject_code,
                query=current_q_text.strip(),
                expected_answer=current_q_expected.strip(),
                line_number=i
            ))
        current_q_id = ""
        current_q_text = ""
        current_q_expected = ""

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Skip empty lines and separators
        if not stripped or stripped.startswith('-') or stripped.startswith('='):
            i += 1
            continue

        # Skip file header
        if 'AKTU' in stripped and '.txt' in stripped:
            i += 1
            continue

        # Parse SUBJECT header
        subject_match = re.match(r'SUBJECT:\s*(.+?)\s*\(([A-Z0-9/]+)\)', stripped, re.IGNORECASE)
        if subject_match:
            # Save any pending question first
            save_question()
            current_subject = subject_match.group(1).strip()
            current_subject_code = subject_match.group(2)
            i += 1
            continue

        # Parse QUESTION (Q1:, Q2:, etc.)
        q_match = re.match(r'(Q\d+):\s*(.+)', stripped, re.IGNORECASE)
        if q_match:
            # Save previous question if exists
            save_question()
            current_q_id = q_match.group(1).upper()
            current_q_text = q_match.group(2).strip()
            current_q_expected = ""
            i += 1
            continue

        # Parse EXPECTED ANSWER
        if stripped.upper().startswith('EXPECTED ANSWER:'):
            answer_text = stripped[len('EXPECTED ANSWER:'):].strip()
            current_q_expected = answer_text
            i += 1

            # Continue reading multi-line expected answer
            while i < len(lines):
                next_line = lines[i].strip()

                # Stop if we hit a new question, subject, or separator
                if not next_line:
                    i += 1
                    continue

                if (re.match(r'Q\d+:', next_line, re.IGNORECASE) or
                    re.match(r'SUBJECT:', next_line, re.IGNORECASE) or
                    next_line.startswith('-') or
                    next_line.startswith('=')):
                    break

                # Append to expected answer
                current_q_expected += " " + next_line
                i += 1

            continue

        i += 1

    # Save the last question
    save_question()

    return questions


def get_subject_alias(subject_name: str) -> Optional[str]:
    """
    Map subject names from questions.txt to internal subject aliases.

    Args:
        subject_name: Subject name from file (e.g., "DIGITAL ELECTRONICS")

    Returns:
        Internal subject code or None
    """
    subject_map = {
        "DIGITAL ELECTRONICS": "DIGITAL_ELECTRONICS",
        "UNIVERSAL HUMAN VALUES": "UHV",
        "CYBER SECURITY": "CYBER_SECURITY",
    }
    return subject_map.get(subject_name.upper())


if __name__ == "__main__":
    # Test parsing
    import os
    filepath = os.path.join(os.path.dirname(__file__), "questions.txt")
    questions = parse_questions_file(filepath)

    print(f"Parsed {len(questions)} questions:\n")
    for q in questions:
        print(f"{q.question_id} [{q.subject}] ({q.subject_code})")
        print(f"  Q: {q.query[:80]}...")
        print(f"  A: {q.expected_answer[:80]}...")
        print()
