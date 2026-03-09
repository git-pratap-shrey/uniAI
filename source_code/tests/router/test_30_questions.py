import os
import sys

# -------------------------------------------------
# Fix import path
# -------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../"))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rag.hybrid_router import route
import random

test_cases = [
    # UHV
    {"question": "Explain the difference between sukh and suvidha.", "expected_subject": "UHV"},
    {"question": "What is meant by holistic development and value education?", "expected_subject": "UHV"},
    {"question": "Discuss the concept of natural acceptance with an example.", "expected_subject": "UHV"},
    {"question": "How do sanyam and swasthya contribute to the harmony of self with the body?", "expected_subject": "UHV"},
    {"question": "Explain the four orders of nature and their interconnectedness.", "expected_subject": "UHV"},
    {"question": "What are the universal human goals for a fearless society?", "expected_subject": "UHV"},
    {"question": "Describe the role of self-exploration in human consciousness.", "expected_subject": "UHV"},
    {"question": "Discuss the importance of ethical human conduct in a professional setting.", "expected_subject": "UHV"},
    {"question": "How can we ensure harmony in human-human relationships?", "expected_subject": "UHV"},
    {"question": "Elaborate on the concept of submergence and recyclability in nature.", "expected_subject": "UHV"},

    # DIGITAL ELECTRONICS
    {"question": "DE syllabus", "expected_subject": "DIGITAL_ELECTRONICS"},
    {"question": "Simplify a boolean expression using Karnaugh maps.", "expected_subject": "DIGITAL_ELECTRONICS"},
    {"question": "Design a full adder circuit using logic gates.", "expected_subject": "DIGITAL_ELECTRONICS"},
    {"question": "Explain the race around condition in JK flip-flops and how to avoid it.", "expected_subject": "DIGITAL_ELECTRONICS"},
    {"question": "What is the difference between sequential circuits and combinational circuits?", "expected_subject": "DIGITAL_ELECTRONICS"},
    {"question": "Construct a 4-to-1 multiplexer.", "expected_subject": "DIGITAL_ELECTRONICS"},
    {"question": "Convert a binary number to its gray code equivalent.", "expected_subject": "DIGITAL_ELECTRONICS"},
    {"question": "Describe the Quine McCluskey method for prime implicants.", "expected_subject": "DIGITAL_ELECTRONICS"},
    {"question": "Explain the operation of a shift register with a diagram.", "expected_subject": "DIGITAL_ELECTRONICS"},
    {"question": "Analyze a clocked sequential circuit to determine its state diagram.", "expected_subject": "DIGITAL_ELECTRONICS"},
    {"question": "Discuss the implementation of boolean functions using a PLA.", "expected_subject": "DIGITAL_ELECTRONICS"},

    # CYBER SECURITY
    {"question": "What is the difference between phishing and social engineering?", "expected_subject": "CYBER_SECURITY"},
    {"question": "Explain the concept of CIA triad in information security.", "expected_subject": "CYBER_SECURITY"},
    {"question": "How does a botnet facilitate DDoS attacks?", "expected_subject": "CYBER_SECURITY"},
    {"question": "Discuss the challenges in mobile device security and authentication.", "expected_subject": "CYBER_SECURITY"},
    {"question": "Describe the steps involved in a computer forensics investigation.", "expected_subject": "CYBER_SECURITY"},
    {"question": "What are the key provisions of the DPDP act?", "expected_subject": "CYBER_SECURITY"},
    {"question": "How does SQL injection vulnerability affect web applications?", "expected_subject": "CYBER_SECURITY"},
    {"question": "Define cyber stalking and defamation under Indian Cyber Law.", "expected_subject": "CYBER_SECURITY"},
    {"question": "Explain the process of digital evidence preservation and chain of custody.", "expected_subject": "CYBER_SECURITY"},
    {"question": "Elaborate on the security implications of wireless network attacks.", "expected_subject": "CYBER_SECURITY"},
]

# Provide a fixed seed so the jumbled order is deterministic but random-looking
random.seed(42)
random.shuffle(test_cases)

def run_tests():
    total = len(test_cases)
    correct = 0
    print(f"============================================================")
    print(f"Running Routing Test Cases (30 questions, 10 per subject)")
    print(f"============================================================\\n")
    
    for i, case in enumerate(test_cases, 1):
        q = case["question"]
        expected = case["expected_subject"]
        
        try:
            res = route(q)
            subj = res.subject
            method = res.method
            
            if subj == expected:
                correct += 1
                status = "✅ PASS"
            else:
                status = f"❌ FAIL (Expected {expected}, got {subj})"
                
            print(f"[{i}/{total}] {status} | METHOD: {method.upper()} | Q: {q}")
        except Exception as e:
            print(f"[{i}/{total}] ❌ ERROR processing Q: {q} - {e}")
            
    print(f"\\n============================================================")
    print(f"Final Routing Accuracy: {correct}/{total} ({(correct/total)*100:.2f}%)")
    print(f"============================================================")

if __name__ == "__main__":
    run_tests()
