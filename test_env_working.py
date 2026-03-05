import sys
import os

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from rag.rag_pipeline import answer_query
    
print("Running execution sweep...")    
res = answer_query("Explain SQL injection", session_subject="CYBER_SECURITY", history=[])
print(res)
