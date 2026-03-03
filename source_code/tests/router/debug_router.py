import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import config
import ollama
from prompts import subject_router

def test_router_model():
    query = "Define ransomware."
    subjects_list = "COA, PYTHON, CYBER_SECURITY"

    prompt = subject_router(query=query, subjects_list=subjects_list)

    print("===== PROMPT SENT TO MODEL =====")
    print(prompt)
    print("=================================\n")

    client = ollama.Client(host=config.OLLAMA_LOCAL_URL)

    response = client.chat(
        model=config.MODEL_ROUTER,
        messages=[
            {"role": "user", "content": prompt}
        ],
        think=False,
        options={
            "temperature": 0,
            "num_predict": 50,
        }
    )

    print("===== RAW RESPONSE OBJECT =====")
    print(response)
    print("================================\n")

    print("===== MODEL OUTPUT =====")
    print(response["message"]["content"])
    print("================================")

if __name__ == "__main__":
    test_router_model()