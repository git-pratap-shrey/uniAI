import os, sys
import chromadb
import ollama

# --- Ensure imports work regardless of working directory ---
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

import config
from pipeline.embeddings.local_mxbai import embed


CHROMA_PATH = config.CHROMA_DB_PATH
MODEL = config.MODEL_CHAT  # configured in config.py

# Initialize DB + Collection
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_collection(config.CHROMA_COLLECTION_NAME)


def retrieve(query, n=5):
    """Return top chunks + metadata using MXBAI embeddings."""
    query_emb = embed([query])[0]
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=n,
        include=["documents", "metadatas"]
    )
    return results


def format_context(results):
    """Convert chunks into the prompt-friendly format."""
    ctx = ""
    for i in range(len(results["documents"][0])):
        ctx += f"\n### Source {i+1}:\n{results['documents'][0][i]}\n"
    return ctx


def answer(query, conversation_history=""):
    """Run full RAG pipeline: retrieve → prompt LLM → return answer."""
    results = retrieve(query)
    context = format_context(results)

    prompt = f"""
You are a university assistant trained on real syllabus, notes, and PYQs.

Answer the user based on the provided context, add that to your own knowledge, and information available on the internet,

your goal is therefore to provide answers that are relevant to the students' academic needs, for exams, assignments, and projects.

If the answer is not in the context, say: "here is some information about the topic from my knowledge: ".

Conversation so far:
{conversation_history}

User question:
{query}

Relevant context:
{context}

Answer clearly with bullet points and examples. Do NOT mention the context explicitly.
"""

    # Initialize client explicitly with config to avoid default host issues
    client = ollama.Client(host=config.OLLAMA_BASE_URL)

    print("   Thinking...", end="", flush=True)
    response = client.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    print("\r", end="") # Clear "Thinking..."

    return response["message"]["content"], results


def chat():
    print("🎓 UniAI RAG Chat — Ask academic questions (type 'exit' to quit)")
    conversation = ""

    while True:
        query = input("\n🧠 You: ")
        if query.lower() in ["exit", "quit"]:
            break

        response_text, results = answer(query, conversation)
        print("\n🤖 AI:", response_text)

        # Show sources
        print("\n📚 Sources Used:")
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        
        for i, (doc, meta) in enumerate(zip(docs, metas)):
            source_file = meta.get('source', 'Unknown')
            page_start = meta.get('page_start', '?')
            print(f"   {i+1}. {source_file} (p.{page_start})")

        # maintain short history
        conversation += f"\nUser: {query}\nAI: {response_text}\n"


if __name__ == "__main__":
    chat()
