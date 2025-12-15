import os, sys
import chromadb
import ollama

# --- Ensure imports work regardless of working directory ---
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

from pipeline.embeddings.local_mxbai import embed


CHROMA_PATH = r"D:\CODE-workingBuild\uniAI\source_code\chroma\python"
MODEL = "llama3.1:8b"  # change later if needed


# Initialize DB + Collection
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_collection("python")


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

    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]


def chat():
    print("🎓 UniAI RAG Chat — Ask academic questions (type 'exit' to quit)")
    conversation = ""

    while True:
        query = input("\n🧠 You: ")
        if query.lower() in ["exit", "quit"]:
            break

        response = answer(query, conversation)
        print("\n🤖 AI:", response)

        # maintain short history
        conversation += f"\nUser: {query}\nAI: {response}\n"


if __name__ == "__main__":
    chat()
