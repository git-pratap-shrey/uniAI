import ollama

def embed(texts: list[str]) -> list[list[float]]:
    vectors = []
    for text in texts:
        res = ollama.embeddings(
            model="mxbai-embed-large",
            prompt=text
        )
        vectors.append(res["embedding"])
    return vectors
