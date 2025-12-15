import chromadb

client = chromadb.PersistentClient(path="source_code/chroma/python")
collection = client.get_collection("python")

query = "explain python loops"
res = collection.query(query_texts=[query], n_results=5)

print(res)
