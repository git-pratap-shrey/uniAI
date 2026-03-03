from chromadb import PersistentClient

client = PersistentClient(path="chroma")  # NOT source_code/chroma
col = client.get_collection("multimodal_notes")

x = col.get(ids=["CYBER_SECURITY_unit1.pdf_p21-22"])
print(x["metadatas"])