from chromadb import PersistentClient
from pathlib import Path

# Adjust this if your persist directory is different
CHROMA_PATH = Path(__file__).resolve().parents[2] / "chroma"


def main():
    print("==== ChromaDB Audit ====\n")

    print(f"Using DB path: {CHROMA_PATH}\n")

    client = PersistentClient(path=str(CHROMA_PATH))

    collections = client.list_collections()

    if not collections:
        print("No collections found.")
        return

    print(f"Found {len(collections)} collections.\n")

    for col in collections:
        collection = client.get_collection(col.name)
        count = collection.count()

        print(f"Collection: {col.name}")
        print(f"Vector count: {count}")
        print("-" * 40)


if __name__ == "__main__":
    main()