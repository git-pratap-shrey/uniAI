from chromadb import PersistentClient
from pathlib import Path
import textwrap

CHROMA_PATH = Path(__file__).resolve().parents[2] / "chroma"

# Output file at same folder level as this script
OUTPUT_FILE = Path(__file__).parent / "db_dump_output.txt"


def log(message, file_handle):
    print(message)
    file_handle.write(message + "\n")


def main():
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

        log("==== ChromaDB Full Dump ====", f)
        log(f"Using DB path: {CHROMA_PATH}\n", f)

        client = PersistentClient(path=str(CHROMA_PATH))
        collections = client.list_collections()

        if not collections:
            log("No collections found.", f)
            return

        for col in collections:
            collection = client.get_collection(col.name)
            count = collection.count()

            log("\n" + "=" * 80, f)
            log(f"Collection: {col.name}", f)
            log(f"Total vectors: {count}", f)
            log("=" * 80 + "\n", f)

            data = collection.get(include=["documents", "metadatas"])

            ids = data.get("ids", [])
            docs = data.get("documents", [])
            metas = data.get("metadatas", [])

            for i in range(len(ids)):
                log(f"[{i+1}] ID: {ids[i]}", f)

                log("Metadata:", f)
                if metas[i]:
                    for k, v in metas[i].items():
                        log(f"   - {k}: {v}", f)
                else:
                    log("   - None", f)

                log("Preview:", f)
                if docs[i]:
                    wrapped = textwrap.fill(docs[i][:300], width=80)
                    log(wrapped, f)
                else:
                    log("None", f)

                log("-" * 80, f)


if __name__ == "__main__":
    main()