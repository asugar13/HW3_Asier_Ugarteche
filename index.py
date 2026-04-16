"""
index.py — Run ONCE to build the ChromaDB vector store.

Usage:
    conda run -n tf python index.py
    # or with a custom data directory:
    conda run -n tf python index.py --data-dir /path/to/books
"""
import argparse
from loader import load_all_books, chunk_book_chapters
from store import get_collection, add_chunks


def main(data_dir: str):
    print("=== Know-it Owl — Indexing Pipeline ===\n")

    print("Step 1: Loading and parsing books…")
    chapters = load_all_books(data_dir)

    print("\nStep 2: Chunking chapters…")
    docs, metas, ids = chunk_book_chapters(chapters)

    print("\nStep 3: Connecting to ChromaDB…")
    collection = get_collection()
    existing = collection.count()
    if existing > 0:
        print(f"  Collection already has {existing} chunks.")
        answer = input("  Re-index from scratch? (y/N): ").strip().lower()
        if answer == "y":
            import chromadb
            client = chromadb.PersistentClient(path="owl_db")
            client.delete_collection("hp_books")
            collection = get_collection()
        else:
            print("  Keeping existing index. Done.")
            return

    print("\nStep 4: Embedding and storing chunks…")
    add_chunks(collection, docs, metas, ids)

    print(f"\nDone! {collection.count()} chunks stored in owl_db/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/",
                        help="Directory containing the HP .txt files")
    args = parser.parse_args()
    main(args.data_dir)
