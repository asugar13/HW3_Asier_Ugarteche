"""
index.py — Build a ChromaDB vector store for a given corpus/chunk-size config.

Your existing hp_books collection = full corpus (novels + companions), 500-char chunks.
Run these to build the ablation collections:

    # Corpus ablation — novels only:
    python index.py --collection hp_books_novels --novels-only

    # Chunk size ablation:
    python index.py --collection hp_chunks_250  --chunk-size 250
    python index.py --collection hp_chunks_1000 --chunk-size 1000
"""
import argparse
import chromadb
from loader import load_all_books, load_novels_only, chunk_book_chapters
from store import get_collection, add_chunks


def main(data_dir: str, collection_name: str, chunk_size: int, novels_only: bool):
    print(f"=== Know-it Owl — Indexing Pipeline ===")
    print(f"  Collection : {collection_name}")
    print(f"  Chunk size : {chunk_size} chars")
    print(f"  Corpus     : {'7 novels only' if novels_only else '7 novels + companions'}\n")

    print("Step 1: Loading and parsing books…")
    chapters = load_novels_only(data_dir) if novels_only else load_all_books(data_dir)

    print("\nStep 2: Chunking chapters…")
    overlap = max(20, chunk_size // 10)
    docs, metas, ids = chunk_book_chapters(chapters, chunk_size=chunk_size, chunk_overlap=overlap)

    print("\nStep 3: Connecting to ChromaDB…")
    collection = get_collection(name=collection_name)
    existing = collection.count()
    if existing > 0:
        print(f"  Collection '{collection_name}' already has {existing} chunks.")
        answer = input("  Re-index from scratch? (y/N): ").strip().lower()
        if answer == "y":
            client = chromadb.PersistentClient(path="owl_db")
            client.delete_collection(collection_name)
            collection = get_collection(name=collection_name)
        else:
            print("  Keeping existing index. Done.")
            return

    print("\nStep 4: Embedding and storing chunks…")
    add_chunks(collection, docs, metas, ids)

    print(f"\nDone! {collection.count()} chunks stored in owl_db/ under '{collection_name}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/",
                        help="Directory containing the HP .txt files")
    parser.add_argument("--collection", default="hp_books",
                        help="ChromaDB collection name")
    parser.add_argument("--chunk-size", type=int, default=500,
                        help="Chunk size in characters")
    parser.add_argument("--novels-only", action="store_true",
                        help="Use 7 novels only (exclude companion texts)")
    args = parser.parse_args()
    main(args.data_dir, args.collection, args.chunk_size, args.novels_only)
