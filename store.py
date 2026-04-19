"""
store.py — ChromaDB interface: initialise, add chunks, query.
"""
import chromadb
from chromadb.utils import embedding_functions

DB_PATH = "owl_db"
DEFAULT_COLLECTION = "hp_books"
EMBED_MODEL = "all-MiniLM-L6-v2"  # fast, good quality, ~80 MB

COLLECTION_LABELS = {
    "hp_books":        "7 novels + companions · 500 chars",
    "hp_books_novels": "7 novels only · 500 chars",
    "hp_chunks_250":   "7 novels + companions · 250 chars",
    "hp_chunks_1000":  "7 novels + companions · 1000 chars",
}


def _get_ef():
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )


def get_client():
    return chromadb.PersistentClient(path=DB_PATH)


def list_collections() -> list[str]:
    """Return names of all collections that actually exist in owl_db."""
    client = get_client()
    existing = {c.name for c in client.list_collections()}
    return [name for name in COLLECTION_LABELS if name in existing]


def get_collection(name: str = DEFAULT_COLLECTION):
    """Return (or create) a named ChromaDB collection."""
    client = get_client()
    ef = _get_ef()
    collection = client.get_or_create_collection(
        name=name,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def add_chunks(collection, docs, metas, ids, batch_size: int = 500):
    """Add document chunks in batches (ChromaDB limit per call)."""
    total = len(docs)
    for i in range(0, total, batch_size):
        batch_docs = docs[i:i + batch_size]
        batch_metas = metas[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]
        collection.add(documents=batch_docs, metadatas=batch_metas, ids=batch_ids)
        print(f"  Stored {min(i + batch_size, total)}/{total} chunks…")


def query_collection(collection, query_text: str, n_results: int = 5):
    """Retrieve top-n chunks most similar to the query."""
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )
    # Flatten single-query results
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]
    return list(zip(docs, metas, distances))
