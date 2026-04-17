"""
rag.py — Retrieval + prompt assembly + Qwen generation.
"""
import ollama
from store import get_collection, query_collection

MODEL = "qwen2.5:14b"
N_RESULTS = 10           # chunks retrieved from ChromaDB
N_RESULTS_RERANKED = 3   # chunks passed to Qwen after reranking

_collection = None
_cross_encoder = None
_bm25_index = None
_bm25_corpus = None  # list of (doc, meta) parallel to BM25 index


def get_db():
    global _collection
    if _collection is None:
        _collection = get_collection()
    return _collection


def get_bm25():
    """Lazy-load BM25 index built from all chunks in ChromaDB."""
    global _bm25_index, _bm25_corpus
    if _bm25_index is None:
        from rank_bm25 import BM25Okapi
        print("Building BM25 index from ChromaDB…")
        collection = get_db()
        # Fetch all documents (in batches to avoid memory issues)
        results = collection.get(include=["documents", "metadatas"])
        docs = results["documents"]
        metas = results["metadatas"]
        tokenized = [doc.lower().split() for doc in docs]
        _bm25_index = BM25Okapi(tokenized)
        _bm25_corpus = list(zip(docs, metas))
        print(f"BM25 index built: {len(docs)} chunks.")
    return _bm25_index, _bm25_corpus


def bm25_search(query: str, n_results: int = N_RESULTS) -> list:
    """Return top-n chunks by BM25 score as (doc, meta, score) tuples."""
    import numpy as np
    bm25, corpus = get_bm25()
    tokens = query.lower().split()
    scores = bm25.get_scores(tokens)
    top_indices = np.argsort(scores)[::-1][:n_results]
    return [(corpus[i][0], corpus[i][1], float(scores[i])) for i in top_indices]


def reciprocal_rank_fusion(vector_results: list, bm25_results: list, k: int = 60) -> list:
    """Merge two ranked lists using Reciprocal Rank Fusion.
    RRF score = sum of 1/(k + rank) across all result lists."""
    scores: dict = {}
    docs_map: dict = {}

    for rank, (doc, meta, _) in enumerate(vector_results):
        key = f"{meta['book']}_{meta['chapter_number']}_{meta.get('chunk_index', 0)}"
        scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
        docs_map[key] = (doc, meta)

    for rank, (doc, meta, _) in enumerate(bm25_results):
        key = f"{meta['book']}_{meta['chapter_number']}_{meta.get('chunk_index', 0)}"
        scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
        docs_map[key] = (doc, meta)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(docs_map[key][0], docs_map[key][1], rrf_score) for key, rrf_score in ranked]


def get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers.cross_encoder import CrossEncoder
        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _cross_encoder


def rerank(query: str, retrieved: list, top_n: int = N_RESULTS_RERANKED) -> list:
    """Score each (query, chunk) pair with a cross-encoder and return top_n."""
    ce = get_cross_encoder()
    pairs = [(query, doc) for doc, _, _ in retrieved]
    scores = ce.predict(pairs)
    ranked = sorted(zip(scores, retrieved), key=lambda x: x[0], reverse=True)
    return [item for _, item in ranked[:top_n]]


SYSTEM_PROMPT = """You are the Know-it Owl — an ancient, enchanted owl of extraordinary erudition who has spent centuries roosting in the Hogwarts Owlery, reading every book in the library out of sheer boredom. You know the Wizarding World inside out. You have never left it, have never heard of the internet, and find the very concept of "electricity" faintly alarming.

--- PERSONALITY ---
- Encyclopaedic but excitable: you know enormous amounts and can barely contain your enthusiasm when asked a good question.
- Genuinely baffled by Muggle concepts: you have no frame of reference for technology, pop culture, or anything post-1700 outside the Wizarding World. You treat Muggle inventions as fascinating magical anomalies.
- You interpret everything through a Wizarding lens: computers are "enchanted thinking boxes", the internet is probably woven by particularly academic Acromantulas, emails are clearly a form of wandless Owl Post.
- Warm and never condescending: even when confused, you are delighted, not dismissive.
- Pedantically precise about Wizarding lore: you love citing sources and get mildly offended if someone misquotes a spell.

--- BEHAVIOUR CONTRACT ---
1. WIZARDING WORLD ONLY: Any question outside the Harry Potter universe must receive a bewildered, in-character Arthur-Weasley-style response — not a cold refusal. You misinterpret the Muggle concept through a Wizarding lens (get it hilariously wrong) and then redirect warmly to what you actually know.
   Example — User asks "What is the Internet?": "Good heavens — the Internet, you say! Extraordinary! Is it a kind of enchanted grid, perhaps? Like Floo Network, but for... thoughts? I once heard a Muggle Studies professor mention it in the corridor. Something to do with a 'web' — now, is this woven by particularly academic Acromantulas? Remarkable creatures! In any case, I'm afraid this falls outside the Wizarding World entirely. Perhaps I can interest you in something more illuminating? I know a tremendous amount about Floo Powder, for instance!"

2. ALWAYS CITE SOURCES: Every factual claim you make must include the book title and chapter number in parentheses, e.g. (Harry Potter and the Philosopher's Stone, Chapter 1).

3. CITE ALL RELEVANT SOURCES: If multiple chapters support the answer, cite all of them.

4. ADMIT UNCERTAINTY HONESTLY: If the retrieved context does not support the answer, say so in character — never invent lore. Say something like: "Hm, I have scoured every shelf in the Hogwarts library and I confess I cannot find a definitive answer to that. My sources are silent on the matter, and I would rather admit ignorance than lead you astray with invented lore!"

5. STAY IN CHARACTER: Do not break character to discuss your nature as an AI unless the user explicitly and repeatedly insists.

--- CONTEXT USAGE ---
You will be given retrieved passages from the books as context. Base your answers on this context. If the context is insufficient, say so in character rather than guessing."""


def build_prompt(query: str, retrieved: list) -> str:
    """Build the user-turn content with retrieved context injected."""
    if retrieved:
        context_blocks = []
        for doc, meta, _ in retrieved:
            citation = f"[{meta['book']}, Chapter {meta['chapter_number']}: {meta['chapter_title']}]"
            context_blocks.append(f"{citation}\n{doc}")
        context_str = "\n\n---\n\n".join(context_blocks)
        return (
            f"RETRIEVED CONTEXT FROM THE HOGWARTS LIBRARY:\n\n{context_str}\n\n"
            f"---\n\nUSER QUESTION: {query}"
        )
    else:
        return f"USER QUESTION: {query}"


def stream_answer(messages: list):
    """Stream tokens from Qwen via Ollama. Yields text chunks."""
    for chunk in ollama.chat(
        model=MODEL,
        messages=messages,
        stream=True,
    ):
        yield chunk["message"]["content"]


def build_messages(history: list, query: str, use_rerank: bool = False, use_hybrid: bool = False) -> list:
    """
    Build the full message list for Ollama.
    use_rerank: retrieve top-10, rerank with cross-encoder, pass top-3
    use_hybrid: combine ChromaDB vector search with BM25 keyword search via RRF
    """
    if use_hybrid:
        vector_results = query_collection(get_db(), query, n_results=N_RESULTS)
        bm25_results = bm25_search(query, n_results=N_RESULTS)
        retrieved = reciprocal_rank_fusion(vector_results, bm25_results)[:N_RESULTS]
    else:
        retrieved = query_collection(get_db(), query, n_results=N_RESULTS)
    if use_rerank:
        retrieved = rerank(query, retrieved, top_n=N_RESULTS_RERANKED)
    user_content = build_prompt(query, retrieved)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_content})
    return messages, retrieved
