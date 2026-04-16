"""
build_spells.py — Extract spells from the indexed ChromaDB chunks and store in SQLite.

Run after index.py:
    conda activate tf && python build_spells.py
"""
import json
import re
import ollama
from store import get_collection, query_collection
import database

MODEL = "qwen2.5:14b"
BATCH_SIZE = 5  # chunks per Qwen call

EXTRACTION_QUERIES = [
    "spell incantation magic effect charm",
    "hex jinx curse dark magic",
    "Expelliarmus Lumos Accio Wingardium Leviosa",
    "spell casting wand movement pronunciation",
    "Unforgivable Curses Avada Kedavra Crucio Imperio",
    "transfiguration spell conjuring summoning",
    "shield charm protection defensive spell",
    "healing spell potion magical remedy",
]

EXTRACT_PROMPT = """You are a precise magical archivist. From the passages below, extract every spell, charm, hex, jinx, or curse that appears.

For each spell return a JSON object with:
- "name": the incantation or spell name (e.g. "Expelliarmus", "Wingardium Leviosa")
- "effect": one sentence describing what it does
- "book": exactly as given in the passage source
- "chapter_number": integer
- "chapter_title": as given in the passage source

Return ONLY a JSON array. If no spells appear, return [].

Passages:
{passages}
"""


def extract_spells_from_chunks(chunks: list) -> list:
    passages = []
    for doc, meta, _ in chunks:
        label = f"[{meta['book']}, Chapter {meta['chapter_number']}: {meta['chapter_title']}]"
        passages.append(f"{label}\n{doc}")

    prompt = EXTRACT_PROMPT.format(passages="\n\n---\n\n".join(passages))
    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0},
    )
    raw = response["message"]["content"].strip()

    match = re.search(r'\[.*\]', raw, re.DOTALL)
    if not match:
        return []
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return []


def main():
    database.init_db()
    collection = get_collection()

    print("=== Know-it Owl — Spell Extraction ===\n")

    seen_chunks = set()
    all_chunks = []

    print("Retrieving spell-related chunks from ChromaDB…")
    for query in EXTRACTION_QUERIES:
        results = query_collection(collection, query, n_results=20)
        for doc, meta, dist in results:
            key = f"{meta['book']}_{meta['chapter_number']}_{meta['chunk_index']}"
            if key not in seen_chunks:
                seen_chunks.add(key)
                all_chunks.append((doc, meta, dist))

    print(f"Found {len(all_chunks)} unique relevant chunks.\n")

    total_spells = 0
    for i in range(0, len(all_chunks), BATCH_SIZE):
        batch = all_chunks[i:i + BATCH_SIZE]
        print(f"Processing chunks {i + 1}–{min(i + BATCH_SIZE, len(all_chunks))}…", end=" ")
        spells = extract_spells_from_chunks(batch)
        for spell in spells:
            try:
                database.save_spell(
                    name=spell["name"],
                    effect=spell["effect"],
                    book=spell["book"],
                    chapter_number=int(spell["chapter_number"]),
                    chapter_title=spell["chapter_title"],
                )
                total_spells += 1
            except (KeyError, ValueError):
                continue
        print(f"extracted {len(spells)} spells.")

    final_count = database.spells_count()
    print(f"\nDone! {final_count} unique spells stored in the encyclopaedia.")


if __name__ == "__main__":
    main()
