"""
loader.py — Parse HP .txt files into chapters, then chunk them.
"""
import re
from dataclasses import dataclass
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter

NOVELS = {
    "HP1.txt": "Harry Potter and the Philosopher's Stone",
    "HP2.txt": "Harry Potter and the Chamber of Secrets",
    "HP3.txt": "Harry Potter and the Prisoner of Azkaban",
    "HP4.txt": "Harry Potter and the Goblet of Fire",
    "HP5.txt": "Harry Potter and the Order of the Phoenix",
    "HP6.txt": "Harry Potter and the Half-Blood Prince",
    "HP7.txt": "Harry Potter and the Deathly Hallows",
}

COMPANIONS = {
    "Fantastic Beasts and Where to Find Them_ The Original Screenplay - J. K. Rowling.txt": "Fantastic Beasts and Where to Find Them",
    "Quidditch Through the Ages - J. K. Rowling.txt": "Quidditch Through the Ages",
    "Tales of Beedle the Bard, The - J.K. Rowling.txt": "The Tales of Beedle the Bard",
    "Harry Potter and the Cursed Child - Parts One and Two - J.K. Rowling & Jack Thorne & John Tiffany.txt": "Harry Potter and the Cursed Child",
}

BOOK_TITLES = {**NOVELS, **COMPANIONS}

CHAPTER_PATTERN = re.compile(
    r'(?i)^[\–\—\-\s]*chapter[\s]+(\w+)[\–\—\-\s:]*(.*)$', re.MULTILINE
)

ROMAN = {
    'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5,
    'VI': 6, 'VII': 7, 'VIII': 8, 'IX': 9, 'X': 10,
    'XI': 11, 'XII': 12, 'XIII': 13, 'XIV': 14, 'XV': 15,
    'XVI': 16, 'XVII': 17, 'XVIII': 18, 'XIX': 19, 'XX': 20,
    'XXI': 21, 'XXII': 22, 'XXIII': 23, 'XXIV': 24, 'XXV': 25,
    'XXVI': 26, 'XXVII': 27, 'XXVIII': 28, 'XXIX': 29, 'XXX': 30,
    'XXXI': 31, 'XXXII': 32, 'XXXIII': 33, 'XXXIV': 34, 'XXXV': 35,
    'XXXVI': 36, 'XXXVII': 37, 'ONE': 1, 'TWO': 2, 'THREE': 3,
    'FOUR': 4, 'FIVE': 5, 'SIX': 6, 'SEVEN': 7, 'EIGHT': 8,
    'NINE': 9, 'TEN': 10, 'ELEVEN': 11, 'TWELVE': 12, 'THIRTEEN': 13,
    'FOURTEEN': 14, 'FIFTEEN': 15, 'SIXTEEN': 16, 'SEVENTEEN': 17,
    'EIGHTEEN': 18, 'NINETEEN': 19, 'TWENTY': 20,
}


@dataclass
class TextChunk:
    text: str
    book: str
    chapter_number: int
    chapter_title: str


def _parse_chapter_num(s: str, fallback: int) -> int:
    s = s.strip().upper()
    try:
        return int(s)
    except ValueError:
        return ROMAN.get(s, fallback)


def parse_book(filepath: str, book_title: str) -> List[TextChunk]:
    """Split a book into chapters and return labelled TextChunks."""
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    matches = list(CHAPTER_PATTERN.finditer(text))
    if not matches:
        # No chapters found — treat whole book as one chunk
        return [TextChunk(text=text.strip(), book=book_title,
                          chapter_number=0, chapter_title="(full text)")]

    chunks = []
    for i, match in enumerate(matches):
        num_str = match.group(1).strip()
        title = match.group(2).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chapter_text = text[start:end].strip()
        if not chapter_text:
            continue
        chapter_num = _parse_chapter_num(num_str, fallback=i + 1)
        chunks.append(TextChunk(
            text=chapter_text,
            book=book_title,
            chapter_number=chapter_num,
            chapter_title=title,
        ))
    return chunks


def _load_books(data_dir: str, titles: dict) -> List[TextChunk]:
    import os
    all_chunks = []
    for filename, title in titles.items():
        path = os.path.join(data_dir, filename)
        if not os.path.exists(path):
            print(f"  [skip] not found: {path}")
            continue
        print(f"  Loading: {title}")
        chapters = parse_book(path, title)
        all_chunks.extend(chapters)
        print(f"    → {len(chapters)} chapters")
    print(f"Total chapters parsed: {len(all_chunks)}")
    return all_chunks


def load_all_books(data_dir: str = "data/") -> List[TextChunk]:
    """Load full corpus: 7 novels + companion texts."""
    return _load_books(data_dir, BOOK_TITLES)


def load_novels_only(data_dir: str = "data/") -> List[TextChunk]:
    """Load 7 novels only (no companion texts)."""
    return _load_books(data_dir, NOVELS)


def chunk_book_chapters(chapters: List[TextChunk], chunk_size: int = 500, chunk_overlap: int = 50):
    """Split chapters into smaller overlapping chunks, preserving metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " "],
    )
    all_docs, all_metas, all_ids = [], [], []
    seen_ids: dict = {}
    for ch in chapters:
        sub_chunks = splitter.split_text(ch.text)
        safe_book = ch.book.replace(" ", "_").replace("'", "")
        for j, sub in enumerate(sub_chunks):
            base_id = f"{safe_book}_ch{ch.chapter_number}_p{j}"
            # Deduplicate: if the same ID appeared before, append a counter
            count = seen_ids.get(base_id, 0)
            seen_ids[base_id] = count + 1
            uid = base_id if count == 0 else f"{base_id}_dup{count}"
            all_docs.append(sub)
            all_metas.append({
                "book": ch.book,
                "chapter_number": ch.chapter_number,
                "chapter_title": ch.chapter_title,
                "chunk_index": j,
            })
            all_ids.append(uid)
    print(f"Total chunks created: {len(all_docs)}")
    return all_docs, all_metas, all_ids
