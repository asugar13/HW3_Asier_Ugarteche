# The Know-it Owl
A Harry Potter RAG chatbot built with ChromaDB, sentence-transformers, and Qwen via Ollama.

## Prerequisites
- [Ollama](https://ollama.com) running locally with `qwen2.5:7b` pulled
- Conda environment with dependencies installed (see below)

## Setup

### 1. Install dependencies
```bash
conda activate tf
pip install -r requirements.txt
```

### 2. Place the book files
Put the Harry Potter `.txt` files in the `data/` directory:
```
data/
├── HP1.txt   (Harry Potter and the Philosopher's Stone)
├── HP2.txt   (Harry Potter and the Chamber of Secrets)
├── HP3.txt   (Harry Potter and the Prisoner of Azkaban)
├── HP4.txt   (Harry Potter and the Goblet of Fire)
├── HP5.txt   (Harry Potter and the Order of the Phoenix)
├── HP6.txt   (Harry Potter and the Half-Blood Prince)
└── HP7.txt   (Harry Potter and the Deathly Hallows)
```
Companion texts (optional, for better coverage):
```
data/
├── Fantastic Beasts and Where to Find Them_ The Original Screenplay - J. K. Rowling.txt
├── Quidditch Through the Ages - J. K. Rowling.txt
├── Tales of Beedle the Bard, The - J.K. Rowling.txt
└── Harry Potter and the Cursed Child - Parts One and Two - J.K. Rowling & Jack Thorne & John Tiffany.txt
```

### 3. Build the index (run once)
```bash
conda activate tf
python index.py
```
This will parse all books, chunk the text, embed the chunks using `all-MiniLM-L6-v2`, and store everything in `owl_db/`. Takes ~10–20 minutes on a CPU. The index persists across runs — you only need to do this once.

To use a custom data directory:
```bash
python index.py --data-dir /path/to/books
```

### 4. Launch the app
```bash
conda activate tf
streamlit run app.py
```
The app opens automatically at `http://localhost:8501`.

## Project structure
```
HW3_Asier_Ugarteche/
├── data/           # HP .txt book files (not included in repo)
├── owl_db/         # ChromaDB vector store (auto-created, not in repo)
├── loader.py       # Book parsing and chunking
├── store.py        # ChromaDB interface
├── index.py        # Run once to build the vector store
├── rag.py          # Retrieval + prompt assembly + Qwen generation
├── app.py          # Streamlit chat UI
├── requirements.txt
└── README.md
```

## How it works
1. **Indexing phase** (`index.py`): Books are parsed into chapters, split into 500-character overlapping chunks, embedded with `sentence-transformers`, and stored in ChromaDB.
2. **Query phase** (per message): The user's question is embedded and matched against stored chunks via cosine similarity. The top-5 most relevant chunks are injected into the prompt as context, and Qwen generates a cited answer.

## Notes
- `owl_db/` is excluded from git (can be several hundred MB)
- The book `.txt` files are also excluded from git
- Ollama must be running before launching the app (`ollama serve`)
