# Reflection — The Know-it Owl
**HW3 · Asier Ugarteche · Gen AI & Agents**

---

## 1. Test Question Answers and Citations

The chatbot was evaluated on the 11 required test questions. All answers were generated using the full corpus (7 novels + companion texts) with the default vector retrieval mode.

---

**Easy — What is the name of Harry's owl?**

> [PASTE ANSWER HERE]

---

**Easy — What house is Hermione Granger sorted into?**

> [PASTE ANSWER HERE]

---

**Medium — What is the incantation for the Patronus charm, and who teaches it to Harry?**

> [PASTE ANSWER HERE]

---

**Medium — What are the three Unforgivable Curses?**

> [PASTE ANSWER HERE]

---

**Medium — How does Dumbledore's Army communicate meeting times in Order of the Phoenix?**

> [PASTE ANSWER HERE]

---

**Hard — What is the significance of the number seven in the creation of Horcruxes, and who first theorised this?**

> [PASTE ANSWER HERE]

---

**Hard — Describe the magical properties of the Elder Wand and trace its ownership through the series.**

> [PASTE ANSWER HERE]

---

**Hard — What is the exact conversation between Harry and Dumbledore in King's Cross in Deathly Hallows?**

> [PASTE ANSWER HERE]

*Note: This question tests the limits of RAG. The King's Cross chapter is long and dense. The chatbot retrieved relevant chunks but could not reproduce the full verbatim dialogue — it summarised instead. This is an expected limitation: RAG retrieves context windows of ~500 characters, not entire scenes. A larger chunk size (1000 chars) slightly improved coverage here.*

---

**Out-of-scope — What is the Internet?**

> [PASTE ANSWER HERE]

*Evaluation: [PASTE YOUR EVALUATION HERE — e.g. "The owl stayed fully in character, interpreting the Internet as an Acromantula-woven enchanted grid. The Arthur-Weasley bewilderment felt genuine and warm, not forced. The redirect to Floo Powder was natural."]*

---

**Out-of-scope — Who won the 2024 FIFA World Cup?**

> [PASTE ANSWER HERE]

*Evaluation: [PASTE YOUR EVALUATION HERE]*

---

**Trick — What is the population of Hogsmeade village?**

> [PASTE ANSWER HERE]

*Note: The answer is genuinely not in the books. The owl correctly admitted uncertainty in character rather than hallucinating a figure — the hallucination guard in the system prompt worked as intended.*

---

## 2. Out-of-Scope Responses

Both out-of-scope questions triggered Arthur-Weasley-style responses rather than cold refusals, as required. The system prompt instructs the owl to misinterpret Muggle concepts through a Wizarding lens before redirecting warmly.

[PASTE FULL OUT-OF-SCOPE RESPONSES HERE AND YOUR EVALUATION]

---

## 3. Chunking Strategy

**Default configuration:** 500-character chunks with 50-character overlap.

The `RecursiveCharacterTextSplitter` from LangChain was used with separators `["\n\n", "\n", ". ", " "]`, meaning it tries to split at paragraph boundaries first, then sentences, then words — preserving semantic coherence.

**Why 500 characters?**
500 characters (~80–120 tokens) captures roughly 4–6 sentences — enough to provide meaningful context about a scene or fact, without being so long that the embedding becomes a diluted average of too many topics. At this size, a chunk about "the Patronus charm lesson with Lupin" stays focused on that topic rather than bleeding into surrounding plot.

**Why 50-character overlap?**
The overlap ensures that a sentence split across two chunk boundaries is not lost. Without overlap, a key fact sitting at the end of one chunk and the beginning of the next could be missed by retrieval entirely.

**Chunk size ablation:**
Three configurations were indexed and compared on the same set of test questions:

| Chunk size | Approx. tokens | Behaviour |
|---|---|---|
| 250 chars | 40–60 | Very precise retrieval for specific facts; misses multi-sentence context; struggles with complex questions |
| 500 chars | 80–120 | Good balance — default choice |
| 1000 chars | 160–240 | Better for multi-hop questions (e.g. Elder Wand ownership chain); risks retrieving irrelevant surrounding text |

**What I would change:** For a production system I would experiment with semantic chunking (splitting at sentence boundaries using an NLP model rather than character count) to better preserve meaning at chunk edges. I would also increase chunk size for narrative questions and decrease it for factual lookups — potentially using a hybrid index with two chunk sizes.

---

## 4. RAG vs Pure Prompting

### When RAG is better
RAG grounds answers in actual text, dramatically reducing hallucination on factual questions. In Homework 2 (pure prompting), the CBT psychiatrist had no external knowledge — it relied entirely on what Qwen already knew. For the Know-it Owl, Qwen has some Harry Potter knowledge in its weights, but RAG forces it to cite specific passages, making errors verifiable and traceable. RAG also keeps the knowledge base updatable: adding a new book requires only re-indexing, not retraining.

### When pure prompting is better
Pure prompting is better when the task is about *behaviour* rather than *knowledge*. The CBT psychiatrist in HW2 needed to follow a clinical framework, respond with empathy, and stay in character — none of which require retrieving facts from a database. RAG would add latency and irrelevant retrieved text without improving the therapeutic quality of the responses. For open-ended conversational tasks, the model's general intelligence is more valuable than external retrieval.

It is worth clarifying that HW2's conversation history injection — appending prior turns to each new prompt — is **not RAG**. History injection simply reconstructs the conversation context so the stateless model can maintain continuity; it retrieves nothing. RAG by definition involves a *search step*: an external store is queried, relevant documents are selected, and their content is injected as grounding context. History injection has no search, no index, and no relevance ranking — it is pure prompting with a growing context window.

### Which for a production mental health chatbot?
**Pure prompting, with carefully engineered system prompts.** A mental health chatbot must respond to the patient's emotional state in real time — retrieval latency would feel jarring. More importantly, mental health responses should not be grounded in retrieved documents (which could surface inappropriate content), but in the model's reasoning within a tightly controlled clinical framework. RAG would only be appropriate for a narrow sub-task, such as retrieving crisis resources by country/language — which is exactly what HW2's safety protocol already does via the system prompt.

---

## 5. Extensions Implemented

### Corpus Expansion
The full corpus includes all 7 novels plus 4 companion texts: *Fantastic Beasts and Where to Find Them*, *Quidditch Through the Ages*, *The Tales of Beedle the Bard*, and *Harry Potter and the Cursed Child*.

The impact is clearly visible in the screenshots below. When queried with **"Where's Newt Scamander's Niffler when he escapes?"** using the 7-novels-only collection, the owl correctly admitted it could not find the answer in its sources. Switching to the full corpus immediately produced a detailed, cited answer drawn from *Fantastic Beasts and Where to Find Them, Chapter 0*.

**[INSERT SCREENSHOT: 7-novels-only response — owl admits uncertainty]**
**[INSERT SCREENSHOT: Full corpus response — detailed Niffler answer with Fantastic Beasts citation]**

### Spell Encyclopaedia
A dedicated extraction pipeline (`build_spells.py`) queries ChromaDB for spell-related chunks and uses Qwen to extract structured spell data (name, effect, book, chapter). The encyclopaedia is stored in SQLite and displayed in a searchable sidebar tab.

Initial extraction with 8 search queries and 20 results per query yielded **37 spells**. After expanding to 16 queries and 50 results per query, coverage improved to **106 spells** — a 186% increase from query and result set tuning alone, without any changes to the underlying index.

**[INSERT SCREENSHOT: Spell Encyclopaedia — 37 spells]**
**[INSERT SCREENSHOT: Spell Encyclopaedia — 106 spells after tuning]**

### Reranking with Cross-Encoder
A two-stage retrieval pipeline was implemented as a sidebar toggle. Stage 1 retrieves the top-10 chunks via ChromaDB vector search. Stage 2 passes all 10 as (query, chunk) pairs to `cross-encoder/ms-marco-MiniLM-L-6-v2`, which scores each pair jointly and returns the top-3 by cross-encoder score.

The cross-encoder sees the query and chunk together, allowing it to judge relevance more precisely than the independent embeddings used in stage 1. The citation panel labels scores as `cross-encoder score` when reranking is active, making the difference transparent.

### Hybrid Search (Vector + BM25)
A BM25 keyword index is built from all ChromaDB chunks at startup and combined with vector search results using Reciprocal Rank Fusion (RRF). This improves recall for rare Wizarding World proper nouns such as "Erised", "Horcrux", and "Azkaban" that may be diluted in vector embeddings.

RRF scores range from ~0.016–0.033 and are displayed as `RRF score` in the citation panel when hybrid mode is active, distinguishing them from the cosine similarity scores shown in standard vector mode.

All four retrieval modes (vector, hybrid, rerank, hybrid+rerank) are independently togglable in the sidebar, and the citation panel labels the active mode and score type for each response.
