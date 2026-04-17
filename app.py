"""
app.py — Streamlit chat UI for the Know-it Owl.

Launch:
    conda activate tf && streamlit run app.py
"""
import streamlit as st
from rag import build_messages, stream_answer
import database

database.init_db()

st.set_page_config(
    page_title="The Know-it Owl",
    page_icon="🦉",
    layout="centered",
)

st.title("🦉 The Know-it Owl")
st.caption(
    "An ancient, enchanted owl of extraordinary erudition — "
    "roosting in the Hogwarts Owlery since time immemorial."
)
st.info(
    "Ask me anything about the Wizarding World! "
    "Every answer cites the book and chapter it comes from.",
    icon="📚",
)


def _show_citations(retrieved, mode: str = "vector"):
    if not retrieved:
        return
    mode_label = {
        "vector": "vector similarity",
        "hybrid": "RRF score",
        "rerank": "cross-encoder score",
        "hybrid+rerank": "cross-encoder score (hybrid)",
    }
    with st.expander("📖 Sources retrieved from the library", expanded=False):
        st.caption(f"Retrieval mode: **{mode}**")
        for doc, meta, score in retrieved:
            if mode == "vector":
                score_str = f"{mode_label[mode]}: {round(1 - float(score), 3)}"
            else:
                score_str = f"{mode_label[mode]}: {round(float(score), 4)}"
            ref = f"**{meta['book']}** — Chapter {meta['chapter_number']}: *{meta['chapter_title']}* ({score_str})"
            st.markdown(ref)
            st.caption(doc)
            st.divider()


# Session state
if "history" not in st.session_state:
    st.session_state.history = []
if "citations" not in st.session_state:
    st.session_state.citations = []
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = database.create_conversation()

# Render prior messages
for i, msg in enumerate(st.session_state.history):
    with st.chat_message(msg["role"], avatar="🦉" if msg["role"] == "assistant" else "🧙"):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and i // 2 < len(st.session_state.citations):
            entry = st.session_state.citations[i // 2]
            _show_citations(entry["retrieved"], mode=entry["mode"])

use_rerank = st.sidebar.toggle("✨ Reranking (top-10 → top-3)", value=False,
                               help="Retrieve 10 chunks, rerank with a cross-encoder, pass top-3 to Qwen")
use_hybrid = st.sidebar.toggle("🔀 Hybrid search (vector + BM25)", value=False,
                                help="Combine ChromaDB vector search with BM25 keyword search via Reciprocal Rank Fusion")

# Chat input
if prompt := st.chat_input("Ask the owl…"):
    with st.chat_message("user", avatar="🧙"):
        st.markdown(prompt)

    if use_hybrid and use_rerank:
        mode = "hybrid+rerank"
    elif use_hybrid:
        mode = "hybrid"
    elif use_rerank:
        mode = "rerank"
    else:
        mode = "vector"

    messages, retrieved = build_messages(st.session_state.history, prompt, use_rerank=use_rerank, use_hybrid=use_hybrid)

    with st.chat_message("assistant", avatar="🦉"):
        response_placeholder = st.empty()
        full_response = ""
        for chunk in stream_answer(messages):
            full_response += chunk
            response_placeholder.markdown(full_response + "▌")
        response_placeholder.markdown(full_response)
        _show_citations(retrieved, mode=mode)

    # Persist to DB
    conv_id = st.session_state.conversation_id
    database.save_message(conv_id, "user", prompt)
    database.save_message(conv_id, "assistant", full_response)

    # Auto-title the conversation from the first question
    if len(st.session_state.history) == 0:
        database.set_title(conv_id, prompt)

    # Update session state
    st.session_state.history.append({"role": "user", "content": prompt})
    st.session_state.history.append({"role": "assistant", "content": full_response})
    st.session_state.citations.append({"retrieved": retrieved, "mode": mode})

# Sidebar
with st.sidebar:
    st.header("🦉 Know-it Owl")
    st.markdown(
        "**Model:** qwen2.5:14b via Ollama  \n"
        "**Retrieval:** ChromaDB + sentence-transformers  \n"
        "**Corpus:** HP Books 1–7 + companion texts"
    )
    st.divider()

    tab_chat, tab_spells = st.tabs(["💬 Conversations", "✨ Spell Encyclopaedia"])

    with tab_spells:
        spell_count = database.spells_count()
        if spell_count == 0:
            st.info("No spells indexed yet. Run `python build_spells.py` to populate the encyclopaedia.")
        else:
            st.caption(f"{spell_count} spells in the library")
            search = st.text_input("Search spells…", key="spell_search")
            spells = database.list_spells(search)
            if not spells:
                st.write("No spells found.")
            for spell in spells:
                with st.expander(f"**{spell['name']}**"):
                    st.markdown(f"**Effect:** {spell['effect']}")
                    st.caption(f"{spell['book']} — Chapter {spell['chapter_number']}: {spell['chapter_title']}")

    with tab_chat:
        if st.button("➕ New conversation"):
            st.session_state.history = []
            st.session_state.citations = []
            st.session_state.conversation_id = database.create_conversation()
            st.rerun()

        if st.button("🗑️ Clear conversation"):
            database.delete_conversation(st.session_state.conversation_id)
            st.session_state.history = []
            st.session_state.citations = []
            st.session_state.conversation_id = database.create_conversation()
            st.rerun()

        st.divider()
        st.subheader("Past conversations")
        if "editing_title" not in st.session_state:
            st.session_state.editing_title = None

        for conv in database.list_conversations():
            if st.session_state.editing_title == conv["id"]:
                new_title = st.text_input("", value=conv["title"], key=f"title_input_{conv['id']}")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Save", key=f"save_{conv['id']}"):
                        database.set_title(conv["id"], new_title)
                        st.session_state.editing_title = None
                        st.rerun()
                with col2:
                    if st.button("Cancel", key=f"cancel_{conv['id']}"):
                        st.session_state.editing_title = None
                        st.rerun()
            else:
                col1, col2, col3 = st.columns([4, 1, 1])
                with col1:
                    if st.button(conv["title"], key=f"conv_{conv['id']}"):
                        msgs = database.load_conversation(conv["id"])
                        st.session_state.history = msgs
                        st.session_state.citations = [{"retrieved": [], "mode": "vector"} for _ in range(len(msgs) // 2)]
                        st.session_state.conversation_id = conv["id"]
                        st.rerun()
                with col2:
                    if st.button("✏️", key=f"edit_{conv['id']}"):
                        st.session_state.editing_title = conv["id"]
                        st.rerun()
                with col3:
                    if st.button("🗑", key=f"del_{conv['id']}"):
                        database.delete_conversation(conv["id"])
                        if st.session_state.conversation_id == conv["id"]:
                            st.session_state.history = []
                            st.session_state.citations = []
                            st.session_state.conversation_id = database.create_conversation()
                        st.rerun()

        st.divider()
        st.caption("All processing runs locally. No data sent externally.")
