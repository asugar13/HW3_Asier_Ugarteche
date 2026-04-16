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


def _show_citations(retrieved):
    if not retrieved:
        return
    with st.expander("📖 Sources retrieved from the library", expanded=False):
        seen = set()
        for _, meta, dist in retrieved:
            ref = f"**{meta['book']}** — Chapter {meta['chapter_number']}: *{meta['chapter_title']}*"
            if ref not in seen:
                seen.add(ref)
                score = round(1 - dist, 3)
                st.markdown(f"- {ref} (similarity: {score})")


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
            _show_citations(st.session_state.citations[i // 2])

# Chat input
if prompt := st.chat_input("Ask the owl…"):
    with st.chat_message("user", avatar="🧙"):
        st.markdown(prompt)

    messages, retrieved = build_messages(st.session_state.history, prompt)

    with st.chat_message("assistant", avatar="🦉"):
        response_placeholder = st.empty()
        full_response = ""
        for chunk in stream_answer(messages):
            full_response += chunk
            response_placeholder.markdown(full_response + "▌")
        response_placeholder.markdown(full_response)
        _show_citations(retrieved)

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
    st.session_state.citations.append(retrieved)

# Sidebar
with st.sidebar:
    st.header("🦉 Know-it Owl")
    st.markdown(
        "**Model:** qwen2.5:14b via Ollama  \n"
        "**Retrieval:** ChromaDB + sentence-transformers  \n"
        "**Corpus:** HP Books 1–7 + companion texts"
    )
    st.divider()

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
                    st.session_state.citations = [[] for _ in range(len(msgs) // 2)]
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
