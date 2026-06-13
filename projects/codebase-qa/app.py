# app.py — Streamlit chat interface for Codebase Q&A.
#
# Run with:
#   streamlit run app.py
#
# The RAGPipeline is cached so the embedding model and ChromaDB connection
# are only loaded once across all interactions.

from __future__ import annotations

from typing import List, Tuple

import streamlit as st

import config
from rag import RAGPipeline, RetrievedChunk


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Codebase Q&A",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 Codebase Q&A")
st.caption(
    f"Ask questions about your indexed codebase · "
    f"Model: `{config.OLLAMA_MODEL}` · "
    f"Top-K: `{config.TOP_K}`"
)


# ---------------------------------------------------------------------------
# Cached pipeline (loaded once per session)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading RAG pipeline …")
def get_pipeline() -> RAGPipeline:
    """Instantiate and cache the RAGPipeline across Streamlit reruns."""
    return RAGPipeline()


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

# chat_history: list of (role, content, sources) tuples
#   role    : "user" | "assistant"
#   content : message text
#   sources : list[RetrievedChunk] for assistant messages, [] for user
if "chat_history" not in st.session_state:
    st.session_state.chat_history: List[Tuple[str, str, List[RetrievedChunk]]] = []


# ---------------------------------------------------------------------------
# Render chat history
# ---------------------------------------------------------------------------

for role, content, sources in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(content)

        if role == "assistant" and sources:
            with st.expander(f"📎 {len(sources)} source(s)"):
                for src in sources:
                    citation = f"{src['path']}:{src['start_line']}-{src['end_line']}"
                    st.markdown(f"**`{citation}`** — score `{src['score']}`")
                    # Show a snippet of the retrieved code
                    snippet = src["text"][:600]
                    if len(src["text"]) > 600:
                        snippet += "\n… (truncated)"
                    st.code(snippet, language="python")


# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------

if prompt := st.chat_input("Ask a question about the codebase …"):
    # Display the user message immediately
    with st.chat_message("user"):
        st.markdown(prompt)

    # Store user turn (no sources)
    st.session_state.chat_history.append(("user", prompt, []))

    # Generate the answer
    pipeline = get_pipeline()
    with st.chat_message("assistant"):
        with st.spinner("Thinking …"):
            answer_text, sources = pipeline.answer(prompt)

        st.markdown(answer_text)

        if sources:
            with st.expander(f"📎 {len(sources)} source(s)"):
                for src in sources:
                    citation = f"{src['path']}:{src['start_line']}-{src['end_line']}"
                    st.markdown(f"**`{citation}`** — score `{src['score']}`")
                    snippet = src["text"][:600]
                    if len(src["text"]) > 600:
                        snippet += "\n… (truncated)"
                    st.code(snippet, language="python")

    # Store assistant turn with sources for history replay
    st.session_state.chat_history.append(("assistant", answer_text, sources))


# ---------------------------------------------------------------------------
# Sidebar: controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Settings")
    st.markdown(f"**Collection:** `{config.COLLECTION_NAME}`")
    st.markdown(f"**Chroma dir:** `{config.CHROMA_DIR}`")
    st.markdown(f"**Embedding model:** `{config.EMBEDDING_MODEL}`")
    st.markdown(f"**LLM:** `{config.OLLAMA_MODEL}` (temp={config.OLLAMA_TEMPERATURE})")

    st.divider()

    if st.button("🗑️ Clear chat history"):
        st.session_state.chat_history = []
        st.rerun()

    st.divider()
    st.markdown(
        "**Need to (re)index?**\n\n"
        "```bash\n"
        "python ingest.py --path /your/repo --reset\n"
        "```"
    )
