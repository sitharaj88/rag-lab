"""
app.py — Streamlit chat interface for the Multi-PDF Research Assistant.

Run with:
    streamlit run app.py

Features
--------
* Persistent chat history within a session.
* Per-answer "Sources" expander showing file name, page number, similarity
  score, and a text snippet for every retrieved chunk.
* The RAGPipeline is cached so the embedding model loads only once.
"""

from __future__ import annotations

import streamlit as st

import config
from rag import RAGPipeline, RetrievedChunk

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Multi-PDF Research Assistant",
    page_icon="📚",
    layout="centered",
)

st.title("📚 Multi-PDF Research Assistant")
st.caption(
    "Ask questions across all your PDFs.  "
    "Answers are grounded in the documents — sources cited per file and page."
)

# ---------------------------------------------------------------------------
# Pipeline (cached so it loads only once per Streamlit session)
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading embedding model and connecting to ChromaDB …")
def load_pipeline() -> RAGPipeline | None:
    """Initialise the RAG pipeline once and cache it for the session.

    Returns None if the collection has not been created yet (i.e. the user
    has not run ingest.py), so the UI can show a helpful error message instead
    of crashing.
    """
    try:
        return RAGPipeline(
            embedding_model=config.EMBEDDING_MODEL,
            ollama_model=config.OLLAMA_MODEL,
            collection_name=config.COLLECTION_NAME,
            chroma_dir=str(config.CHROMA_DIR),
            top_k=config.TOP_K,
            temperature=config.OLLAMA_TEMPERATURE,
        )
    except Exception as exc:
        # Return the exception as a string so the UI can surface it.
        return str(exc)  # type: ignore[return-value]


pipeline_or_error = load_pipeline()

# If pipeline failed to load, show a setup guide and stop rendering.
if isinstance(pipeline_or_error, str):
    st.error(
        "**Could not connect to the vector store.**\n\n"
        f"```\n{pipeline_or_error}\n```\n\n"
        "**Quick fix:**\n"
        "1. Drop PDF files into the `pdfs/` folder.\n"
        "2. Run `python ingest.py` to build the index.\n"
        "3. Refresh this page."
    )
    st.stop()

pipeline: RAGPipeline = pipeline_or_error  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Session state — chat history
# ---------------------------------------------------------------------------

# Each entry is a dict:  {"role": "user"|"assistant", "content": str,
#                          "sources": list[RetrievedChunk] | None}
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# ---------------------------------------------------------------------------
# Render existing chat history
# ---------------------------------------------------------------------------


def render_sources(sources: list[RetrievedChunk]) -> None:
    """Render a collapsible sources panel beneath an assistant message."""
    if not sources:
        return

    with st.expander(f"📎 Sources ({len(sources)} chunks retrieved)", expanded=False):
        # Deduplicate by (source, page) for the summary line, but still show
        # individual chunk snippets below.
        for i, chunk in enumerate(sources, start=1):
            citation = f"**{chunk.source}** (p.{chunk.page})"
            similarity = f"similarity {chunk.score:.2f}"
            snippet = chunk.text[:280].replace("\n", " ")
            if len(chunk.text) > 280:
                snippet += " …"

            st.markdown(
                f"{i}. {citation} — {similarity}  \n"
                f"<small>{snippet}</small>",
                unsafe_allow_html=True,
            )


for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            render_sources(message["sources"])

# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------

user_query = st.chat_input("Ask a question about your documents …")

if user_query:
    # ---- show user message immediately ----
    st.session_state["messages"].append(
        {"role": "user", "content": user_query, "sources": None}
    )
    with st.chat_message("user"):
        st.markdown(user_query)

    # ---- run RAG pipeline ----
    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating answer …"):
            answer_text, sources = pipeline.answer(user_query)

        st.markdown(answer_text)
        render_sources(sources)

    # ---- store assistant message in history ----
    st.session_state["messages"].append(
        {"role": "assistant", "content": answer_text, "sources": sources}
    )
