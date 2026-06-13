"""Streamlit chat UI for the local RAG pipeline.

Run:
    streamlit run app.py

Make sure you've run `python ingest.py` and that Ollama is running
(`ollama serve`) with the model pulled (`ollama pull llama3.2`).
"""
from __future__ import annotations

import streamlit as st

import config
from rag import RAGPipeline

st.set_page_config(page_title="Local RAG — RAG Lab", page_icon="🧪")


@st.cache_resource(show_spinner="Loading embedding model and vector store…")
def get_pipeline() -> RAGPipeline:
    return RAGPipeline()


st.title("🧪 Local RAG")
st.caption(
    f"Embeddings: `{config.EMBEDDING_MODEL.split('/')[-1]}` · "
    f"LLM: `{config.OLLAMA_MODEL}` (Ollama) · Store: ChromaDB"
)

pipeline = get_pipeline()
count = pipeline.collection.count()

if count == 0:
    st.warning("No documents indexed yet. Add files to `data/` and run `python ingest.py`.")
else:
    st.success(f"Knowledge base ready — {count} chunks indexed.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(f"**{s.source}** · similarity {s.score:.2f}")
                    st.caption(s.text[:300] + ("…" if len(s.text) > 300 else ""))

if prompt := st.chat_input("Ask a question about your documents…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating…"):
            answer, sources = pipeline.answer(prompt)
        st.markdown(answer)
        if sources:
            with st.expander("Sources"):
                for s in sources:
                    st.markdown(f"**{s.source}** · similarity {s.score:.2f}")
                    st.caption(s.text[:300] + ("…" if len(s.text) > 300 else ""))

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
