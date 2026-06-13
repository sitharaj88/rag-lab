# Streamlit Chat App

Wrap the ChromaDB RAG pipeline in a browser-based chat interface using Streamlit. Users can ask questions in a familiar chat window, session history is preserved across turns, and retrieved source chunks are shown in a collapsible expander.

## What you'll learn

- How to build a persistent chat UI with `st.chat_input` and `st.chat_message`
- How to store conversation history in `st.session_state`
- How to load the RAG pipeline once with `@st.cache_resource`
- How to surface retrieved source chunks alongside answers
- How to run a Streamlit app from the command line

## Prerequisites

- [Environment Setup](../getting-started/environment-setup.md)
- [Running a Local LLM with Ollama](../getting-started/local-llm-ollama.md)
- Completed [Tutorial 02 — RAG with Ollama + ChromaDB](02-rag-with-ollama-chroma.md) (the ChromaDB store must already be populated)
- `pip install streamlit sentence-transformers chromadb ollama`

## Step 1 — Project layout

This tutorial adds a single file to the `rag_chroma/` directory from Tutorial 02:

```
rag_chroma/
├── chroma_db/      # created by ingest.py in Tutorial 02
├── ingest.py       # unchanged from Tutorial 02
├── query.py        # unchanged from Tutorial 02
└── app.py          # NEW — Streamlit chat UI
```

If you have not yet run `ingest.py` from Tutorial 02, do that first so the `chroma_db/` directory exists.

## Step 2 — Write the Streamlit app

```python
# app.py
# Streamlit chat UI wrapping the ChromaDB RAG pipeline.
#
# Run with:  streamlit run app.py

import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import ollama

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "rag_demo"
EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama3.2"
TOP_K = 3

# ---------------------------------------------------------------------------
# Page config — must be the first Streamlit call.
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="RAG Chat",
    page_icon="🔍",
    layout="centered",
)


# ---------------------------------------------------------------------------
# Load the RAG pipeline once per session.
#
# @st.cache_resource runs the function exactly once and caches the return
# value for all subsequent reruns. This prevents the model and DB client
# from being re-initialised on every user message.
# ---------------------------------------------------------------------------
@st.cache_resource
def load_pipeline():
    """Initialise ChromaDB client, collection, and embedding model."""
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(COLLECTION_NAME)
    embedder = SentenceTransformer(EMBED_MODEL)
    return collection, embedder


# ---------------------------------------------------------------------------
# RAG helpers
# ---------------------------------------------------------------------------
def retrieve(query: str, collection, embedder, top_k: int = TOP_K) -> list[dict]:
    """Return the top_k chunks most relevant to the query."""
    query_embedding = embedder.encode([query], normalize_embeddings=True).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append({
            "text": doc,
            "source": meta.get("source", "unknown"),
            "chunk_index": meta.get("chunk_index", 0),
            "distance": dist,
        })
    return hits


def generate_answer(query: str, hits: list[dict]) -> str:
    """Build a grounded prompt and call Ollama."""
    context_parts = []
    for i, hit in enumerate(hits, start=1):
        context_parts.append(
            f"[{i}] (source: {hit['source']}, chunk {hit['chunk_index']})\n"
            f"{hit['text']}"
        )
    context = "\n\n".join(context_parts)

    prompt = (
        "You are a helpful assistant. Answer the question using ONLY the "
        "numbered context passages below. Cite sources as [1], [2], etc. "
        "If the context is insufficient, say so.\n\n"
        f"{context}\n\n"
        f"Question: {query}\n\nAnswer:"
    )

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"]


# ---------------------------------------------------------------------------
# Session state — initialise once per browser session.
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    # Each entry: {"role": "user"|"assistant", "content": str,
    #              "hits": list[dict] | None}
    st.session_state.messages = []


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("RAG Chat")
st.caption(
    f"Local RAG powered by **{LLM_MODEL}** via Ollama · "
    f"**{EMBED_MODEL}** embeddings · ChromaDB vector store"
)

# Load pipeline (cached after first call).
collection, embedder = load_pipeline()

# Render the full conversation history on every rerun.
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # Show retrieved sources for assistant messages.
        if msg["role"] == "assistant" and msg.get("hits"):
            with st.expander("Retrieved sources", expanded=False):
                for i, hit in enumerate(msg["hits"], start=1):
                    st.markdown(
                        f"**[{i}] {hit['source']}** · "
                        f"chunk {hit['chunk_index']} · "
                        f"distance `{hit['distance']:.4f}`"
                    )
                    st.caption(hit["text"])

# Accept new user input.
if user_input := st.chat_input("Ask a question about the documents…"):
    # Display user message immediately.
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append(
        {"role": "user", "content": user_input, "hits": None}
    )

    # Retrieve and generate.
    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating…"):
            hits = retrieve(user_input, collection, embedder)
            assistant_text = generate_answer(user_input, hits)

        st.markdown(assistant_text)

        # Show sources in an expander below the answer.
        if hits:
            with st.expander("Retrieved sources", expanded=False):
                for i, hit in enumerate(hits, start=1):
                    st.markdown(
                        f"**[{i}] {hit['source']}** · "
                        f"chunk {hit['chunk_index']} · "
                        f"distance `{hit['distance']:.4f}`"
                    )
                    st.caption(hit["text"])

    # Persist assistant message with its hits for history replay.
    st.session_state.messages.append(
        {"role": "assistant", "content": assistant_text, "hits": hits}
    )
```

## Step 3 — Run the app

```bash
streamlit run app.py
```

Streamlit starts a local web server and opens the app in your default browser at `http://localhost:8501`. You should see:

- A title bar with the model and tool names
- An empty chat area
- A text input at the bottom

Type a question and press Enter. The spinner appears while Ollama generates, then the answer appears with a "Retrieved sources" expander below it.

## Step 4 — Understand the Streamlit patterns

### `@st.cache_resource`

```python
@st.cache_resource
def load_pipeline():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(COLLECTION_NAME)
    embedder = SentenceTransformer(EMBED_MODEL)
    return collection, embedder
```

Streamlit re-runs the entire script on every user interaction. Without caching, `SentenceTransformer` would reload the model (a several-second operation) every time the user types a message. `@st.cache_resource` caches the returned objects for the lifetime of the server process.

### `st.session_state`

```python
if "messages" not in st.session_state:
    st.session_state.messages = []
```

`st.session_state` is a per-browser-tab dictionary that persists across reruns. Storing the full message list here is how Streamlit chat apps maintain multi-turn history.

### Rendering history

```python
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
```

Every rerun replays the entire history before showing the new exchange. This is the idiomatic Streamlit pattern — the UI is always rebuilt from state, never mutated in place.

### The walrus operator in `st.chat_input`

```python
if user_input := st.chat_input("Ask a question…"):
```

`st.chat_input` returns `None` when the user has not yet submitted anything and a string when they have. The walrus operator (`:=`) assigns and tests in one expression, keeping the logic tight.

### Sources expander

```python
with st.expander("Retrieved sources", expanded=False):
    for i, hit in enumerate(hits, start=1):
        st.markdown(f"**[{i}] {hit['source']}** …")
        st.caption(hit["text"])
```

The expander starts collapsed so it does not clutter the conversation. Users who want to verify the model's sources can click to expand. Storing `hits` alongside each assistant message in `st.session_state` means the sources are shown correctly even when scrolling back through history.

## Step 5 — Experiment

!!! tip "Things to try"
    - Ask a follow-up question that references the previous answer to see how the stateless pipeline handles (or mishandles) multi-turn context.
    - Change `TOP_K` to 5 and observe whether longer context helps or hurts answer quality.
    - Add a `st.sidebar` with a slider for `TOP_K` so users can tune retrieval at runtime.
    - Replace the static `CHROMA_PATH` with a sidebar file uploader and an on-the-fly ingest call.

!!! note "Multi-turn context"
    The current pipeline is stateless: each question is answered independently using only retrieved chunks. If you need the model to remember earlier turns, append previous `user`/`assistant` messages to the `messages` list passed to `ollama.chat`. This is shown in the reference implementation at [`../projects/local-rag`](../projects/local-rag.md).

## Next steps

- [Tutorial 04 — PDF Q&A](04-pdf-qa.md): extend the pipeline to load and query PDF files with page-level citations.
- [Local RAG project](../projects/local-rag.md): a complete, production-ready reference implementation that extends this app with multi-turn history and configurable settings.
