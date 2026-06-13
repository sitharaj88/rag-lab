# Streamlit — RAG Chat UI Reference

Streamlit lets you build interactive data apps in pure Python. This page is a reference for the chat primitives, state management, and caching patterns you need to wire a RAG pipeline into a conversational UI.

!!! tip "Full tutorial available"
    This page covers the Streamlit API surface. For a step-by-step walkthrough that builds a complete chat app from scratch, see the [Streamlit Chat App tutorial](../tutorials/03-streamlit-chat-app.md).

## What you'll learn

- The chat primitives: `st.chat_input` and `st.chat_message`
- How to persist conversation history with `st.session_state`
- How to load a heavy pipeline once with `@st.cache_resource`
- How to display retrieved source documents in an expander
- How to run the app

## Install

```bash
pip install streamlit
```

## Running the app

```bash
streamlit run app.py
```

Streamlit opens a browser tab automatically at `http://localhost:8501`.

## Chat primitives

### st.chat_input

`st.chat_input` renders a fixed-to-bottom input bar and returns the submitted string (or `None` if nothing was submitted this run).

```python
import streamlit as st

prompt = st.chat_input("Ask a question about your documents...")
if prompt:
    st.write(f"You asked: {prompt}")
```

### st.chat_message

`st.chat_message` renders a bubble with an avatar. Use it as a context manager to write any Streamlit content inside the bubble.

```python
with st.chat_message("user"):
    st.markdown(prompt)

with st.chat_message("assistant"):
    st.markdown("Here is what I found...")
```

Valid role strings: `"user"`, `"assistant"`, `"ai"`, `"human"`, or any string (Streamlit renders a generic avatar for unknown roles).

## Persisting conversation history

Streamlit re-runs the entire script on every interaction. Use `st.session_state` to keep state between runs.

```python
import streamlit as st

# Initialise history once per session
if "messages" not in st.session_state:
    st.session_state.messages = []

# Replay existing messages on each re-run
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle new input
prompt = st.chat_input("Ask something...")
if prompt:
    # Append user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and append assistant response
    with st.chat_message("assistant"):
        response = "Placeholder answer"
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
```

## Loading the pipeline once

RAG pipelines are expensive to initialise — they load embedding models, open vector-store connections, and may download weights. `@st.cache_resource` runs the decorated function once per process and caches the return value for the lifetime of the app.

```python
import streamlit as st

@st.cache_resource
def load_pipeline():
    # from my_pipeline import RAGPipeline
    # return RAGPipeline.load("./index")
    return {"status": "pipeline placeholder"}

pipeline = load_pipeline()
```

!!! warning "cache_resource vs cache_data"
    Use `@st.cache_resource` for objects that hold connections or loaded models (not serialisable). Use `@st.cache_data` for pure data transformations that return serialisable results (DataFrames, dicts, lists).

## Displaying source documents

Show retrieved chunks in an expandable section so the UI stays clean but sources are inspectable.

```python
def display_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for i, doc in enumerate(sources, 1):
            st.markdown(f"**[{i}] {doc.get('source', 'Unknown')}** — score: {doc.get('score', 0):.2f}")
            st.caption(doc.get("content", ""))
            st.divider()
```

## Streaming responses

Streamlit's `st.write_stream` accepts a generator and renders tokens as they arrive.

```python
import streamlit as st
from typing import Generator

def token_generator(answer: str) -> Generator[str, None, None]:
    for word in answer.split():
        yield word + " "

with st.chat_message("assistant"):
    response = st.write_stream(token_generator("The answer is Paris."))
# response now holds the fully-assembled string
st.session_state.messages.append({"role": "assistant", "content": response})
```

If your pipeline returns a streaming iterator directly, pass it to `st.write_stream` unchanged.

## Complete minimal example

```python
import streamlit as st

st.title("RAG Chat")

@st.cache_resource
def load_pipeline():
    # Replace with your actual pipeline
    return None

pipeline = load_pipeline()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"Sources ({len(msg['sources'])})"):
                for doc in msg["sources"]:
                    st.caption(f"{doc['source']}: {doc['content'][:200]}")

prompt = st.chat_input("Ask a question...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # result = pipeline.query(prompt) if pipeline else None
        answer = "Pipeline not wired up yet."
        sources = []
        st.markdown(answer)
        if sources:
            with st.expander(f"Sources ({len(sources)})"):
                for doc in sources:
                    st.caption(f"{doc['source']}: {doc['content'][:200]}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })
```

## Streamlit vs Gradio

| | Streamlit | Gradio |
|---|---|---|
| **Best for** | Data apps, dashboards, custom layouts | Quick ML demos, model showcases |
| **Layout control** | Full (columns, tabs, custom components) | Limited outside `gr.Blocks` |
| **Sharing** | Deploy to Streamlit Community Cloud | `demo.launch(share=True)` for a public URL |
| **State management** | Explicit via `session_state` | Managed internally per session |
| **Learning curve** | Slightly higher | Lower for simple I/O demos |

See [Gradio](gradio.md) for a side-by-side code comparison and the Gradio-specific RAG setup.

## Next steps

- [Streamlit Chat App tutorial](../tutorials/03-streamlit-chat-app.md) — full walkthrough building this app from scratch
- [Gradio](gradio.md) — alternative UI framework for quick demos and easy sharing
- [Local RAG project](../projects/local-rag.md) — end-to-end example combining a pipeline with this UI
