# Gradio — RAG Chat Interface

Gradio makes it trivial to wrap a Python function in a web UI and share it with a public link. It is a natural fit for RAG demos and model showcases where you want fast results without building a full application.

## What you'll learn

- How to build a RAG chat with `gr.ChatInterface` in a few lines
- How to create a custom layout with `gr.Blocks` and source-document display
- How to launch locally and generate a public share link
- When to choose Gradio over Streamlit

## Install

```bash
pip install gradio
```

## The fastest path: gr.ChatInterface

`gr.ChatInterface` wraps any function that accepts `(message, history)` and returns a string reply. It handles the UI, history rendering, and input box automatically.

```python
import gradio as gr

def rag_chat(message: str, history: list[list[str]]) -> str:
    """
    message  — the latest user input string
    history  — list of [user_msg, assistant_msg] pairs (Gradio format)
    """
    # Wire your pipeline here:
    # result = pipeline.query(message)
    # return result["answer"]
    return f"(Pipeline not wired) You asked: {message}"

demo = gr.ChatInterface(
    fn=rag_chat,
    title="RAG Chat",
    description="Ask questions about your documents.",
    examples=["What is RAG?", "Summarise the key findings."],
)

demo.launch()
```

```bash
python app.py
# Opens http://127.0.0.1:7860
```

### Streaming with ChatInterface

Return a generator instead of a string to stream tokens.

```python
from typing import Generator

def rag_stream(message: str, history: list[list[str]]) -> Generator[str, None, None]:
    partial = ""
    # async for token in pipeline.astream(message):
    for word in f"Answer to '{message}': Paris is the capital.".split():
        partial += word + " "
        yield partial

demo = gr.ChatInterface(fn=rag_stream, title="RAG Stream")
demo.launch()
```

Gradio detects the generator and updates the output bubble in-place as tokens arrive.

## Custom layout with gr.Blocks

Use `gr.Blocks` when you need more than a single chat column — for example, to show retrieved source documents alongside the answer.

```python
import gradio as gr

def rag_with_sources(message: str, history: list) -> tuple[str, str]:
    """Return (answer, sources_markdown) as a tuple."""
    # result = pipeline.query(message)
    # answer = result["answer"]
    # sources_md = "\n\n".join(
    #     f"**{d['source']}** (score {d['score']:.2f})\n\n{d['content']}"
    #     for d in result["sources"]
    # )
    answer = f"Answer to: {message}"
    sources_md = "**doc.pdf** (score 0.91)\n\nRelevant chunk text here."
    return answer, sources_md

with gr.Blocks(title="RAG with Sources") as demo:
    gr.Markdown("## RAG Chat with Source Documents")

    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Conversation")
            msg = gr.Textbox(placeholder="Ask a question...", label="Question")
            clear = gr.Button("Clear")

        with gr.Column(scale=1):
            sources_box = gr.Markdown(label="Retrieved Sources", value="*Sources appear here.*")

    def respond(message: str, chat_history: list) -> tuple:
        answer, sources_md = rag_with_sources(message, chat_history)
        chat_history = chat_history + [[message, answer]]
        return "", chat_history, sources_md

    msg.submit(respond, inputs=[msg, chatbot], outputs=[msg, chatbot, sources_box])
    clear.click(lambda: ([], "*Sources appear here.*"), outputs=[chatbot, sources_box])

demo.launch()
```

### Loading the pipeline once

Initialise heavy objects outside the handler function so they load once, not on every request.

```python
import gradio as gr

# Runs at import time — once per process
# from my_pipeline import RAGPipeline
# pipeline = RAGPipeline.load("./index")
pipeline = None  # replace with your pipeline

def rag_chat(message: str, history: list) -> str:
    if pipeline is None:
        return "Pipeline not loaded."
    result = pipeline.query(message)
    return result["answer"]

demo = gr.ChatInterface(fn=rag_chat)
demo.launch()
```

## Sharing with a public URL

Add `share=True` to `demo.launch()` to get a temporary public HTTPS URL powered by Gradio's tunnel service.

```python
demo.launch(share=True)
# Running on public URL: https://xxxx.gradio.live
```

!!! warning "share=True is temporary"
    The public URL expires after 72 hours. For persistent hosting, deploy to Hugging Face Spaces, a cloud VM, or see [Docker](docker.md) for containerisation.

## Additional launch options

```python
demo.launch(
    server_name="0.0.0.0",   # listen on all interfaces (needed inside Docker)
    server_port=7860,
    share=False,
    auth=("admin", "password"),  # simple HTTP auth
)
```

## Gradio vs Streamlit

| | Gradio | Streamlit |
|---|---|---|
| **Best for** | Quick ML demos, easy sharing | Data apps, dashboards, custom pipelines |
| **Sharing** | `share=True` gives instant public URL | Streamlit Community Cloud or self-host |
| **Layout** | `ChatInterface` (zero config) or `Blocks` | Full column/tab/component system |
| **State** | Managed per-session internally | Explicit `st.session_state` |
| **Streaming** | Generator return from handler | `st.write_stream(generator)` |
| **Learning curve** | Lower for simple demos | More powerful, slightly steeper |

Use Gradio when you want to share a demo link immediately or prototype a model interface in under 20 lines. Use Streamlit when you need rich layouts, custom theming, or integration with pandas/charts. For the Streamlit-specific patterns see [Streamlit](streamlit.md) and the full [Streamlit Chat App tutorial](../tutorials/03-streamlit-chat-app.md).

## Complete standalone example

```python
import gradio as gr

# pipeline = RAGPipeline.load("./index")  # load once here

def chat(message: str, history: list[list[str]]) -> str:
    # result = pipeline.query(message)
    # return result["answer"]
    return f"(demo) Answer to: {message}"

demo = gr.ChatInterface(
    fn=chat,
    title="RAG Demo",
    description="Ask questions answered by your document index.",
    examples=["What is the main topic?", "Summarise in one sentence."],
    theme=gr.themes.Soft(),
)

if __name__ == "__main__":
    demo.launch()
```

```bash
python app.py
# Local: http://127.0.0.1:7860
```

## Next steps

- [Streamlit](streamlit.md) — richer app framework for production UIs
- [Streamlit Chat App tutorial](../tutorials/03-streamlit-chat-app.md) — full walkthrough of a chat UI
- [FastAPI](fastapi.md) — expose RAG as a JSON API that Gradio (or any client) can call
