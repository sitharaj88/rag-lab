# Tutorials

Work through these hands-on tutorials to build real RAG applications from scratch, using only local tools: Ollama, ChromaDB, and sentence-transformers.

## What you'll learn

- How to build a minimal RAG pipeline with no external dependencies
- How to scale up to a persistent vector store with ChromaDB
- How to wrap your pipeline in a Streamlit chat interface
- How to answer questions over PDF documents with page-level citations

## Prerequisites

Before starting, make sure you have completed:

- [Environment Setup](../getting-started/environment-setup.md) — Python environment, pip packages
- [Running a Local LLM with Ollama](../getting-started/local-llm-ollama.md) — Ollama installed and `llama3.2` pulled

## Tutorial track

| # | Tutorial | What you build | Difficulty | Time |
|---|----------|---------------|------------|------|
| 01 | [Minimal RAG (~40 lines)](01-minimal-rag.md) | Single-file in-memory RAG with numpy cosine similarity | Beginner | 20 min |
| 02 | [RAG with Ollama + ChromaDB](02-rag-with-ollama-chroma.md) | Persistent vector store with ingest and query scripts | Beginner | 30 min |
| 03 | [Streamlit Chat App](03-streamlit-chat-app.md) | Full chat UI with session history and source expander | Intermediate | 45 min |
| 04 | [PDF Q&A](04-pdf-qa.md) | Q&A over PDF files with page-level citations | Intermediate | 45 min |

## How to follow these tutorials

Each tutorial is self-contained and ends with runnable code. You can follow them in order — each one builds on concepts introduced in the previous — or jump directly to the topic you need.

All tutorials use the same local stack:

- **Embeddings:** `sentence-transformers` with `all-MiniLM-L6-v2`
- **LLM:** Ollama with `llama3.2`
- **Vector store:** ChromaDB (from tutorial 02 onward)

A complete reference implementation lives at [`../projects/local-rag`](../projects/local-rag.md) if you want to compare your work against a finished project.

!!! tip "New to RAG?"
    If you haven't read the foundations yet, start with [What is RAG?](../getting-started/what-is-rag.md) and [Embeddings](../foundations/embeddings.md) before diving into code.
