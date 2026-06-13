# Sample Projects

Hands-on projects that wire together every concept covered in this portal into a working, end-to-end RAG application you can run on your own machine.

## What you'll learn

- How individual building blocks (loaders, embeddings, vector stores, LLMs) combine into a real system
- Common patterns for structuring a RAG codebase
- How to extend and adapt a working baseline to your own data and use-case

## Available projects

| Project | Stack | Difficulty | What you learn |
|---------|-------|-----------|----------------|
| [Local RAG](local-rag.md) | sentence-transformers · ChromaDB · Ollama · Streamlit | Beginner | End-to-end local pipeline, chunking strategy, Streamlit chat UI |
| [Multi-PDF Research Assistant](multi-pdf-research-assistant.md) | pypdf · ChromaDB · Ollama · Streamlit | Intermediate | Multi-document ingestion, page-aware metadata, per-file/page citations |
| [Codebase Q&A](codebase-qa.md) | sentence-transformers · ChromaDB · Ollama | Intermediate | Walking a repo, line-based chunking, `path:line` citations |
| [Hybrid + Rerank App](hybrid-rerank-rag.md) | BM25 · ChromaDB · cross-encoder · Ollama | Advanced | Hybrid search, RRF fusion, cross-encoder reranking, stage-by-stage comparison |

## How to use these projects

Each project page includes:

1. An architecture diagram
2. A file-by-file code walkthrough
3. Step-by-step setup and run instructions
4. Config knobs you can tune
5. Ideas for extending the project further

All projects run entirely on your local machine — no API keys or cloud accounts required.

!!! tip "New to RAG?"
    Work through the [tutorials](../tutorials/index.md) before tackling a full project. The tutorials introduce each concept in isolation so the project code makes sense at a glance.

## Next steps

- Start simple: [Local RAG project walkthrough](local-rag.md)
- Go multi-document: [Multi-PDF Research Assistant](multi-pdf-research-assistant.md)
- Ask your own repo: [Codebase Q&A](codebase-qa.md)
- Level up retrieval: [Hybrid + Rerank App](hybrid-rerank-rag.md)
