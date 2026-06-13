# Tools & Ecosystem

The RAG ecosystem is a collection of focused libraries — you compose them rather than buying a monolith. This page maps every major role and recommends a minimal stack to get started.

## What you'll learn

- How the RAG toolchain is divided by role (runtime, orchestration, storage, embeddings, evaluation)
- Which open-source options exist at each layer and how they compare
- A "start here" minimal stack recommendation for local development
- Where to find deep-dive pages for each tool

---

## Ecosystem map

### LLM runtimes

These tools run large language models locally on your machine or on-premise infrastructure.

| Tool | Description | Page |
|------|-------------|------|
| **Ollama** | One-command local LLM server; supports llama, mistral, phi, gemma, nomic-embed and more | [ollama.md](ollama.md) |

### Orchestration frameworks

Frameworks that wire loaders → splitters → embedders → vector stores → LLMs into runnable pipelines.

| Tool | Strengths | Page |
|------|-----------|------|
| **LangChain** | Huge ecosystem, LCEL composability, best documentation coverage | [langchain.md](langchain.md) |
| **LlamaIndex** | Document-first abstractions, advanced query engines, knowledge graphs | [llamaindex.md](llamaindex.md) |
| **Haystack** | Production-grade pipelines, strong eval tooling, YAML-configurable | *(community docs)* |

### Vector stores

Databases that store and search embedding vectors.

| Tool | Mode | Page |
|------|------|------|
| **ChromaDB** | Embedded or server, easy Python API | [chromadb.md](chromadb.md) |
| **FAISS** | In-memory, ultra-fast similarity search from Meta | [vector-stores.md](vector-stores.md) |
| **Qdrant** | Server or cloud, rich filtering, Rust core | [vector-stores.md](vector-stores.md) |
| **Weaviate** | GraphQL API, multi-modal, hybrid search built-in | [vector-stores.md](vector-stores.md) |
| **Milvus** | Distributed, billion-scale, ANNS algorithms | [vector-stores.md](vector-stores.md) |
| **pgvector** | PostgreSQL extension — vector search inside your existing DB | [vector-stores.md](vector-stores.md) |

### Embedding models

Models that convert text to dense vectors for semantic search.

| Source | Examples |
|--------|---------|
| **sentence-transformers** | `all-MiniLM-L6-v2`, `bge-small-en-v1.5`, `e5-large-v2` |
| **Ollama** | `nomic-embed-text`, `mxbai-embed-large` |
| **API** | OpenAI `text-embedding-3-small`, Cohere `embed-v3` |

See the full guide: [embedding-models.md](embedding-models.md)

### Rerankers

Cross-encoders that re-score retrieved chunks for higher precision. See [../advanced/reranking.md](../advanced/reranking.md).

### Evaluation

| Tool | Purpose |
|------|---------|
| **Ragas** | Automated RAG metrics: faithfulness, answer relevancy, context recall |
| **TruLens** | LLM app evaluation with triad metrics |

See [../advanced/evaluation.md](../advanced/evaluation.md).

---

## What do I actually need?

!!! tip "Minimal local stack"
    For a working prototype that runs entirely offline, you need exactly four components:

    ```text
    Ollama              ← runs your LLM and embedding model
    ChromaDB            ← stores and searches vectors
    LangChain or LlamaIndex  ← wires the pipeline
    sentence-transformers    ← optional if Ollama embeddings suffice
    ```

    Install them all in one shot:

    ```bash
    pip install langchain langchain-community langchain-ollama chromadb sentence-transformers
    ```

!!! info "When to add more"
    - **Scale past a few thousand documents** → swap Chroma for Qdrant or Milvus
    - **Need hybrid BM25 + vector search** → see [../advanced/hybrid-search.md](../advanced/hybrid-search.md)
    - **Measure quality systematically** → add Ragas; see [../advanced/evaluation.md](../advanced/evaluation.md)
    - **Go to production** → see [../advanced/production.md](../advanced/production.md)

---

## Next steps

- [ollama.md](ollama.md) — Install and serve local models in minutes
- [langchain.md](langchain.md) — Build your first RAG chain
- [../tutorials/01-minimal-rag.md](../tutorials/01-minimal-rag.md) — End-to-end minimal RAG walkthrough
