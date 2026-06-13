# LlamaIndex

LlamaIndex (formerly GPT Index) is a data framework purpose-built for connecting LLMs to external knowledge — its document/node abstraction and first-class query engines make it particularly strong for complex document retrieval scenarios.

## What you'll learn

- LlamaIndex's core concepts: Documents, Nodes, Indices, and Query Engines
- How to build a local RAG pipeline using Ollama and HuggingFace embeddings
- When to prefer LlamaIndex over LangChain
- How to persist an index and reload it without re-embedding

---

## Installation

```bash
pip install llama-index-core llama-index-llms-ollama llama-index-embeddings-huggingface
```

!!! note "Modular packages"
    LlamaIndex 0.10+ splits integrations into separate packages. The top-level `llama-index` meta-package still exists but install `llama-index-core` directly to keep dependencies lean and explicit.

!!! warning "`ServiceContext` removed in v0.11"
    `ServiceContext.from_defaults(...)` and `set_global_service_context(...)` were removed in LlamaIndex v0.11. Use the `Settings` object instead: `from llama_index.core import Settings; Settings.llm = ...; Settings.embed_model = ...`. See [Versions & Deprecations](../sdks/versions-and-deprecations.md) for the full migration guide, and the [LlamaIndex deep-dive tutorial](../sdks/llamaindex.md) for current patterns.

---

## Core concepts

| Concept | Description |
|---------|-------------|
| **Document** | A loaded piece of content with text + metadata |
| **Node** | A chunk of a Document; the atomic unit of retrieval |
| **Index** | A data structure that organizes Nodes for search (e.g., `VectorStoreIndex`) |
| **Retriever** | Queries an Index and returns the top-k Nodes |
| **Query Engine** | Combines a Retriever with an LLM to answer questions end-to-end |
| **Response Synthesizer** | Controls how retrieved Nodes are merged into a final answer |

---

## Minimal local example

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 1. Configure models globally (avoids passing them everywhere)
Settings.llm = Ollama(model="llama3.2", request_timeout=120.0)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.chunk_size = 512
Settings.chunk_overlap = 50

# 2. Load documents from a directory
documents = SimpleDirectoryReader("data/").load_data()

# 3. Build (or load) a vector index
index = VectorStoreIndex.from_documents(documents, show_progress=True)

# 4. Persist so you don't re-embed next time
index.storage_context.persist(persist_dir="./storage")

# 5. Create a query engine and ask a question
query_engine = index.as_query_engine(similarity_top_k=4)
response = query_engine.query("What are the main topics covered in these documents?")
print(response)
```

### Reload a persisted index

```python
from llama_index.core import StorageContext, load_index_from_storage

storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_context)
query_engine = index.as_query_engine(similarity_top_k=4)
```

---

## Streaming responses

```python
query_engine = index.as_query_engine(streaming=True, similarity_top_k=4)
streaming_response = query_engine.query("Summarize the document.")

for token in streaming_response.response_gen:
    print(token, end="", flush=True)
```

---

## Advanced: chat engine

LlamaIndex provides a `ChatEngine` that maintains conversation history automatically:

```python
chat_engine = index.as_chat_engine(chat_mode="condense_plus_context", verbose=True)
response = chat_engine.chat("Tell me about the introduction section.")
response2 = chat_engine.chat("What did you just say about it?")  # context is preserved
```

---

## LlamaIndex vs LangChain

| Dimension | LlamaIndex | LangChain |
|-----------|------------|-----------|
| **Document abstractions** | Richer — Nodes carry metadata, scores, relationships | Simpler Document wrapper |
| **Query engines** | Built-in tree, keyword, knowledge graph engines | Manual composition with LCEL |
| **Multi-document reasoning** | `SubQuestionQueryEngine`, router | Requires custom logic |
| **General LLM apps** | More limited | Much broader (agents, tools, memory) |
| **Learning curve** | Moderate | Moderate (steeper with LCEL) |
| **Best for** | Document-heavy RAG, complex retrieval | General LLM pipelines, agents |

!!! tip "When to pick LlamaIndex"
    Choose LlamaIndex when your core challenge is **document retrieval quality** — multi-document synthesis, structured node metadata, or query routing across many sources. Choose LangChain when you need **agentic behavior**, a wider integration catalog, or LangSmith observability.

---

## Next steps

- [langchain.md](langchain.md) — See the LangChain equivalent for a direct comparison
- [chromadb.md](chromadb.md) — Use ChromaDB as the backing vector store for LlamaIndex
