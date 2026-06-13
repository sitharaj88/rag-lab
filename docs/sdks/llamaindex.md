# LlamaIndex SDK

LlamaIndex is a data framework purpose-built for connecting LLMs to external knowledge sources through a Document → Node → Index → QueryEngine abstraction. This page covers the v0.10+ modular API with global `Settings`, local Ollama inference, and HuggingFace embeddings.

## What you'll learn

- The core LlamaIndex abstractions: Documents, Nodes, Indexes, and QueryEngines
- How to configure the global `Settings` object instead of the deprecated `ServiceContext`
- How to build a fully local pipeline over a directory of files
- How to persist an index to disk and reload it without re-embedding
- Which import patterns have changed and how to avoid stale tutorials

## Install

LlamaIndex v0.10+ uses a modular namespace layout. Install only what you need:

```bash
pip install llama-index-core \
            llama-index-llms-ollama \
            llama-index-embeddings-huggingface \
            llama-index-vector-stores-chroma
```

| Package | Purpose |
|---|---|
| `llama-index-core` | `VectorStoreIndex`, `SimpleDirectoryReader`, `Settings`, base classes |
| `llama-index-llms-ollama` | `Ollama` LLM integration |
| `llama-index-embeddings-huggingface` | `HuggingFaceEmbedding` |
| `llama-index-vector-stores-chroma` | `ChromaVectorStore` integration |

!!! warning "ServiceContext has been removed — flat imports are gone too"
    In LlamaIndex v0.10, `ServiceContext` was fully removed. Do **not** write `ServiceContext.from_defaults(llm=..., embed_model=...)`. The replacement is the global `Settings` object: `Settings.llm = ...` and `Settings.embed_model = ...`. Additionally, the old flat import style (`from llama_index import VectorStoreIndex`) no longer works — all imports must use the namespaced form (`from llama_index.core import VectorStoreIndex`). See [versions-and-deprecations.md](versions-and-deprecations.md) for the full changelog.

## Core abstractions

| Abstraction | Description |
|---|---|
| **Document** | A unit of raw text plus metadata, loaded from a file, database, or API |
| **Node** | A chunk derived from a Document during ingestion; the unit actually embedded |
| **Index** | A data structure (e.g., `VectorStoreIndex`) that organises Nodes for retrieval |
| **QueryEngine** | A retrieval + synthesis interface built on top of an Index |

## Configure global Settings

Set the LLM and embedding model once before building any index:

```python
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Local inference via Ollama (must be running: `ollama serve`)
Settings.llm = Ollama(model="llama3.2", request_timeout=120.0)

# Local embeddings — no API key required
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Optional: tune chunking globally
Settings.chunk_size = 512
Settings.chunk_overlap = 50
```

## Full local RAG example

### Step 1 — load documents

```python
from llama_index.core import SimpleDirectoryReader

# Recursively loads .txt, .pdf, .md, etc. from the directory
documents = SimpleDirectoryReader("./docs_data").load_data()
print(f"Loaded {len(documents)} document(s)")
```

### Step 2 — build a VectorStoreIndex

```python
from llama_index.core import VectorStoreIndex

# Nodes are created, embedded, and stored in-memory by default
index = VectorStoreIndex.from_documents(documents)
```

### Step 3 — query

```python
query_engine = index.as_query_engine(similarity_top_k=4)
response = query_engine.query("What is retrieval-augmented generation?")
print(response)
```

The `response` object contains `.response` (the answer string) and `.source_nodes` (the retrieved chunks with scores).

## Persist and reload an index

Rebuild from disk on subsequent runs — no re-embedding required:

```python
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage

PERSIST_DIR = "./storage"

# First run: build and persist
index = VectorStoreIndex.from_documents(documents)
index.storage_context.persist(persist_dir=PERSIST_DIR)

# Subsequent runs: reload
storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
index = load_index_from_storage(storage_context)
query_engine = index.as_query_engine(similarity_top_k=4)
```

## Using Chroma as a persistent vector store

For larger corpora, swap the default in-memory store for Chroma:

```python
import chromadb
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore

chroma_client = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = chroma_client.get_or_create_collection("rag_lab")

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

# Reload later
index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store,
    storage_context=StorageContext.from_defaults(vector_store=vector_store),
)
query_engine = index.as_query_engine(similarity_top_k=4)
```

## Pros and cons

| Pros | Cons |
|---|---|
| Purpose-built for document Q&A and knowledge retrieval | Less flexible for non-RAG agent patterns |
| Rich Node metadata and source attribution built in | Modular split means more packages to manage |
| Persist/reload cycle is clean and built into core | Smaller tool/integration ecosystem than LangChain |
| Global `Settings` makes configuration concise | v0.10 migration broke many existing tutorials |

**When to use LlamaIndex:** Choose it when your primary use case is document ingestion and question-answering, you need fine-grained node metadata, or you want source attribution out of the box. For broader integration needs or LCEL-style composition, see [LangChain](langchain.md).

## Next steps

- Explore the local tooling guide: [../tools/llamaindex.md](../tools/llamaindex.md)
- Compare with [LangChain](langchain.md) and [Haystack](haystack.md)
- Review breaking changes: [versions-and-deprecations.md](versions-and-deprecations.md)
- Understand the retrieval layer: [../foundations/retrieval.md](../foundations/retrieval.md)
- See a full worked tutorial: [../tutorials/02-rag-with-ollama-chroma.md](../tutorials/02-rag-with-ollama-chroma.md)
