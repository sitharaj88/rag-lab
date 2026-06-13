# ChromaDB

ChromaDB is an open-source embedding database that runs embedded inside your Python process or as a standalone server, making it the lowest-friction vector store for local RAG development.

## What you'll learn

- The difference between `EphemeralClient`, `PersistentClient`, and `HttpClient`
- How to create collections, add documents, and run similarity queries
- Metadata filtering with `where` and `where_document` clauses
- Distance spaces (cosine, L2, inner product) and when to use each
- Bringing your own embeddings vs using a built-in embedding function

---

## Installation

```bash
pip install chromadb
```

---

## Client modes

```python
import chromadb

# In-memory only — data lost when the process exits
client = chromadb.EphemeralClient()

# Persisted to disk — survives restarts
client = chromadb.PersistentClient(path="./chroma_db")

# Remote server — run `chroma run --path /db` first
client = chromadb.HttpClient(host="localhost", port=8000)
```

!!! tip "Development vs production"
    Use `PersistentClient` for local development and single-machine deployments. Switch to `HttpClient` with a dedicated Chroma server when you need concurrent writes from multiple processes.

---

## Collections

A collection is a named namespace for a set of embeddings. You specify the distance function at creation time.

```python
# cosine is the most common choice for semantic search
collection = client.get_or_create_collection(
    name="my_docs",
    metadata={"hnsw:space": "cosine"},  # options: cosine | l2 | ip
)
```

| Distance | `hnsw:space` | Best for |
|----------|-------------|---------|
| Cosine similarity | `cosine` | Semantic search (most common) |
| Euclidean (L2) | `l2` | When vector magnitudes matter |
| Inner product | `ip` | Pre-normalized vectors, maximum retrieval speed |

---

## Adding documents

```python
collection.add(
    ids=["doc1", "doc2", "doc3"],
    documents=[
        "Retrieval-augmented generation grounds LLMs in external knowledge.",
        "Vector databases store high-dimensional embeddings for fast similarity search.",
        "Chunking breaks long documents into passages that fit within context windows.",
    ],
    metadatas=[
        {"source": "intro.pdf", "page": 1},
        {"source": "foundations.pdf", "page": 5},
        {"source": "foundations.pdf", "page": 8},
    ],
)
```

When `embeddings` are not provided, Chroma uses its default embedding function (`all-MiniLM-L6-v2` via sentence-transformers). Pass your own to override.

---

## Querying

```python
results = collection.query(
    query_texts=["How does RAG work?"],
    n_results=3,
    include=["documents", "metadatas", "distances"],
)

for doc, meta, dist in zip(
    results["documents"][0],
    results["metadatas"][0],
    results["distances"][0],
):
    print(f"[{dist:.4f}] {meta['source']} — {doc[:80]}")
```

### Metadata filtering

```python
# Only search documents from a specific source
results = collection.query(
    query_texts=["embeddings"],
    n_results=2,
    where={"source": "foundations.pdf"},          # exact match on metadata
    where_document={"$contains": "vector"},        # substring in document text
)
```

Chroma supports `$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`, `$in`, `$nin`, `$and`, `$or` in `where` clauses.

---

## Bring your own embeddings

Pass pre-computed vectors directly to skip Chroma's built-in embedding step:

```python
import numpy as np

my_embeddings = np.random.rand(3, 384).tolist()  # replace with real embeddings

collection.add(
    ids=["a", "b", "c"],
    embeddings=my_embeddings,
    documents=["text a", "text b", "text c"],
)

query_vector = np.random.rand(384).tolist()
results = collection.query(query_embeddings=[query_vector], n_results=2)
```

### Custom embedding function

```python
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

ef = SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-small-en-v1.5")

collection = client.get_or_create_collection(
    name="custom_embed",
    embedding_function=ef,
    metadata={"hnsw:space": "cosine"},
)
```

---

## CRUD operations

```python
# Update document text and metadata
collection.update(ids=["doc1"], documents=["Updated text."], metadatas=[{"version": 2}])

# Upsert (insert or update)
collection.upsert(ids=["doc4"], documents=["New document."])

# Delete by ID
collection.delete(ids=["doc3"])

# Count items
print(collection.count())
```

---

## Next steps

- [vector-stores.md](vector-stores.md) — Compare ChromaDB against FAISS, Qdrant, Weaviate, Milvus, and pgvector
- [../foundations/vector-databases.md](../foundations/vector-databases.md) — How vector databases work under the hood
