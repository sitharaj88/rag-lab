# Vector Search and Embedding Models

Vector search retrieves items from a database by geometric proximity in a
high-dimensional embedding space.  It is the core technology behind modern
semantic search, recommendation systems, and retrieval-augmented generation.

## How Embeddings Work

An embedding model maps a piece of text to a fixed-length vector of floating-
point numbers (e.g. 384 dimensions for all-MiniLM-L6-v2).  The model is
trained so that semantically similar texts produce vectors that are close in
cosine distance, regardless of surface wording.

Sentence-transformers (the `sentence-transformers` Python library) provides
pretrained embedding models that run entirely on CPU.  The `encode()` method
accepts a string or list of strings and returns a NumPy array.

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
vector = model.encode("What is retrieval-augmented generation?")
# vector.shape == (384,)
```

## Approximate Nearest Neighbour Search

Exact nearest-neighbour search (brute-force dot product over every vector)
scales as O(N * D) per query, which is too slow for large corpora.
Approximate Nearest Neighbour (ANN) indexes trade a small accuracy loss for
orders-of-magnitude speedup.

Common ANN algorithms:
- **HNSW** (Hierarchical Navigable Small World): graph-based, very fast
  queries, used by ChromaDB and Qdrant by default.
- **IVF** (Inverted File Index): clusters vectors into Voronoi cells; used by
  FAISS for large-scale deployments.
- **ScaNN**: Google's production ANN library, optimised for dot-product.

ChromaDB uses HNSW internally via the `hnswlib` library and exposes a simple
Python API.  Persistence is handled automatically when you use
`PersistentClient`.

## ChromaDB Basics

ChromaDB is an open-source, embeddable vector database.  Key concepts:

- **Collection**: a named set of documents + embeddings, analogous to a table.
- **add()**: upsert documents, embeddings, and metadata.
- **query()**: ANN search returning ids, distances, and documents.
- **PersistentClient**: saves the collection to disk so it survives restarts.

```python
import chromadb

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(
    name="my_docs",
    metadata={"hnsw:space": "cosine"},
)
collection.add(
    ids=["doc_0", "doc_1"],
    embeddings=[[0.1, 0.2, ...], [0.3, 0.4, ...]],
    documents=["First passage", "Second passage"],
)
results = collection.query(query_embeddings=[[0.1, 0.2, ...]], n_results=5)
```

## Distance Metrics

- **Cosine similarity**: angle between vectors; normalised for vector length.
  Best for text embeddings where magnitude is not meaningful.
- **Dot product**: cosine * magnitude product; preferred when embeddings are
  trained with dot-product objectives (e.g. OpenAI embeddings).
- **Euclidean (L2)**: straight-line distance; sensitive to magnitude; less
  common for text.

ChromaDB's `hnsw:space` metadata key accepts `"cosine"`, `"ip"` (inner
product), or `"l2"`.  For sentence-transformers, `"cosine"` is recommended.

## Choosing an Embedding Model

| Model                    | Dims | Size   | Speed (CPU) | Notes                     |
|--------------------------|------|--------|-------------|---------------------------|
| all-MiniLM-L6-v2         | 384  | ~80 MB | Fast        | Best all-round RAG choice |
| all-mpnet-base-v2        | 768  | ~420 MB| Medium      | Higher quality, slower    |
| multi-qa-MiniLM-L6-cos-v1| 384  | ~80 MB | Fast        | Tuned for QA retrieval    |
| bge-small-en-v1.5        | 384  | ~67 MB | Fast        | Strong MTEB scores        |

For local RAG demos, `all-MiniLM-L6-v2` hits the best speed/quality balance.

## Embedding Dimensions and Chunking Strategy

The embedding model has a maximum token limit (typically 256–512 tokens for
MiniLM).  Chunks longer than this limit are truncated, losing information from
the tail.  Best practices:

1. Chunk size should not exceed ~75% of the model's token limit.
2. Add a small overlap (10–20% of chunk size) so boundary sentences appear in
   at least one chunk fully.
3. Prefer splitting on paragraph or sentence boundaries rather than fixed
   character counts when the document structure allows it.

For production systems, document-specific chunking (e.g. heading-aware splits
for Markdown, table-aware splits for PDFs) dramatically improves retrieval
quality compared to naive fixed-size chunking.
