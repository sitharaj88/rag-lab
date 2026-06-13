# Qdrant

Qdrant is a vector database and similarity search engine that supports rich payload filtering, multiple distance metrics, and both in-memory and persistent storage. This guide uses the `qdrant-client` v1.16+ API.

## What you'll learn

- Running Qdrant locally in-memory and with on-disk persistence
- Creating a collection and upserting points with `client.upsert`
- Querying with `client.query_points`
- Filtering results by payload fields
- Understanding which legacy methods were removed in v1.9+

## Install

```bash
pip install qdrant-client
```

!!! note "Running Qdrant server"
    For a persistent, production-grade setup you can run the official Docker image:
    ```bash
    docker run -p 6333:6333 qdrant/qdrant
    ```
    Then connect with `QdrantClient(host="localhost", port=6333)`.

## Imports

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
```

## Quick start — in-memory collection

```python
# ── 1. In-memory client (great for prototyping and tests) ─────────────────────
client = QdrantClient(":memory:")

COLLECTION = "documents"
DIM = 384  # must match your embedding model

# ── 2. Create collection ──────────────────────────────────────────────────────
client.create_collection(
    collection_name=COLLECTION,
    vectors_config=VectorParams(size=DIM, distance=Distance.COSINE),
)

# ── 3. Sample documents ───────────────────────────────────────────────────────
docs = [
    {"id": 1, "text": "The quick brown fox jumps over the lazy dog.", "category": "animals"},
    {"id": 2, "text": "A fast auburn fox leaps above a sleepy hound.",  "category": "animals"},
    {"id": 3, "text": "Quantum computing harnesses superposition.",      "category": "science"},
    {"id": 4, "text": "Machine learning improves with more data.",       "category": "science"},
]

# Simulate embeddings — replace with your real encoder output
import numpy as np
rng = np.random.default_rng(42)
embeddings = rng.standard_normal((len(docs), DIM)).astype("float32")

# ── 4. Upsert points ──────────────────────────────────────────────────────────
client.upsert(
    collection_name=COLLECTION,
    points=[
        PointStruct(
            id=doc["id"],
            vector=embeddings[i].tolist(),
            payload={"text": doc["text"], "category": doc["category"]},
        )
        for i, doc in enumerate(docs)
    ],
)

print(client.get_collection(COLLECTION).points_count, "points stored")
```

## Querying with `query_points`

```python
query_vector = rng.standard_normal(DIM).astype("float32").tolist()

results = client.query_points(
    collection_name=COLLECTION,
    query=query_vector,
    limit=2,
)

for point in results.points:
    print(f"id={point.id}  score={point.score:.4f}  text={point.payload['text']}")
```

`query_points` returns a `QueryResponse` object; iterate over `.points` to access `id`, `score`, and `payload`.

## Payload filtering

Qdrant lets you combine vector similarity with structured filters on any payload field:

```python
from qdrant_client.models import Filter, FieldCondition, MatchValue

results = client.query_points(
    collection_name=COLLECTION,
    query=query_vector,
    query_filter=Filter(
        must=[FieldCondition(key="category", match=MatchValue(value="science"))]
    ),
    limit=5,
)

for point in results.points:
    print(point.payload["text"])
```

## Persistent local storage

```python
# Vectors are written to disk inside ./qdrant_storage
persistent_client = QdrantClient(path="./qdrant_storage")

persistent_client.create_collection(
    collection_name="my_docs",
    vectors_config=VectorParams(size=DIM, distance=Distance.DOT),
)
```

Use `QdrantClient(host="localhost", port=6333)` to connect to a running Qdrant server instead.

!!! warning "Removed legacy methods — do NOT use"
    `qdrant-client` v1.9 removed several methods that appear in older tutorials and blog posts.
    These methods **no longer exist** and calling them raises `AttributeError`:

    - `client.search(...)` — replaced by `client.query_points(...)`
    - `client.recommend(...)` — replaced by `client.query_points(query=RecommendQuery(...))`
    - `client.discover(...)` — replaced by `client.query_points(query=DiscoverQuery(...))`
    - All `*_batch` variants (`search_batch`, `recommend_batch`, etc.) — batch via lists inside `query_points`

    Always use `client.upsert(collection_name, points=[...])` and `client.query_points(collection_name, query=vector, limit=k)`.

    See [Versions and Deprecations](../sdks/versions-and-deprecations.md) for a full migration timeline.

## Distance metrics

| `Distance` constant | Use case |
|---|---|
| `Distance.COSINE` | Sentence embeddings (most common) |
| `Distance.DOT` | OpenAI `text-embedding-3-*` (already normalised) |
| `Distance.EUCLID` | Image / pixel embeddings |

## When to use Qdrant

Qdrant is a good fit when you need:

- Metadata / payload filtering alongside vector search
- A standalone database process (local Docker or cloud-hosted)
- Production-grade features: snapshots, collections, named vectors, sparse vectors

For a pure in-process option with no server, see [FAISS](faiss.md).

## Next steps

- Review the API change history: [Versions and Deprecations](../sdks/versions-and-deprecations.md)
- Compare vector store options: [Vector Stores](../tools/vector-stores.md)
- In-process alternative: [FAISS](faiss.md)
