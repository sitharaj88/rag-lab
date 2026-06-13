# Weaviate

Weaviate is an open-source vector database with a GraphQL and REST API, built-in vectorisation modules, and multi-tenancy support. This guide uses the **v4 Python client**, which is a complete rewrite of v3 — the API is not backwards-compatible.

## What you'll learn

- Installing `weaviate-client` v4 and understanding the v3 → v4 breaking changes
- Connecting to a local Weaviate instance and to an embedded (in-process) instance
- Creating a collection with a vector index configuration
- Inserting objects and querying by vector with `near_vector`
- Properly closing the client connection

## Install

```bash
pip install weaviate-client
```

!!! note "Running a local Weaviate server"
    Start Weaviate with Docker:
    ```bash
    docker run -p 8080:8080 -p 50051:50051 cr.weaviate.io/semitechnologies/weaviate:latest
    ```
    The gRPC port `50051` is required by the v4 client.

## v4 client — what changed from v3

!!! warning "v4 is a full rewrite"
    `weaviate-client` v4 (released 2024) is **not compatible** with v3 code.
    Key changes:

    - `weaviate.Client(url=...)` is gone — use `weaviate.connect_to_local()` / `connect_to_embedded()`
    - `.schema.create_class(...)` is replaced by `client.collections.create(...)`
    - `.data_object.create(...)` is replaced by `collection.data.insert(...)`
    - `.query.get(...)` is replaced by `collection.query.near_vector(...)`
    - The client must be explicitly closed — use a `with` block or call `client.close()`

## Connecting

```python
import weaviate

# ── Option A: local Docker instance ───────────────────────────────────────────
client = weaviate.connect_to_local()   # default: localhost:8080 + grpc:50051

# ── Option B: embedded (in-process, no Docker required) ───────────────────────
# client = weaviate.connect_to_embedded()

print(client.is_ready())
```

Always close the connection when you are done:

```python
client.close()
```

Or use it as a context manager:

```python
with weaviate.connect_to_local() as client:
    # all operations inside here
    pass
# client is automatically closed on exit
```

## Creating a collection

```python
import weaviate
import weaviate.classes as wvc

with weaviate.connect_to_local() as client:

    # Delete if it already exists (useful during development)
    client.collections.delete("Documents")

    documents = client.collections.create(
        name="Documents",
        vectorizer_config=wvc.config.Configure.Vectorizer.none(),   # we supply our own vectors
        vector_index_config=wvc.config.Configure.VectorIndex.hnsw(
            distance_metric=wvc.config.VectorDistances.COSINE,
        ),
        properties=[
            wvc.config.Property(name="text",     data_type=wvc.config.DataType.TEXT),
            wvc.config.Property(name="category", data_type=wvc.config.DataType.TEXT),
            wvc.config.Property(name="source",   data_type=wvc.config.DataType.TEXT),
        ],
    )
    print("Collection created:", documents.config.get().name)
```

## Inserting objects

```python
import numpy as np

DIM = 384
rng = np.random.default_rng(42)

docs = [
    {"text": "The quick brown fox jumps over the lazy dog.", "category": "animals",  "source": "demo"},
    {"text": "Quantum computing harnesses superposition.",   "category": "science",  "source": "demo"},
    {"text": "Machine learning improves with more data.",    "category": "science",  "source": "demo"},
]

with weaviate.connect_to_local() as client:
    collection = client.collections.get("Documents")

    for doc in docs:
        vector = rng.standard_normal(DIM).astype("float32").tolist()
        collection.data.insert(
            properties={"text": doc["text"], "category": doc["category"], "source": doc["source"]},
            vector=vector,
        )

    print(f"Inserted {len(docs)} objects")
```

!!! tip "Batch insertion"
    For large datasets use `collection.data.insert_many(objects)` to send objects in a single batch request.

## Querying with near_vector

```python
with weaviate.connect_to_local() as client:
    collection = client.collections.get("Documents")

    query_vector = rng.standard_normal(DIM).astype("float32").tolist()

    results = collection.query.near_vector(
        near_vector=query_vector,
        limit=2,
        return_metadata=wvc.query.MetadataQuery(distance=True),
    )

    for obj in results.objects:
        print(f"distance={obj.metadata.distance:.4f}  text={obj.properties['text']}")
```

## Filtering alongside vector search

```python
from weaviate.classes.query import Filter

with weaviate.connect_to_local() as client:
    collection = client.collections.get("Documents")

    results = collection.query.near_vector(
        near_vector=query_vector,
        filters=Filter.by_property("category").equal("science"),
        limit=5,
    )

    for obj in results.objects:
        print(obj.properties["text"])
```

## Using the embedded client (no Docker)

```python
with weaviate.connect_to_embedded() as client:
    # Weaviate runs as a child process — no separate Docker container needed
    # Ideal for notebooks, CI, and quick prototypes
    client.collections.delete("Demo")
    demo = client.collections.create(
        name="Demo",
        vectorizer_config=wvc.config.Configure.Vectorizer.none(),
    )
    print("Embedded Weaviate ready:", client.is_ready())
```

!!! warning "Always call client.close()"
    Forgetting to close the connection can leave the embedded Weaviate process running. Prefer the `with` statement to guarantee cleanup.

## When to use Weaviate

Weaviate suits projects that need:

- A full-featured vector database with hybrid search (BM25 + vector)
- Built-in vectorisation modules (OpenAI, Cohere, etc.) configured at the collection level
- Multi-tenancy for SaaS applications
- GraphQL and REST APIs alongside the Python client

For simpler setups or when you prefer staying in the Postgres ecosystem, see [Qdrant](qdrant.md) or the [Vector Stores overview](../tools/vector-stores.md).

## Next steps

- Understand the underlying theory: [Vector Databases](../foundations/vector-databases.md)
- Compare all vector store options: [Vector Stores](../tools/vector-stores.md)
- Explore another server-based option: [Qdrant](qdrant.md)
