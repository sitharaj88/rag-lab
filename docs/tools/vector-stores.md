# Vector Stores Comparison

Choosing the right vector store determines how far your RAG application can scale and how easily it integrates with the rest of your stack. This page compares the most popular options across the dimensions that matter.

## What you'll learn

- How six major vector stores differ across deployment model, persistence, filtering, and scale
- A decision guide mapping project scenarios to the right tool
- How to migrate from a prototype store to a production one without rewriting your pipeline

---

## Comparison table

| Store | Deployment | Persistence | Metadata filtering | Scale | Best for |
|-------|-----------|-------------|-------------------|-------|---------|
| **ChromaDB** | Embedded or server | Yes (SQLite + HNSW) | Rich (`$eq`, `$in`, `$and`, …) | Up to ~1 M vectors comfortably | Prototypes, local RAG, notebooks |
| **FAISS** | In-process library | Manual (save/load index file) | None built-in | Hundreds of millions (single node) | Ultra-fast similarity search, research |
| **Qdrant** | Server or cloud | Yes (WAL + snapshots) | Rich (nested JSON payload) | Tens of millions per node, horizontal scaling | Production RAG, hybrid search |
| **Weaviate** | Server or cloud | Yes (LSM-tree) | GraphQL + BM25 hybrid | Multi-node distributed | Multi-modal, hybrid search, SaaS products |
| **Milvus** | Distributed server | Yes (etcd + object storage) | Attribute filtering | Billions of vectors | Enterprise, high-throughput indexing |
| **pgvector** | PostgreSQL extension | Yes (full ACID) | Full SQL `WHERE` clauses | Millions (single Postgres node) | Existing Postgres users, transactional workloads |

---

## Store profiles

### ChromaDB

The easiest on-ramp: `pip install chromadb` and you have a fully functional vector store with zero infrastructure.

```python
import chromadb

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("docs", metadata={"hnsw:space": "cosine"})
```

See the full guide at [chromadb.md](chromadb.md).

### FAISS

Meta's library is a toolkit of ANN algorithms, not a database. You own the serialization and metadata lookup.

```python
import faiss
import numpy as np

dim = 768
index = faiss.IndexFlatIP(dim)          # inner product (use normalized vectors for cosine)
vectors = np.random.rand(1000, dim).astype("float32")
faiss.normalize_L2(vectors)
index.add(vectors)

query = np.random.rand(1, dim).astype("float32")
faiss.normalize_L2(query)
distances, indices = index.search(query, k=5)

faiss.write_index(index, "my.index")    # persist manually
```

!!! warning "FAISS has no metadata"
    You must maintain a parallel data structure (e.g., a list or SQLite table) to map integer indices back to document IDs and metadata.

### Qdrant

Rich payload filtering, hybrid search, and a clean REST + gRPC API make Qdrant a strong production choice.

```bash
docker run -p 6333:6333 qdrant/qdrant
pip install qdrant-client
```

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

client = QdrantClient("localhost", port=6333)
client.create_collection("docs", vectors_config=VectorParams(size=768, distance=Distance.COSINE))

client.upsert("docs", points=[
    PointStruct(id=1, vector=[0.1] * 768, payload={"source": "file.pdf", "page": 3}),
])

results = client.search("docs", query_vector=[0.1] * 768, limit=5,
                        query_filter={"must": [{"key": "source", "match": {"value": "file.pdf"}}]})
```

### Weaviate

Weaviate has built-in BM25 hybrid search and a GraphQL API. It is multi-modal and schema-driven.

```bash
pip install weaviate-client
```

```python
import weaviate

client = weaviate.connect_to_local()
collection = client.collections.get("Document")
results = collection.query.hybrid(query="RAG pipeline", limit=5)
client.close()
```

### Milvus

For billion-scale workloads. Requires more infrastructure (etcd, MinIO or local storage).

```bash
pip install pymilvus
```

```python
from pymilvus import MilvusClient

client = MilvusClient("./milvus_local.db")  # lite mode for prototyping
client.create_collection("docs", dimension=768)
client.insert("docs", [{"id": 1, "vector": [0.1] * 768, "text": "sample"}])
results = client.search("docs", data=[[0.1] * 768], limit=5, output_fields=["text"])
```

### pgvector

If you already run PostgreSQL, pgvector adds vector similarity search without a new service.

```sql
-- Enable extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create table with vector column
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    content TEXT,
    source TEXT,
    embedding vector(768)
);

-- Insert
INSERT INTO documents (content, source, embedding) VALUES ('RAG doc', 'intro.pdf', '[0.1, 0.2, ...]');

-- Query (cosine similarity — note the <=> operator)
SELECT content, 1 - (embedding <=> '[0.1, 0.2, ...]'::vector) AS similarity
FROM documents
ORDER BY embedding <=> '[0.1, 0.2, ...]'::vector
LIMIT 5;
```

```bash
pip install psycopg2-binary pgvector
```

---

## Decision guide by scenario

!!! example "Scenario: local prototype or notebook demo"
    **Use ChromaDB** — embedded, zero config, rich filtering, easy to reset.

!!! example "Scenario: fastest possible retrieval, research benchmark"
    **Use FAISS** — no overhead, battle-tested at scale. Handle metadata yourself.

!!! example "Scenario: production RAG API with filtering and hybrid search"
    **Use Qdrant** — clean API, Docker or Qdrant Cloud, payload filtering, hybrid BM25+vector.

!!! example "Scenario: existing PostgreSQL stack, ACID guarantees needed"
    **Use pgvector** — one `CREATE EXTENSION` command, full SQL filtering, transactional consistency.

!!! example "Scenario: multi-modal or SaaS product with GraphQL"
    **Use Weaviate** — schema-driven, hybrid search built-in, multi-tenancy support.

!!! example "Scenario: billion-scale enterprise workload"
    **Use Milvus** — distributed architecture, GPU-accelerated indexing, enterprise support.

---

## Migrating from prototype to production

Most LangChain and LlamaIndex integrations accept a `vectorstore` argument. Swapping Chroma for Qdrant typically means changing the client initialization and collection creation — the retriever interface stays the same.

```python
# Before (Chroma)
from langchain_chroma import Chroma
vectorstore = Chroma(collection_name="docs", embedding_function=embeddings, persist_directory="./db")

# After (Qdrant) — same retriever API downstream
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient
qdrant_client = QdrantClient("localhost", port=6333)
vectorstore = Qdrant(client=qdrant_client, collection_name="docs", embeddings=embeddings)
```

---

## Next steps

- [chromadb.md](chromadb.md) — Full ChromaDB guide with filtering and custom embeddings
- [../advanced/production.md](../advanced/production.md) — Production deployment patterns for RAG systems
