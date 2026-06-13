# pgvector

pgvector is a PostgreSQL extension that adds a native `vector` column type and approximate nearest-neighbour index support. It is the right choice when your application already uses Postgres and you want to store embeddings in the same database as the rest of your relational data.

## What you'll learn

- Installing and enabling the `pgvector` extension
- Creating tables with `vector(n)` columns
- Inserting embeddings and querying with the three distance operators
- Connecting from Python with `psycopg` and the `pgvector` adapter
- When pgvector is the right (and wrong) tool for the job

## Install

```bash
# Python dependencies
pip install psycopg[binary] pgvector numpy
```

The pgvector extension must be installed on the Postgres server. The easiest local approach is Docker:

```bash
docker run --name pgvector-demo \
  -e POSTGRES_PASSWORD=password \
  -p 5432:5432 \
  -d pgvector/pgvector:pg16
```

The `pgvector/pgvector` image ships with the extension pre-built.

## Enable the extension

Connect to your database (e.g. `psql -h localhost -U postgres`) and run once:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

## Schema setup

```sql
-- Create a table with a 384-dimensional embedding column
CREATE TABLE documents (
    id        SERIAL PRIMARY KEY,
    content   TEXT        NOT NULL,
    category  TEXT,
    source    TEXT,
    embedding vector(384)          -- dimension must match your model
);

-- Optional: HNSW index for approximate nearest-neighbour (faster at scale)
CREATE INDEX ON documents USING hnsw (embedding vector_cosine_ops);
-- For L2:          vector_l2_ops
-- For inner product: vector_ip_ops
```

!!! tip "When to add an HNSW index"
    For fewer than ~100 k rows a sequential scan is fast enough. Add the HNSW (or IVFFlat) index once query latency becomes noticeable.

## Distance operators

| Operator | Metric | Lower = more similar? |
|---|---|---|
| `<=>` | Cosine distance | Yes |
| `<->` | L2 (Euclidean) distance | Yes |
| `<#>` | Negative inner product | Yes (more negative = closer) |

## Inserting and querying from Python

```python
import numpy as np
import psycopg
from pgvector.psycopg import register_vector

DIM = 384
rng = np.random.default_rng(42)

# ── 1. Connect ─────────────────────────────────────────────────────────────────
conn = psycopg.connect(
    host="localhost",
    dbname="postgres",
    user="postgres",
    password="password",
)

# ── 2. Register the vector type so psycopg knows how to serialise it ──────────
register_vector(conn)

# ── 3. Ensure extension & table exist ─────────────────────────────────────────
with conn.cursor() as cur:
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id        SERIAL PRIMARY KEY,
            content   TEXT,
            category  TEXT,
            embedding vector(%s)
        )
    """, (DIM,))
    conn.commit()

# ── 4. Insert documents with embeddings ───────────────────────────────────────
docs = [
    ("The quick brown fox jumps over the lazy dog.", "animals"),
    ("A fast auburn fox leaps above a sleepy hound.", "animals"),
    ("Quantum computing harnesses superposition.",    "science"),
    ("Machine learning improves with more data.",     "science"),
]

with conn.cursor() as cur:
    for text, category in docs:
        embedding = rng.standard_normal(DIM).astype("float32")
        cur.execute(
            "INSERT INTO documents (content, category, embedding) VALUES (%s, %s, %s)",
            (text, category, embedding),
        )
    conn.commit()

print("Inserted", len(docs), "documents")
```

## Querying nearest neighbours

```python
query_vector = rng.standard_normal(DIM).astype("float32")

with conn.cursor() as cur:
    cur.execute("""
        SELECT id, content, category,
               1 - (embedding <=> %s::vector) AS cosine_similarity
        FROM   documents
        ORDER  BY embedding <=> %s::vector
        LIMIT  3
    """, (query_vector, query_vector))

    rows = cur.fetchall()

for row in rows:
    print(f"id={row[0]}  similarity={row[3]:.4f}  text={row[1]}")

conn.close()
```

!!! note "Cosine distance vs cosine similarity"
    The `<=>` operator returns **cosine distance** (0 = identical, 2 = opposite). Subtract from 1 to get similarity, or just `ORDER BY embedding <=> query` to rank by closeness.

## Filtering with SQL

Because embeddings live in a regular Postgres table, you get full SQL for free:

```sql
SELECT content, 1 - (embedding <=> '[...]'::vector) AS score
FROM   documents
WHERE  category = 'science'
ORDER  BY embedding <=> '[...]'::vector
LIMIT  5;
```

No special filter syntax — just a `WHERE` clause.

## When to use pgvector

| Situation | Recommendation |
|---|---|
| Already running Postgres | Strong fit — single database, familiar tooling |
| Need transactional consistency between embeddings and relational data | Strong fit |
| Starting fresh with no existing Postgres | Consider a dedicated vector DB ([FAISS](faiss.md) for in-process, Qdrant for server) |
| Need billions of vectors with sub-10 ms p99 latency | Consider a purpose-built vector store (see [Vector Stores](../tools/vector-stores.md)) |

pgvector keeps your stack simple when Postgres is already the source of truth.

## Next steps

- Compare all vector store options: [Vector Stores](../tools/vector-stores.md)
- Production deployment considerations: [Production](../advanced/production.md)
- In-process alternative with no server: [FAISS](faiss.md)
