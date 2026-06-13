# FAISS

FAISS (Facebook AI Similarity Search) is a high-performance library for efficient similarity search and clustering of dense vectors. It runs entirely in-process as a library — there is no server — so you own id mapping and metadata storage yourself.

## What you'll learn

- Installing and importing `faiss-cpu`
- Building a flat index with cosine similarity via normalized `IndexFlatIP`
- Adding vectors and querying for nearest neighbours
- Managing ids and metadata with a parallel Python structure
- Persisting an index to disk with `faiss.write_index` / `faiss.read_index`

## Install

```bash
pip install faiss-cpu numpy
```

!!! note "GPU variant"
    If you have a CUDA-capable GPU, install `faiss-gpu` instead of `faiss-cpu`. The Python API is identical.

## Core concepts

FAISS operates directly on `numpy` arrays of `float32`. It does not store your text, document ids, or any metadata — only raw float vectors. You must maintain a parallel data structure (list, dict, database) that maps a FAISS integer index position to your real id and payload.

## Quick start — cosine similarity index

```python
import faiss
import numpy as np

# ── 1. Dimension of your embedding model ──────────────────────────────────────
DIM = 384  # e.g. all-MiniLM-L6-v2

# ── 2. Create a flat inner-product index ─────────────────────────────────────
#    Cosine similarity = inner product of unit-norm vectors
index = faiss.IndexFlatIP(DIM)

# ── 3. Sample documents & embeddings ─────────────────────────────────────────
docs = [
    "The quick brown fox jumps over the lazy dog.",
    "A fast auburn fox leaps above a sleepy hound.",
    "Quantum computing harnesses superposition and entanglement.",
    "Machine learning models improve with more training data.",
]

# Simulate embeddings (replace with your real encoder)
rng = np.random.default_rng(42)
raw_embeddings = rng.standard_normal((len(docs), DIM)).astype("float32")

# ── 4. Normalize to unit length (required for cosine via IndexFlatIP) ─────────
faiss.normalize_L2(raw_embeddings)        # in-place normalisation

# ── 5. Add to index — FAISS assigns positions 0, 1, 2, … ─────────────────────
index.add(raw_embeddings)
print(f"Index now holds {index.ntotal} vectors")

# ── 6. Keep a parallel metadata store keyed by FAISS position ────────────────
metadata = {i: {"text": doc, "source": "demo"} for i, doc in enumerate(docs)}
```

## Querying

```python
# Encode & normalise the query the same way as the indexed vectors
query_raw = rng.standard_normal((1, DIM)).astype("float32")
faiss.normalize_L2(query_raw)

k = 2  # top-k results
distances, indices = index.search(query_raw, k)

print("Top results:")
for rank, (score, idx) in enumerate(zip(distances[0], indices[0]), start=1):
    print(f"  {rank}. score={score:.4f}  text={metadata[idx]['text']}")
```

`index.search` returns two arrays of shape `(n_queries, k)`:

- `distances` — inner-product scores (higher = more similar for cosine)
- `indices` — FAISS integer positions (map back through `metadata`)

!!! tip "L2 distance instead of cosine"
    Swap `IndexFlatIP` for `IndexFlatL2` and **skip** the `normalize_L2` step. L2 returns distances where **lower** is better.

## Persistence

```python
# Save
faiss.write_index(index, "my_index.faiss")

# Load later
loaded_index = faiss.read_index("my_index.faiss")

# You must separately persist your metadata dict (e.g. with json or pickle)
import json, pathlib
pathlib.Path("metadata.json").write_text(json.dumps(metadata))
```

!!! warning "Metadata is not saved inside the `.faiss` file"
    `write_index` serialises only the vectors. Always save your `metadata` dict alongside the index file and reload both together.

## Scaling up — IVF indexes

For millions of vectors, `IndexFlatIP` becomes slow (it does a full scan). Consider an approximate index:

```python
# IVF with flat quantiser — trades some recall for speed
quantiser = faiss.IndexFlatIP(DIM)
ivf_index = faiss.IndexIVFFlat(quantiser, DIM, 100, faiss.METRIC_INNER_PRODUCT)

# IVF indexes must be trained before adding vectors
ivf_index.train(raw_embeddings)
ivf_index.add(raw_embeddings)

ivf_index.nprobe = 10          # cells to visit at query time (recall vs speed)
D, I = ivf_index.search(query_raw, k)
```

## When to use FAISS

| Situation | Recommendation |
|---|---|
| Prototype / small corpus (< 1 M vectors) | `IndexFlatIP` — exact, no training |
| Large corpus, latency matters | `IndexIVFFlat` or `IndexHNSWFlat` |
| Already running a vector-database server | Prefer [Qdrant](qdrant.md) or a managed service |
| Need metadata filtering or persistence handled for you | Use a purpose-built vector store (see [Vector Stores](../tools/vector-stores.md)) |

FAISS is ideal when you want maximum control, minimal dependencies, and you are comfortable managing ids and metadata yourself.

## Next steps

- Explore server-based alternatives: [Qdrant SDK](qdrant.md)
- Understand the theory: [Vector Databases](../foundations/vector-databases.md)
- Compare all vector store options: [Vector Stores](../tools/vector-stores.md)
