# Embeddings

Embeddings convert text into dense numerical vectors so that semantically similar sentences end up close together in vector space — this is the foundation of every RAG retrieval step.

## What you'll learn

- What an embedding vector is and why it captures meaning
- How cosine similarity measures semantic closeness
- How to embed sentences with `sentence-transformers`
- What embedding dimensions mean for quality and speed
- How to choose and normalize an embedding model

---

## What are embeddings?

An **embedding** is a list of floating-point numbers — for `all-MiniLM-L6-v2` that list has 384 numbers. The model is trained so that text with similar meaning maps to nearby points in this 384-dimensional space.

```text
"The cat sat on the mat."  →  [0.021, -0.134, 0.089, ...]  (384 dims)
"A kitten rested on a rug." →  [0.019, -0.128, 0.091, ...]  (384 dims)
"Quarterly revenue grew."   →  [-0.201,  0.332, -0.445, ...] (384 dims)
```

The first two sentences are semantically close; the third is far away.

---

## Cosine similarity

**Cosine similarity** measures the angle between two vectors, ignoring magnitude. It ranges from -1 (opposite) through 0 (orthogonal) to 1 (identical direction).

```python
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")  # 384-dim, ~90 MB

sentences = [
    "The cat sat on the mat.",
    "A kitten rested on a rug.",
    "Quarterly revenue grew 12 percent.",
]

embeddings = model.encode(sentences, normalize_embeddings=True)

def cosine_sim(a, b):
    # With normalized vectors, dot product == cosine similarity
    return float(np.dot(a, b))

print(cosine_sim(embeddings[0], embeddings[1]))  # ~0.85 — very similar
print(cosine_sim(embeddings[0], embeddings[2]))  # ~0.10 — unrelated
```

!!! note "Always normalize"
    Passing `normalize_embeddings=True` ensures vectors have unit length, so a simple dot product equals cosine similarity. ChromaDB's cosine metric expects this automatically, but it is good practice to normalize before any manual math.

---

## Dimensions and model choice

| Model | Dims | Size | Speed | Use case |
|---|---|---|---|---|
| `all-MiniLM-L6-v2` | 384 | ~90 MB | Fast (CPU-friendly) | Default RAG, dev work |
| `all-mpnet-base-v2` | 768 | ~420 MB | Medium | Higher quality retrieval |
| `bge-large-en-v1.5` | 1024 | ~1.3 GB | Slow | Best-in-class quality |

!!! tip "Start with `all-MiniLM-L6-v2`"
    It runs on CPU in milliseconds per sentence and scores well on retrieval benchmarks. Switch to a larger model only when retrieval quality becomes the bottleneck.

---

## Embedding a batch of documents

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

docs = [
    "ChromaDB stores vectors locally with no server required.",
    "Ollama runs open-source LLMs on your laptop.",
    "Sentence transformers produce dense semantic embeddings.",
]

# Returns a numpy array of shape (3, 384)
doc_embeddings = model.encode(docs, normalize_embeddings=True, batch_size=32)
print(doc_embeddings.shape)  # (3, 384)
```

!!! example "Batch encoding is faster"
    Always pass a list rather than a loop of single strings. The model processes sentences in parallel on GPU or uses vectorized CPU ops.

---

## Choosing the right model

- **Same model for indexing and querying** — this is mandatory. Mixing models produces garbage similarity scores.
- **Domain matters** — general English models work for most RAG tasks. For code, medical, or legal text, look for domain-fine-tuned variants.
- **Check the MTEB leaderboard** at `huggingface.co/spaces/mteb/leaderboard` for up-to-date quality rankings.

---

## Next steps

- [Vector Databases](vector-databases.md) — store and search your embeddings at scale
- [Embedding Models (tools)](../tools/embedding-models.md) — compare popular models and how to swap them in
