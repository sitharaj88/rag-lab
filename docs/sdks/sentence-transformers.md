# sentence-transformers SDK

sentence-transformers is a Python library that wraps Hugging Face encoders with a clean, task-oriented API for producing dense embeddings, sparse embeddings, and cross-encoder reranking scores — the three core signals in modern RAG pipelines.

## What you'll learn

- Generating dense embeddings with `SentenceTransformer` and the `encode()` method
- Computing cosine similarity with `util.cos_sim`
- Reranking candidate passages with `CrossEncoder`
- Producing sparse embeddings with `SparseEncoder` (new in v5)
- How to pick the right model for your task

## Install

```bash
pip install sentence-transformers
```

PyTorch is installed automatically as a dependency. For GPU acceleration install a CUDA-enabled build of PyTorch first.

## Dense embeddings with SentenceTransformer

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

documents = [
    "Retrieval-augmented generation grounds LLM answers in external knowledge.",
    "Vector databases store and search high-dimensional embeddings efficiently.",
    "Chunking splits documents into pieces that fit the model's context window.",
]

# encode() accepts a list via the `inputs` keyword (v5+)
embeddings = model.encode(inputs=documents, normalize_embeddings=True)

print(embeddings.shape)                   # (3, 384)
print(model.get_embedding_dimension())    # 384
```

!!! warning "Renamed parameters in v5"
    sentence-transformers v5 renamed two commonly used APIs:

    | Old (v4, deprecated) | New (v5+) |
    |---|---|
    | `model.encode(sentences=[...])` | `model.encode(inputs=[...])` |
    | `model.get_sentence_embedding_dimension()` | `model.get_embedding_dimension()` |

    Both old names still function but emit `DeprecationWarning`. Migrate now to keep
    your code forward-compatible. See [versions and deprecations](versions-and-deprecations.md).

## Cosine similarity

`normalize_embeddings=True` produces unit-length vectors, so a dot product equals cosine similarity. The library also ships `util.cos_sim` for convenience.

```python
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

query = "How does RAG reduce hallucinations?"
candidates = [
    "RAG retrieves relevant documents and feeds them to the LLM as context.",
    "Hallucinations occur when models generate plausible but false information.",
    "PyTorch is a deep learning framework developed by Meta.",
]

query_emb = model.encode(inputs=[query], normalize_embeddings=True)
cand_embs = model.encode(inputs=candidates, normalize_embeddings=True)

scores = util.cos_sim(query_emb, cand_embs)
print(scores)  # tensor([[0.72, 0.43, 0.08]])
```

## CrossEncoder for reranking

A `CrossEncoder` takes a (query, passage) pair and outputs a relevance score. It is more accurate than cosine similarity but slower — use it to rerank a short-list retrieved by a `SentenceTransformer`.

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

query = "How does RAG reduce hallucinations?"
passages = [
    "RAG retrieves relevant documents and feeds them to the LLM as context.",
    "Hallucinations occur when models generate plausible but false information.",
    "PyTorch is a deep learning framework developed by Meta.",
]

pairs = [(query, p) for p in passages]
scores = reranker.predict(pairs)

ranked = sorted(zip(scores, passages), reverse=True)
for score, passage in ranked:
    print(f"{score:.3f}  {passage}")
```

See [reranking strategies](../advanced/reranking.md) for a full two-stage retrieval pipeline.

## SparseEncoder (new in v5)

`SparseEncoder` produces SPLADE-style sparse vectors that complement dense retrieval. Supported by vector stores that accept sparse indices.

```python
from sentence_transformers import SparseEncoder

sparse_model = SparseEncoder("naver/splade-cocondenser-selfdistil")

sparse_vecs = sparse_model.encode(
    inputs=["RAG combines retrieval with generation."],
    normalize_embeddings=True,
)
print(type(sparse_vecs))  # scipy sparse matrix or dict, depending on backend
```

## Choosing a model

| Use case | Recommended model |
|---|---|
| General-purpose English embeddings | `all-MiniLM-L6-v2` (fast, 384-dim) |
| Higher accuracy English embeddings | `all-mpnet-base-v2` (768-dim) |
| Multilingual embeddings | `paraphrase-multilingual-MiniLM-L12-v2` |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Sparse retrieval | `naver/splade-cocondenser-selfdistil` |

Browse the full leaderboard at [sbert.net/docs/sentence_transformer/pretrained_models.html](https://www.sbert.net/docs/sentence_transformer/pretrained_models.html).

## Next steps

- Understand the theory behind embeddings in [Embeddings fundamentals](../foundations/embeddings.md)
- Build a two-stage retriever with [reranking](../advanced/reranking.md)
- Track API changes and migration paths in [versions and deprecations](versions-and-deprecations.md)
