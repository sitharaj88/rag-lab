# Minimal RAG in ~40 Lines

The fastest way to understand Retrieval-Augmented Generation is to build it with nothing but Python, sentence-transformers, numpy, and Ollama — no vector database, no framework.

## What you'll learn

- How to embed documents and a query into the same vector space
- How cosine similarity selects the most relevant documents
- How to build a prompt that grounds the LLM in retrieved context
- Why this in-memory approach works for small corpora and what its limits are
- The exact data flow that every RAG system follows, no matter how complex

## Prerequisites

- [Environment Setup](../getting-started/environment-setup.md)
- [Running a Local LLM with Ollama](../getting-started/local-llm-ollama.md)
- `pip install sentence-transformers numpy ollama`

## Step 1 — Understand the idea

RAG has three steps that repeat for every user question:

1. **Retrieve** — find the documents most similar to the question
2. **Augment** — paste those documents into the prompt as context
3. **Generate** — let the LLM answer using that context

In this tutorial the "database" is a plain Python list. Similarity is computed with numpy. That is the entire infrastructure.

## Step 2 — Write the script

Create a file called `minimal_rag.py` and paste the following code. Every section is explained below.

```python
# minimal_rag.py
# A complete RAG pipeline in ~40 lines — no vector DB required.

import numpy as np
from sentence_transformers import SentenceTransformer
import ollama

# ---------------------------------------------------------------------------
# 1. Knowledge base — replace or extend with your own documents.
# ---------------------------------------------------------------------------
DOCUMENTS = [
    "Retrieval-Augmented Generation (RAG) combines a retrieval step with a "
    "language model to answer questions grounded in a document corpus.",

    "ChromaDB is an open-source embedding database that stores vectors on "
    "disk and supports fast approximate nearest-neighbour search.",

    "Ollama lets you run large language models such as llama3.2 locally "
    "without sending data to an external API.",

    "Sentence-transformers produce dense vector embeddings for sentences "
    "and paragraphs. The all-MiniLM-L6-v2 model outputs 384-dimensional "
    "vectors and runs efficiently on CPU.",

    "Cosine similarity measures the angle between two vectors. A score of 1 "
    "means the vectors point in the same direction; 0 means orthogonal.",

    "Chunking splits large documents into smaller passages before embedding "
    "so that each vector represents a coherent unit of meaning.",
]

# ---------------------------------------------------------------------------
# 2. Embed every document once at startup.
# ---------------------------------------------------------------------------
MODEL_NAME = "all-MiniLM-L6-v2"
embedder = SentenceTransformer(MODEL_NAME)

print(f"Embedding {len(DOCUMENTS)} documents with {MODEL_NAME} …")
doc_embeddings = embedder.encode(DOCUMENTS, normalize_embeddings=True)
# doc_embeddings shape: (num_docs, 384)


# ---------------------------------------------------------------------------
# 3. Retrieve the top-k most relevant documents for a query.
# ---------------------------------------------------------------------------
def retrieve(query: str, top_k: int = 2) -> list[str]:
    """Return the top_k documents most similar to the query."""
    query_embedding = embedder.encode([query], normalize_embeddings=True)
    # Because both sides are L2-normalised, dot product == cosine similarity.
    scores = (doc_embeddings @ query_embedding.T).squeeze()
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [DOCUMENTS[i] for i in top_indices]


# ---------------------------------------------------------------------------
# 4. Build a grounded prompt and call Ollama.
# ---------------------------------------------------------------------------
def answer(query: str, top_k: int = 2) -> str:
    """Retrieve context, build a prompt, and generate an answer."""
    context_docs = retrieve(query, top_k=top_k)
    context = "\n\n".join(f"- {doc}" for doc in context_docs)

    prompt = (
        "You are a helpful assistant. Answer the question using ONLY the "
        "context provided below. If the context does not contain enough "
        "information, say so.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )

    response = ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"]


# ---------------------------------------------------------------------------
# 5. Interactive loop — type a question, press Enter, type 'quit' to exit.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\nMinimal RAG ready. Type a question (or 'quit' to exit).\n")
    while True:
        query = input("You: ").strip()
        if not query or query.lower() in {"quit", "exit", "q"}:
            break
        print(f"\nAssistant: {answer(query)}\n")
```

## Step 3 — Run it

```bash
python minimal_rag.py
```

You should see the embedding step complete in a few seconds, then a prompt:

```
Minimal RAG ready. Type a question (or 'quit' to exit).

You:
```

Try:

```
You: What is cosine similarity?
You: How does Ollama help with privacy?
You: What is chunking and why does it matter?
```

## Step 4 — Understand each part

### Normalised embeddings and the dot-product trick

```python
doc_embeddings = embedder.encode(DOCUMENTS, normalize_embeddings=True)
scores = (doc_embeddings @ query_embedding.T).squeeze()
```

`normalize_embeddings=True` sets every vector to unit length. For unit vectors, the dot product equals the cosine similarity, so one matrix multiplication ranks all documents at once. No loop required.

### Why `np.argsort(scores)[::-1][:top_k]`?

`np.argsort` returns indices that would sort the array in ascending order. `[::-1]` reverses that to descending (highest score first). `[:top_k]` keeps only the top results.

### The prompt template

```python
prompt = (
    "You are a helpful assistant. Answer the question using ONLY the "
    "context provided below. …"
    f"Context:\n{context}\n\n"
    f"Question: {query}\n\nAnswer:"
)
```

The word **ONLY** is important: it instructs the model to stay within the retrieved context rather than hallucinating from training memory. See [Prompting for RAG](../foundations/prompting-rag.md) for a deeper treatment.

### The `ollama.chat` call

```python
response = ollama.chat(
    model="llama3.2",
    messages=[{"role": "user", "content": prompt}],
)
```

This sends the prompt to the locally running Ollama daemon and returns the generated text. No internet required.

## Step 5 — Experiment

!!! tip "Things to try"
    - Add more documents to `DOCUMENTS` and see how retrieval changes.
    - Change `top_k` from 2 to 3 or 1 and observe the effect on answer quality.
    - Ask a question whose answer is not in any document — notice how the model says it doesn't know.
    - Print `scores` inside `retrieve()` to see the raw similarity values.

## Limitations of this approach

| Limitation | Impact | Solution in later tutorials |
|------------|--------|-----------------------------|
| Documents live in RAM | Lost on restart | ChromaDB persistence (Tutorial 02) |
| Embeddings recomputed at startup | Slow for large corpora | Ingest once, query many times (Tutorial 02) |
| No metadata or citations | Hard to trace answers | Page-aware chunking (Tutorial 04) |
| Single script, no UI | Hard to share | Streamlit chat app (Tutorial 03) |

!!! info "How retrieval works under the hood"
    For a deeper explanation of cosine similarity, embedding spaces, and approximate nearest-neighbour search, read [Retrieval](../foundations/retrieval.md).

## Next steps

- [Tutorial 02 — RAG with Ollama + ChromaDB](02-rag-with-ollama-chroma.md): add a persistent vector store so you only embed documents once.
- [Retrieval](../foundations/retrieval.md): understand the theory behind similarity search.
