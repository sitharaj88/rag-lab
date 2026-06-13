# Generation

Generation is the online half of RAG: given a user question, embed it, retrieve the most relevant chunks, build a grounded prompt, and call the local LLM — all in a few hundred milliseconds.

## What you'll learn

- How to embed a query with the same model used at index time
- How to retrieve the top-k chunks from ChromaDB
- How to build a context-aware prompt that keeps the LLM grounded
- How to call Ollama for both blocking and streaming responses
- How to return cited sources alongside the generated answer

---

## The query-time flow

```
question
   │
   ▼
embed_query()          ← same model as ingest (all-MiniLM-L6-v2)
   │
   ▼
chroma.query(top_k)    ← cosine similarity search
   │
   ▼
build_prompt()         ← inject retrieved passages
   │
   ▼
ollama.chat()          ← llama3.2 running locally
   │
   ▼
answer + sources
```

---

## Full `answer()` function

Install dependencies (already present if you ran the indexing pipeline):

```bash
pip install chromadb sentence-transformers ollama
```

```python
"""generation.py — online RAG query against a ChromaDB index."""
from __future__ import annotations
import textwrap

import chromadb
import ollama
from sentence_transformers import SentenceTransformer

# ── Configuration ──────────────────────────────────────────────────────────────
DB_PATH = "./chroma_db"
COLLECTION_NAME = "rag_docs"
EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama3.2"
TOP_K = 5

# Initialise once at module load (cheap after the first call)
_embedder: SentenceTransformer | None = None
_collection: chromadb.Collection | None = None


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def _get_collection() -> chromadb.Collection:
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=DB_PATH)
        _collection = client.get_collection(COLLECTION_NAME)
    return _collection


# ── Retrieval ──────────────────────────────────────────────────────────────────

def retrieve(question: str, top_k: int = TOP_K) -> list[dict]:
    """Embed *question* and return the top-k chunks with metadata."""
    embedder = _get_embedder()
    collection = _get_collection()

    query_vector = embedder.encode(
        [question],
        normalize_embeddings=True,   # must match ingest normalisation
    ).tolist()

    results = collection.query(
        query_embeddings=query_vector,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text": doc,
            "source": meta.get("source", "unknown"),
            "chunk_index": meta.get("chunk_index", -1),
            "distance": round(dist, 4),
        })
    return chunks


# ── Prompt building ────────────────────────────────────────────────────────────

def build_prompt(question: str, chunks: list[dict]) -> str:
    """Assemble a grounded prompt from retrieved passages."""
    context_blocks = []
    for i, chunk in enumerate(chunks, 1):
        context_blocks.append(
            f"[Source {i}: {chunk['source']} chunk {chunk['chunk_index']}]\n"
            f"{chunk['text']}"
        )
    context = "\n\n".join(context_blocks)

    return textwrap.dedent(f"""
        You are a helpful assistant. Answer the question using ONLY the provided
        context. If the context does not contain enough information, say so — do
        not invent facts.

        Context:
        {context}

        Question: {question}

        Answer (cite sources by their [Source N] label):
    """).strip()


# ── Blocking generation ────────────────────────────────────────────────────────

def answer(question: str, top_k: int = TOP_K) -> dict:
    """Return {"answer": str, "sources": list[dict]}.

    Calls Ollama in blocking mode — waits for the full response.
    """
    chunks = retrieve(question, top_k=top_k)
    prompt = build_prompt(question, chunks)

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    answer_text = response["message"]["content"]

    sources = [
        {"source": c["source"], "chunk_index": c["chunk_index"],
         "distance": c["distance"]}
        for c in chunks
    ]
    return {"answer": answer_text, "sources": sources}


# ── Streaming generation ───────────────────────────────────────────────────────

def answer_stream(question: str, top_k: int = TOP_K):
    """Generator that yields answer tokens as they arrive from Ollama.

    Usage:
        for token in answer_stream("What is RAG?"):
            print(token, end="", flush=True)
    """
    chunks = retrieve(question, top_k=top_k)
    prompt = build_prompt(question, chunks)

    stream = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    for part in stream:
        token = part["message"]["content"]
        if token:
            yield token

    # After the generator is exhausted, callers can request sources separately
    # via retrieve() if needed.


# ── CLI demo ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    question = " ".join(sys.argv[1:]) or "What is retrieval-augmented generation?"

    print(f"\nQuestion: {question}\n")
    print("Answer (streaming):\n")

    for token in answer_stream(question):
        print(token, end="", flush=True)

    print("\n\n--- Sources ---")
    for src in retrieve(question):
        print(f"  {src['source']}  chunk={src['chunk_index']}  dist={src['distance']}")
```

Run an ad-hoc query:

```bash
python generation.py "How does chunking affect retrieval quality?"
```

---

## Returning citations in a structured way

!!! example "Structured answer with sources"
    ```python
    result = answer("What embedding model does this project use?")

    print(result["answer"])
    # → "This project uses all-MiniLM-L6-v2 [Source 1] …"

    for s in result["sources"]:
        print(s["source"], "–", s["distance"])
    # → ./data/architecture.md – 0.1823
    ```

!!! tip "Distance interpretation"
    ChromaDB returns **cosine distance** (0 = identical, 2 = opposite). A distance below ~0.4 generally indicates a strong semantic match; above 0.8 the chunk is probably irrelevant and you can filter it out before building the prompt.

---

## Improving answer quality

!!! note "Temperature and system prompts"
    Pass `options={"temperature": 0.2}` to `ollama.chat` for more deterministic, fact-bound answers. Higher temperature suits creative tasks; lower is better for Q&A over documents.

!!! warning "Context window limits"
    `llama3.2` has a default context window of 128 k tokens, but each extra chunk adds latency. Keep `top_k` between 3 and 8 for a good speed/quality trade-off.

---

## Next steps

- [Foundations — Prompting for RAG](../foundations/prompting-rag.md) — learn how prompt structure shapes answer grounding
- [Tutorial 02 — RAG with Ollama + ChromaDB](../tutorials/02-rag-with-ollama-chroma.md) — full end-to-end walkthrough that uses this exact generation module
