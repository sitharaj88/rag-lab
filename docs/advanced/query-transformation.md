# Query Transformation

The query a user types is often a poor retrieval key — ambiguous, elliptical, or mismatched to how your documents are written. Query transformation rewrites or expands the query before hitting the retriever, dramatically improving recall without changing your index.

## What you'll learn

- Four practical query transformation techniques: rewriting, multi-query, HyDE, and step-back prompting
- When each technique is most useful
- A runnable multi-query expansion example using Ollama (`llama3.2`) with deduplication
- How to combine transformed queries with your existing retriever

## Technique overview

| Technique | Idea | Best for |
|---|---|---|
| Query rewriting | Rephrase the query to better match document language | Colloquial or ambiguous queries |
| Multi-query | Generate N alternative phrasings; union results | Broad topics, underspecified questions |
| HyDE | Ask LLM to write a *hypothetical* answer; embed that | Queries far from document style |
| Step-back prompting | Generalize to a parent concept first | Multi-hop, reasoning-heavy questions |

## Query rewriting

A simple prompt instructs the LLM to rephrase the user query in a way more likely to match indexed content:

```python
# rewrite_query.py
import ollama

def rewrite_query(query: str) -> str:
    prompt = (
        "Rewrite the following user question as a precise, self-contained "
        "search query optimized for a technical document retrieval system. "
        "Output ONLY the rewritten query, no explanation.\n\n"
        f"Original: {query}"
    )
    response = ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"].strip()

print(rewrite_query("what's that thing about tokens expiring?"))
# → "JWT token expiration configuration and handling"
```

## HyDE — Hypothetical Document Embeddings

Instead of embedding the query, embed a *hypothetical answer* the LLM generates. This brings the embedding into the same semantic space as real documents.

```python
# hyde.py
import ollama
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

def generate_hypothetical_doc(query: str) -> str:
    prompt = (
        "Write a short technical paragraph (3–5 sentences) that would "
        f"directly answer this question: {query}\n"
        "Write as if it is an excerpt from documentation."
    )
    r = ollama.chat(model="llama3.2",
                    messages=[{"role": "user", "content": prompt}])
    return r["message"]["content"].strip()

# Use the hypothetical doc as the retrieval query
hyp_doc = generate_hypothetical_doc("How does attention mechanism work in transformers?")
# Now embed hyp_doc and query ChromaDB instead of the raw question
```

## Multi-query expansion (runnable example)

Multi-query generates several differently-phrased versions of the original query, retrieves for each, then deduplicates results before passing to the LLM.

```bash
pip install ollama chromadb sentence-transformers
```

```python
# multi_query.py
import ollama
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# --- Sample collection ---
docs = [
    "Dense retrieval uses vector embeddings to find semantically similar documents.",
    "Sparse retrieval with BM25 matches exact keyword terms in documents.",
    "Hybrid search fuses dense and sparse scores using Reciprocal Rank Fusion.",
    "A cross-encoder reranker scores query-document pairs jointly for higher accuracy.",
    "Chunking strategies affect retrieval quality in RAG pipelines.",
]
ids = [f"d{i}" for i in range(len(docs))]

ef = SentenceTransformerEmbeddingFunction("all-MiniLM-L6-v2")
client = chromadb.Client()
col = client.create_collection("mq_demo", embedding_function=ef)
col.add(documents=docs, ids=ids)


def generate_queries(original: str, n: int = 3) -> list[str]:
    """Use Ollama to generate N alternative phrasings."""
    prompt = (
        f"Generate {n} different search queries that retrieve information relevant to:\n"
        f"'{original}'\n\n"
        "Output ONLY the queries, one per line, no numbering or extra text."
    )
    r = ollama.chat(model="llama3.2",
                    messages=[{"role": "user", "content": prompt}])
    lines = [l.strip() for l in r["message"]["content"].splitlines() if l.strip()]
    return lines[:n]


def multi_query_retrieve(query: str, top_k: int = 3) -> list[str]:
    """Retrieve and deduplicate across multiple query variants."""
    queries = [query] + generate_queries(query)
    seen_ids: set[str] = set()
    ordered_docs: list[str] = []

    for q in queries:
        results = col.query(query_texts=[q], n_results=top_k)
        for doc_id, doc in zip(results["ids"][0], results["documents"][0]):
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                ordered_docs.append(doc)

    return ordered_docs[:top_k * 2]  # return a deduplicated superset


if __name__ == "__main__":
    original_query = "how do I make my search find both exact words and similar meanings?"
    results = multi_query_retrieve(original_query)
    print("Retrieved (deduplicated):")
    for i, r in enumerate(results, 1):
        print(f"  {i}. {r}")
```

## Step-back prompting

Step-back asks the LLM to generalize the specific question to a broader principle first, then retrieve on both the original and the generalized query.

```python
# stepback.py
import ollama

def step_back(query: str) -> str:
    prompt = (
        "Given this specific question, what is the more general concept or "
        "principle needed to answer it? Output only the general question.\n\n"
        f"Specific: {query}"
    )
    r = ollama.chat(model="llama3.2",
                    messages=[{"role": "user", "content": prompt}])
    return r["message"]["content"].strip()

specific = "Why does my HNSW index use more RAM after adding 1 million vectors?"
general = step_back(specific)
# → "How does HNSW index memory scale with dataset size?"
# Retrieve on both `specific` and `general`, union results
```

!!! tip "Combine techniques"
    Multi-query + reranking is a powerful combination: expand recall with multi-query, then use a cross-encoder (see [Reranking](../advanced/reranking.md)) to trim the merged candidate set back to a high-precision top-k.

## Next steps

- [Agentic RAG](agentic-rag.md) — let an agent decide *which* transformation to apply based on query type
- [Retrieval fundamentals](../foundations/retrieval.md) — understand recall, precision, and why query quality matters
