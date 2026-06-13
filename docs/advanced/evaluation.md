# Evaluation

You cannot improve what you do not measure. RAG evaluation covers two distinct concerns: did retrieval find the right documents, and did generation produce a faithful, relevant answer from those documents?

## What you'll learn

- Retrieval metrics: hit rate, MRR, and Recall@K
- Generation metrics: faithfulness, answer relevance, and the RAG triad
- How to build a minimal evaluation dataset
- A runnable Hit Rate@K calculator
- LLM-as-judge with Ollama for faithfulness scoring

## Retrieval metrics

| Metric | Formula | Measures |
|---|---|---|
| Hit Rate@K | `relevant docs found in top-K / total queries` | Did the right doc appear at all? |
| MRR@K | `mean of 1/rank for first relevant doc` | How high does the relevant doc rank? |
| Recall@K | `relevant docs in top-K / total relevant docs` | What fraction of all relevant docs were found? |
| Precision@K | `relevant docs in top-K / K` | How noisy is the top-K set? |

### Runnable Hit Rate@K example

```python
# eval_retrieval.py
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# Evaluation dataset: list of (query, expected_doc_id)
eval_set = [
    ("how does HNSW indexing work?", "d0"),
    ("what is reciprocal rank fusion?", "d1"),
    ("explain cross-encoder reranking", "d2"),
    ("how are embeddings used in retrieval?", "d3"),
]

# Build collection
docs = [
    "HNSW builds a hierarchical graph for fast approximate nearest-neighbor search.",
    "Reciprocal Rank Fusion merges ranked lists by summing reciprocal rank scores.",
    "Cross-encoders rerank query-document pairs by encoding them jointly.",
    "Dense embeddings represent text as vectors; cosine similarity finds nearest neighbors.",
]
ids = [f"d{i}" for i in range(len(docs))]

ef = SentenceTransformerEmbeddingFunction("all-MiniLM-L6-v2")
client = chromadb.Client()
col = client.create_collection("eval_demo", embedding_function=ef)
col.add(documents=docs, ids=ids)


def hit_rate_at_k(col, eval_set: list[tuple[str, str]], k: int = 3) -> float:
    hits = 0
    for query, expected_id in eval_set:
        results = col.query(query_texts=[query], n_results=k)
        retrieved_ids = results["ids"][0]
        if expected_id in retrieved_ids:
            hits += 1
    return hits / len(eval_set)


def mrr_at_k(col, eval_set: list[tuple[str, str]], k: int = 3) -> float:
    rr_sum = 0.0
    for query, expected_id in eval_set:
        results = col.query(query_texts=[query], n_results=k)
        retrieved_ids = results["ids"][0]
        if expected_id in retrieved_ids:
            rank = retrieved_ids.index(expected_id) + 1  # 1-indexed
            rr_sum += 1.0 / rank
    return rr_sum / len(eval_set)


for k in [1, 2, 3]:
    hr = hit_rate_at_k(col, eval_set, k=k)
    mrr = mrr_at_k(col, eval_set, k=k)
    print(f"K={k}  Hit Rate={hr:.2f}  MRR={mrr:.3f}")
```

## Generation metrics and the RAG triad

The **RAG triad** (coined by TruLens/Ragas) defines three axes:

```
         Context Relevance
              ↑
Query ──────► Context ──────► Answer
              ↓
         Groundedness    Answer Relevance
```

| Metric | Question | How to measure |
|---|---|---|
| **Context relevance** | Is the retrieved context relevant to the query? | LLM-as-judge or NLI model |
| **Groundedness / Faithfulness** | Is the answer supported by the context? | LLM-as-judge; check each claim |
| **Answer relevance** | Does the answer actually address the query? | Embed answer + query; cosine sim |

## LLM-as-judge for faithfulness

```python
# llm_judge.py
import ollama

FAITHFULNESS_PROMPT = """You are an evaluator. Given a context and an answer, 
determine whether every claim in the answer is supported by the context.

Context:
{context}

Answer:
{answer}

Rate faithfulness from 0.0 (completely unfaithful) to 1.0 (fully grounded).
Output ONLY a JSON object: {{"score": <float>, "reason": "<brief reason>"}}"""

def score_faithfulness(context: str, answer: str) -> dict:
    import json
    prompt = FAITHFULNESS_PROMPT.format(context=context, answer=answer)
    r = ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": prompt}]
    )
    raw = r["message"]["content"].strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


context = "HNSW builds a hierarchical graph structure for efficient ANN search."
answer = "HNSW uses a hierarchical graph, making it fast for nearest-neighbor queries."
result = score_faithfulness(context, answer)
print(f"Faithfulness: {result['score']}  Reason: {result['reason']}")
```

## Building an evaluation dataset

!!! tip "Start small but representative"
    20–50 high-quality (query, context, expected answer) triples are enough to catch regressions. Curate from real user queries when possible.

A minimal eval dataset structure:

```python
eval_data = [
    {
        "query": "What indexing algorithm does ChromaDB use?",
        "expected_doc_id": "d0",
        "reference_answer": "ChromaDB uses HNSW for approximate nearest-neighbor indexing.",
    },
    # ... more examples
]
```

Store as JSON or CSV, commit to your repo, and run evaluation in CI on every pipeline change.

## Ragas — automated RAG evaluation

[Ragas](https://docs.ragas.io) automates the RAG triad with a consistent framework. It can use local models via Ollama as the judge LLM, making it fully local:

```bash
pip install ragas
```

```python
# ragas_eval.py (sketch — see Ragas docs for full setup)
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall
# Configure Ragas to use a local Ollama model as the LLM judge
# See Ragas documentation for LangchainLLM / custom LLM wrappers
```

!!! note "Hosted alternatives"
    TruLens, UpTrain, and LangSmith all offer RAG evaluation dashboards. For fully local evaluation without API calls, the LLM-as-judge pattern above or Ragas with a local LLM judge is the recommended path.

## Next steps

- [Production](production.md) — integrate evaluation into CI and monitor quality in deployment
- [Prompting for RAG](../foundations/prompting-rag.md) — improve faithfulness at the generation stage
