# Reranking and Rank Fusion in Information Retrieval

Reranking and rank fusion are two complementary techniques that improve the
precision of a retrieval pipeline after the initial candidate set has been
gathered.

## Why a Two-Stage Pipeline?

First-stage retrievers (BM25, dense ANN) are optimised for *speed* — they
must search millions of documents in milliseconds.  To achieve this, they use
lightweight representations (bag-of-words statistics or fixed-size vectors)
that lose some relevance signal.

A second-stage *reranker* operates on a much smaller candidate set (typically
20–100 documents) and can afford to use a heavier model that jointly encodes
the query and each candidate.  This two-stage design keeps overall latency low
while pushing recall and precision close to the theoretical maximum for the
corpus.

## Reciprocal Rank Fusion (RRF)

RRF was introduced by Cormack, Clarke, and Buettcher (SIGIR 2009) as a simple,
parameter-free method to combine ranked lists from multiple retrieval systems.

The formula for a single document `d` across `n` ranked lists is:

```
RRF(d) = sum_{i=1}^{n}  1 / (k + rank_i(d))
```

- `rank_i(d)` is the 1-based position of document `d` in list `i` (documents
  not present in list `i` are typically assigned rank = infinity, contributing
  0 to the sum).
- `k` is a smoothing constant, usually 60.  It prevents a document ranked #1
  in one list from completely swamping all other signals.

RRF properties:
- No need to normalise scores across systems (only ranks are used).
- Robust to outlier lists — one poor retriever does not dominate.
- Works with any number of input lists.
- Consistently beats learned fusion methods in zero-shot settings.

## Cross-Encoder Rerankers

A cross-encoder is a transformer model that takes a *(query, passage)* pair
concatenated as a single sequence and outputs a relevance score.  Unlike a
bi-encoder (embedding model) that encodes query and passage independently, the
cross-encoder's attention layers can model fine-grained interactions:

- Exact phrase matching within the passage.
- Negation ("Python is *not* faster than C" should score low for "languages
  faster than C").
- Entity co-reference ("it", "they" resolved against antecedents in the passage).

The `sentence-transformers` library exposes cross-encoders via `CrossEncoder`:

```python
from sentence_transformers import CrossEncoder

model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
scores = model.predict([
    ("What is BM25?", "BM25 is a ranking function used by search engines."),
    ("What is BM25?", "ChromaDB is a vector database."),
])
# scores == [8.3, -4.1]  (raw logits, higher = more relevant)
```

The ms-marco models were trained on 500k human-labelled query-passage pairs
from the MS MARCO dataset, giving strong out-of-the-box performance for
English passage ranking.

## Comparing Retrieval Strategies

| Strategy          | Keyword match | Semantic match | Latency | Quality |
|-------------------|---------------|----------------|---------|---------|
| BM25 only         | Excellent     | Poor           | Low     | Moderate|
| Dense only        | Moderate      | Excellent      | Low     | Good    |
| BM25 + Dense RRF  | Excellent     | Excellent      | Low     | Better  |
| RRF + Rerank      | Excellent     | Excellent      | Medium  | Best    |

Hybrid retrieval with reranking is the current best practice for production
RAG systems that must handle diverse user queries.

## Latency Considerations

Cross-encoder reranking adds latency proportional to the number of candidates.
Typical times on CPU for ms-marco-MiniLM-L-6-v2:

- 5 candidates: ~50 ms
- 20 candidates: ~180 ms
- 100 candidates: ~900 ms

For interactive applications, keep the candidate set at 20–40 and consider
running the cross-encoder on GPU if available.  For batch pipelines (nightly
indexing, report generation) latency is not a concern.
