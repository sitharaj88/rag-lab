# RAG Retrieval Techniques

Retrieval-Augmented Generation (RAG) grounds a language model's answers in a
corpus of documents retrieved at query time.  The quality of the final answer
depends heavily on *which* chunks end up in the context window, making
retrieval the most important engineering lever in a RAG system.

## Naive RAG: Embedding-Only Retrieval

The simplest RAG pipeline encodes every document chunk with a dense embedding
model and stores the vectors in a vector database.  At query time the question
is embedded with the same model, and the nearest neighbours are returned using
cosine similarity or dot-product search.

This approach works well when the question and the relevant passage share
semantic meaning even without overlapping keywords.  For example, the query
"What is the capital of France?" matches "Paris is the seat of the French
government" even though the words capital and seat differ.

Weaknesses of embedding-only retrieval:
- Exact keyword matches (product codes, proper nouns, rare terms) may be
  diluted by the embedding.
- The model was trained on general text, so domain-specific jargon may not
  map to meaningful vector distances.
- Approximate nearest-neighbour indexes (HNSW, IVF) can miss relevant chunks
  that are geometrically close but fall outside the beam.

## BM25: Sparse Keyword Retrieval

BM25 (Best Match 25) is a probabilistic term-frequency ranking function that
scores documents by how often query terms appear, normalised by document
length.  It has no semantic understanding but excels at exact keyword matches.

BM25 is the backbone of production search engines like Elasticsearch and
Solr.  It handles:
- Rare or technical terms that the embedding model has never seen together.
- Boolean-style queries: "Python list comprehension performance" returns
  chunks that literally contain all three words.
- Product IDs, error codes, and version strings.

BM25Okapi (the `rank-bm25` library's default) adds per-term IDF weighting and
a smoothed term-frequency saturation function, preventing a single very common
word from dominating the score.

## Hybrid Retrieval

Hybrid retrieval runs both a dense (embedding) retriever and a sparse (BM25)
retriever independently, then *fuses* their ranked lists.  The intuition is
that:

1. BM25 catches exact matches that the dense model misses.
2. The dense model catches semantically equivalent phrasings that BM25 misses.
3. Chunks that rank high in *both* lists are almost certainly relevant.

Reciprocal Rank Fusion (RRF) is the standard fusion algorithm:

```
score(doc) = sum over each list of  1 / (k + rank_in_list)
```

where `k` (typically 60) prevents very high ranks from completely dominating.
Hybrid retrieval consistently outperforms either retriever alone on BEIR and
MTEB benchmarks.

## Reranking with a Cross-Encoder

After fusion you have a candidate set (e.g. 20 chunks).  A *cross-encoder*
reranker reads *both* the query and each candidate as a single sequence and
produces a calibrated relevance score.  Because the model sees the query and
passage jointly, it can model deep interactions (pronoun resolution, negation,
entity co-reference) that a bi-encoder embedding model cannot.

The ms-marco-MiniLM cross-encoder was fine-tuned on the MS MARCO passage
ranking dataset, giving it strong out-of-the-box performance for RAG.

Reranking trade-offs:
- Latency: O(candidates) forward passes vs. a single embedding for dense search.
- Quality: typically +3–8% NDCG@10 over dense-only retrieval.
- Best practice: run cheap BM25+dense retrieval to get top-20 candidates, then
  use the cross-encoder to select the final top-4.  Never rerank the full corpus.

## Grounded Generation

Once the top-k chunks are selected, they are injected into the LLM prompt as
context.  A grounded prompt template instructs the model to:
1. Answer using *only* the provided context.
2. Cite the source document for each factual claim.
3. Say "I don't know" if the context does not contain enough information.

Low temperature (0.0–0.2) reduces hallucination by making the model stick
closely to the retrieved text rather than generating from prior knowledge.
