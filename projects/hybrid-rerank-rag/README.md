# Hybrid Search + Reranking RAG

A fully local, runnable demonstration of the advanced RAG retrieval stack:

```
BM25 (sparse)  ─┐
                 ├─ RRF fusion ─► cross-encoder rerank ─► Ollama LLM
Dense (vector) ─┘
```

No API keys. Everything runs on your machine.

---

## What this project teaches

| Stage | What happens | Why it helps |
|---|---|---|
| **BM25** | Keyword frequency ranking over all chunks | Catches exact terms, rare words, product codes |
| **Dense** | Cosine similarity in embedding space | Catches semantically equivalent phrasings |
| **RRF fusion** | Merge both ranked lists by reciprocal rank | Chunks good in *both* lists rise to the top |
| **Cross-encoder rerank** | Score each (query, passage) jointly | Resolves negation, co-reference, fine-grained relevance |
| **Grounded generation** | Ollama LLM with citations | Factual, auditable answers |

---

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running (`ollama serve`)
- The `llama3.2` model pulled: `ollama pull llama3.2`

---

## Install

```bash
# Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

The `sentence-transformers` package will download two models on first run:

- `all-MiniLM-L6-v2` (~80 MB) — embedding model for dense retrieval
- `cross-encoder/ms-marco-MiniLM-L-6-v2` (~70 MB) — reranker

Both are cached in `~/.cache/huggingface/` after the first download.

---

## Quick start

### Step 1 — Ingest the sample documents

```bash
python ingest.py
```

This reads `data/*.md`, chunks them, embeds with `all-MiniLM-L6-v2`, stores
in ChromaDB under `chroma_db/`, and writes `chunks.json` for BM25.

To re-ingest from scratch:

```bash
python ingest.py --reset
```

### Step 2 — See the stage-by-stage difference

```bash
python compare.py "What is reciprocal rank fusion?"
python compare.py "BM25 term frequency saturation"
python compare.py "cosine similarity distance metric"
python compare.py --top 3 "how does cross-encoder reranking work?"
```

`compare.py` prints four panels side-by-side:
1. BM25-only top results
2. Dense-only top results
3. Hybrid (RRF) top results
4. Hybrid + cross-encoder rerank top results

Watch how the top-1 chunk shifts between strategies. Keyword-heavy queries
often show BM25 and hybrid agreeing, while semantic queries show dense and
hybrid agreeing.

### Step 3 — Chat with the RAG pipeline

```bash
python rag.py
```

Starts an interactive REPL. Type your question, get a grounded answer with
source citations.

Single-question mode:

```bash
python rag.py --question "How does HNSW indexing work?"
python rag.py --question "What is chunking overlap?" --verbose
```

`--verbose` prints the per-stage debug info before the answer.

---

## Retrieval strategy explainer

### BM25

BM25 (Best Match 25) scores documents by term frequency, normalised by
document length and global IDF.  It needs no training and runs in milliseconds.
Its weakness: it does not understand synonyms or paraphrases.

### Dense (semantic) retrieval

An embedding model maps both the query and every chunk to fixed-length vectors.
Cosine similarity finds the geometrically closest chunks.  It handles semantic
matching beautifully but can miss rare keywords because the embedding averages
over all token meanings.

### Reciprocal Rank Fusion (RRF)

```
RRF(doc) = sum_over_lists  1 / (k + rank_in_list)
```

A simple, parameter-free way to merge ranked lists.  Documents that appear
high in *both* lists get the highest combined score.  No score normalisation
needed — only ranks matter.  `k=60` is the standard default.

### Cross-encoder reranking

The cross-encoder reads the query and passage *together* as one sequence.
Its attention layers model deep interactions (negation, co-reference, exact
phrases) that a bi-encoder cannot see.  It is run only on the ~20 fused
candidates, keeping latency manageable.

---

## When hybrid+rerank helps most

- Queries with specific technical terms (`BM25 IDF formula`) — BM25 anchors
  the candidate set; dense and rerank refine it.
- Natural-language questions (`how does chunking overlap work?`) — dense
  retrieval finds semantically relevant chunks; rerank picks the most precise.
- Mixed queries (`Python implementation of cosine similarity`) — hybrid covers
  both the keyword (Python, cosine) and the semantic intent.

**Trade-off:** cross-encoder reranking adds ~100–300 ms on CPU for 20
candidates.  BM25+dense retrieval alone is sub-10 ms.  For latency-sensitive
applications keep `CANDIDATES_K` small or skip the cross-encoder.

---

## Project layout

```
hybrid-rerank-rag/
├── config.py          # All constants (paths, model names, hyperparams)
├── ingest.py          # Load, chunk, embed, store in Chroma + write chunks.json
├── retrieval.py       # HybridRetriever class: BM25, dense, RRF, rerank
├── rag.py             # Grounded answer generation via Ollama + REPL
├── compare.py         # Stage-by-stage comparison script
├── data/
│   ├── rag-techniques.md
│   ├── vector-search.md
│   └── reranking-and-fusion.md
├── chroma_db/         # Created by ingest.py
├── chunks.json        # Created by ingest.py
└── requirements.txt
```

---

## Troubleshooting

**`FileNotFoundError: chunks.json not found`**
Run `python ingest.py` first.

**`ollama.ResponseError` or connection refused**
Make sure Ollama is running: `ollama serve` in a separate terminal.
Check the model is downloaded: `ollama list`.

**`ValueError: Collection hybrid_rag_docs does not exist`**
Run `python ingest.py` — the collection is created during ingestion.

**Cross-encoder takes a long time on first run**
The model is downloaded from Hugging Face (~70 MB).  Subsequent runs use the
local cache and load in ~2 seconds.

**Out of memory**
Reduce `CANDIDATES_K` in `config.py` (e.g. to 10) to pass fewer candidates
to the cross-encoder.
