# Choosing Embedding and Reranking Models

Picking the right embedding or reranking model shapes retrieval quality more than almost any other single decision in a RAG pipeline — yet the landscape shifts every few months.

## What you'll learn

- How to read the live MTEB leaderboard to make your own informed choice
- The dimensions that actually matter when comparing models
- A dated snapshot of strong open-weight options (October 2025)
- How rerankers fit into the picture
- A practical "start here, upgrade when" recommendation tied to this site's default model

---

## The MTEB leaderboard: your primary source

The **[MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard)** (Massive Text Embedding Benchmark) is the community-standard ranking for text embedding models. It covers dozens of tasks — retrieval, classification, clustering, semantic textual similarity, reranking, and more — across many languages.

!!! warning "Rankings move monthly"
    A model that tops the chart today may be overtaken next month. **Do not memorise a list of "best" models.** Instead, learn to filter the leaderboard for your specific situation.

### How to read the leaderboard

1. **Filter by task** — For RAG you almost always want the **Retrieval** task category. Classification-optimised models may rank poorly on retrieval.
2. **Filter by language** — If your documents are not English, filter for your target language or use a multilingual task category.
3. **Filter by model size** — The leaderboard shows parameter counts. Balance quality against your inference budget.
4. **Check the license column** — Some top-ranked models carry non-commercial licenses. Deploying them in a commercial product without a licence agreement can create legal risk (see [Licensing](#licensing-flag) below).
5. **Check max sequence length** — Models trained with short context windows truncate long chunks silently, degrading retrieval.

---

## Dimensions that matter

| Dimension | Why it matters | What to look for |
|---|---|---|
| **Task fit** | Retrieval ≠ classification ≠ clustering | Filter MTEB by the Retrieval task |
| **Language** | Multilingual models serve more corpora but may lag English-only on English benchmarks | Match your document language; multilingual if mixed |
| **Model size / speed** | Larger = better quality but higher latency and VRAM | Pick the smallest model that meets your quality bar |
| **Max sequence length** | Chunks longer than the model's context window are truncated | Check `max_seq_length` in the model card; 512 is common, some go to 8192+ |
| **License** | Determines whether you can use the model commercially | Apache-2.0 / MIT = permissive; CC BY-NC = non-commercial only; custom licences vary |
| **Embedding dimensions** | Higher dimensions can encode more information but cost more storage and compute | Typical range: 384–4096; match to your vector store configuration |
| **Sparse / dense / multi-vector** | Some models output only dense vectors; others support sparse or multi-vector (ColBERT-style) | Relevant if you plan hybrid search |

---

## Licensing flag

!!! danger "Non-commercial licences require care"
    Several high-performing models on MTEB carry **non-commercial licences** (e.g. CC BY-NC, NSCLv1). Using these in a product or service that generates revenue — even indirectly — may violate the licence terms.

    **Always read the model card's licence section before adopting a model for anything beyond personal experimentation.** When in doubt, prefer Apache-2.0 or MIT licensed models.

---

## October 2025 snapshot — strong open-weight options

!!! info "Dated snapshot — verify before use"
    The table below reflects the state of the MTEB leaderboard around **October 2025**. Rankings and available models change frequently. Check the [live leaderboard](https://huggingface.co/spaces/mteb/leaderboard) for current standings before making a production decision.

### Embedding models

| Model | Licence | Params | Max seq len | Notable strengths | Notes |
|---|---|---|---|---|---|
| **Qwen3-Embedding-8B** | Apache-2.0 | 8B | 32 768 | Top multilingual retrieval, strong English | Large; needs GPU for low-latency serving |
| **BAAI/bge-m3** | MIT | 570M | 8 192 | Dense + sparse + multi-vector (ColBERT), multilingual | Versatile; good hybrid-search support |
| **infgrad/stella_en_1.5B_v5** | Apache-2.0 | 1.5B | 512 | Strong English retrieval for its size | English-focused |
| **google/embeddinggemma-300m** | Apache-2.0 | 300M | 2 048 | Lightweight, edge-friendly | Good quality-per-MB ratio |
| **NVIDIA/llama-embed-nemotron-8b** | NSCLv1 | 8B | 32 768 | Strong multilingual | ⚠ **Non-commercial licence (NSCLv1)** — not suitable for commercial use without an agreement |
| **sentence-transformers/all-MiniLM-L6-v2** | Apache-2.0 | 22M | 256 | Very fast, extremely small, well-tested | This site's default; quality ceiling is lower than larger models |

### Reranking models

| Model | Type | Licence | Notes |
|---|---|---|---|
| **BAAI/bge-reranker-v2-m3** | Cross-encoder | MIT | Multilingual, strong quality |
| **mixedbread-ai/mxbai-rerank-large-v1** | Cross-encoder | Apache-2.0 | English-focused, competitive on MTEB rerank tasks |
| **BAAI/bge-reranker-base** | Cross-encoder | MIT | Lighter option when latency matters |

---

## Rerankers: what they are and when to add one

A **reranker** (cross-encoder) takes a query and a candidate document as a pair and outputs a relevance score. This is more accurate than embedding cosine similarity because the model sees both texts together — but it is also much slower.

**Typical pipeline with reranking:**

```
User query
    │
    ▼
Embedding retrieval  ─►  top-k candidates (e.g. k=50)
    │
    ▼
Reranker             ─►  re-scored and sorted
    │
    ▼
Top-n passed to LLM  (e.g. n=5)
```

```python
# Example: reranking with sentence-transformers cross-encoder
from sentence_transformers import CrossEncoder

model = CrossEncoder("BAAI/bge-reranker-v2-m3")

query = "What is retrieval-augmented generation?"
candidates = [
    "RAG combines retrieval with generation.",
    "Python is a programming language.",
    "Embeddings represent text as vectors.",
]

scores = model.predict([[query, doc] for doc in candidates])
ranked = sorted(zip(scores, candidates), reverse=True)

for score, doc in ranked:
    print(f"{score:.4f}  {doc}")
```

!!! tip "When to add a reranker"
    Add a reranker when your initial retrieval returns noisy results and passing extra tokens to the LLM is not enough to compensate. Reranking adds latency, so only retrieve a large first-stage pool (k=20–100) when you need it. See [Reranking](../advanced/reranking.md) for a full treatment.

---

## Recommendation: good default and when to upgrade

### Default: `all-MiniLM-L6-v2`

All tutorials on this site use `sentence-transformers/all-MiniLM-L6-v2` as the embedding model. It is:

- 22 M parameters — runs on CPU in milliseconds
- Apache-2.0 licensed — safe for any use
- Widely documented with many community examples
- Good enough for English RAG at learning and prototype scale

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(["Hello world", "RAG pipelines"])
```

### When to upgrade

| Situation | Suggested upgrade path |
|---|---|
| Non-English or multilingual documents | `BAAI/bge-m3` (MIT) or `Qwen3-Embedding-8B` (Apache-2.0) |
| Long documents (chunks > 256 tokens) | Any model with `max_seq_length` ≥ 512; `bge-m3` supports up to 8 192 |
| Higher English retrieval quality needed | `stella_en_1.5B_v5` or check current MTEB Retrieval (English) top-5 |
| Hybrid search (dense + sparse) | `bge-m3` natively supports this |
| Noisy retrieval results | Add `bge-reranker-v2-m3` or `mxbai-rerank-large-v1` as a second stage |
| Production embedding throughput | Serve any model via [HF TEI](local-serving.md) on GPU |

!!! tip "The principle"
    Start with `all-MiniLM-L6-v2`. Measure retrieval quality on your actual data (e.g. using [evaluation](../advanced/evaluation.md) tooling). Only upgrade when measurement shows the model is the bottleneck — not before.

---

## Further reading on embeddings

- [Embeddings foundations](../foundations/embeddings.md) — conceptual background on how embeddings work
- [Embedding models SDK page](embedding-models.md) — code examples for loading and using embedding models
- [Reranking](../advanced/reranking.md) — full pipeline patterns for adding a reranker
- [Live MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard) — always check here before choosing a model

---

## Next steps

- Explore [local serving engines](local-serving.md) to decide how to host your chosen model.
- Read [Reranking](../advanced/reranking.md) to understand when a cross-encoder pays off.
- See [Hybrid search](../advanced/hybrid-search.md) for pipelines that combine dense and sparse retrieval.
