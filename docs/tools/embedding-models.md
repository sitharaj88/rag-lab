# Embedding Models

Embedding models convert text into dense numeric vectors that capture semantic meaning — the quality of these vectors directly determines how well your retriever finds relevant passages. Choosing the right model involves balancing accuracy, speed, dimension size, and whether you can run it locally.

## What you'll learn

- The key dimensions for evaluating an embedding model (dimensions, sequence length, domain, speed)
- The most useful `sentence-transformers` models and when to reach for each
- How the MTEB leaderboard helps you compare models objectively
- The trade-offs between local and API-hosted embeddings
- A working code example you can drop into any RAG pipeline

---

## Key evaluation dimensions

| Dimension | Why it matters |
|-----------|---------------|
| **Vector dimensions** | Higher = more expressive, more VRAM/disk, slower ANN search |
| **Max sequence length** | Tokens beyond the limit are silently truncated — critical for long chunks |
| **Domain** | General models may underperform on legal, medical, or code text |
| **Speed (ms/batch)** | Throughput bottleneck during ingestion and at query time |
| **MTEB score** | Standardized benchmark; higher is better, but domain matters more than rank |

---

## sentence-transformers model guide

`sentence-transformers` is the de-facto library for running embedding models locally.

```bash
pip install sentence-transformers
```

| Model | Dims | Max tokens | Size | Retrieval MTEB | Best for |
|-------|------|-----------|------|---------------|---------|
| `all-MiniLM-L6-v2` | 384 | 256 | 22 MB | Good | Fast prototyping, CPU-only |
| `BAAI/bge-small-en-v1.5` | 384 | 512 | 33 MB | Very good | Balanced speed/quality, local RAG |
| `BAAI/bge-base-en-v1.5` | 768 | 512 | 109 MB | Excellent | Quality-focused local RAG |
| `BAAI/bge-large-en-v1.5` | 1024 | 512 | 335 MB | Top local | High-accuracy production |
| `intfloat/e5-small-v2` | 384 | 512 | 33 MB | Very good | Instruction-tuned, multilingual family |
| `intfloat/e5-large-v2` | 1024 | 512 | 335 MB | Excellent | Best open-source accuracy |
| `thenlper/gte-small` | 384 | 512 | 67 MB | Very good | Strong for passage retrieval |
| `thenlper/gte-large` | 1024 | 512 | 670 MB | Top local | Maximum quality, large GPU |

!!! tip "Recommended starting point"
    `BAAI/bge-small-en-v1.5` gives excellent retrieval quality at 33 MB and runs comfortably on CPU. Upgrade to `bge-base` or `e5-large` when you have a GPU and need higher accuracy.

---

## The MTEB leaderboard

MTEB (Massive Text Embedding Benchmark) evaluates models across 56 tasks — retrieval, clustering, classification, semantic similarity, and more.

- Full leaderboard: [https://huggingface.co/spaces/mteb/leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
- Filter by **Retrieval** task type to find models optimized for RAG
- Filter by model size to stay within your hardware budget

!!! warning "Benchmark vs your data"
    A model ranked #1 on MTEB may not be #1 on your specific domain. Always evaluate on a sample of your own data before committing to a model.

---

## Local vs API embeddings

| | Local (sentence-transformers / Ollama) | API (OpenAI, Cohere) |
|--|----------------------------------------|----------------------|
| **Cost** | Hardware only — zero per-call cost | Per-token pricing |
| **Privacy** | Data never leaves your machine | Data sent to provider |
| **Latency** | CPU: slow; GPU: fast | Network round-trip |
| **Quality** | Competitive at bge-large / e5-large | Top-tier (3072-dim models) |
| **Setup** | One `pip install` | API key + network |
| **Best for** | Local RAG, sensitive data, cost control | Highest accuracy, managed infra |

---

## Usage example

### sentence-transformers

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-small-en-v1.5")

# Embed a batch of passages
passages = [
    "Retrieval-augmented generation combines search with generation.",
    "Vector databases store embeddings for semantic similarity search.",
    "Chunking splits documents into passages that fit context windows.",
]
embeddings = model.encode(passages, normalize_embeddings=True)
print(embeddings.shape)  # (3, 384)

# Embed a query (BGE models recommend a query prefix)
query = "query: How does RAG work?"
query_embedding = model.encode([query], normalize_embeddings=True)
```

!!! note "BGE query prefix"
    BGE models are fine-tuned with an `"query: "` prefix on queries and no prefix on passages. Skipping this slightly reduces retrieval quality.

### Ollama embeddings (no Python deps beyond `ollama`)

```python
import ollama

result = ollama.embed(model="nomic-embed-text", input="How does RAG work?")
vector = result["embeddings"][0]
print(len(vector))  # 768
```

### LangChain integration

```python
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
vector = embeddings.embed_query("What is a vector database?")
```

---

## GPU acceleration

```python
from sentence_transformers import SentenceTransformer

# Automatically uses CUDA if available
model = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cuda")

# Large batch sizes are much faster on GPU
embeddings = model.encode(passages, batch_size=64, show_progress_bar=True)
```

---

## Next steps

- [../foundations/embeddings.md](../foundations/embeddings.md) — How embeddings work mathematically and why they enable semantic search
- [ollama.md](ollama.md) — Running embedding models locally via Ollama with no Python dependencies
