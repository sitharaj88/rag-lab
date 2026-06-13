# RAG with Ollama and ChromaDB

Upgrade the in-memory pipeline from Tutorial 01 to a persistent ChromaDB store. Documents are embedded once during an ingest step and reused across sessions — the foundation of any production RAG system.

## What you'll learn

- Why a persistent vector store beats an in-memory list for real workloads
- How to chunk text with a sliding overlap window before indexing
- How to ingest documents into ChromaDB with sentence-transformers embeddings
- How to retrieve top-k results and surface citations alongside answers
- How to separate ingest and query into reusable scripts

## Prerequisites

- [Environment Setup](../getting-started/environment-setup.md)
- [Running a Local LLM with Ollama](../getting-started/local-llm-ollama.md)
- Completed [Tutorial 01 — Minimal RAG](01-minimal-rag.md) or familiar with the concept
- `pip install sentence-transformers chromadb ollama`

## Step 1 — Why ChromaDB?

| | In-memory list (Tutorial 01) | ChromaDB |
|---|---|---|
| Persistence | Lost on restart | Saved to disk |
| Scalability | Rebuilds all embeddings at startup | Embed once, query forever |
| Metadata | Not supported | Arbitrary key-value pairs per document |
| Search | Full matrix scan | Approximate NN (HNSW index) |
| Citations | Manual | Built-in via metadata |

For a corpus of dozens of documents the in-memory approach is fine. For hundreds or thousands of pages it becomes impractical. ChromaDB stores vectors in a local SQLite-backed HNSW index and survives restarts.

## Step 2 — Project layout

Create a directory for this tutorial:

```bash
mkdir rag_chroma && cd rag_chroma
```

You will create two files:

```
rag_chroma/
├── ingest.py   # chunk, embed, and store documents
└── query.py    # retrieve, prompt, and answer
```

## Step 3 — Write the ingest script

```python
# ingest.py
# Chunk a corpus of text, embed each chunk, and store in ChromaDB.

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import uuid

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CHROMA_PATH = "./chroma_db"          # local directory for the vector store
COLLECTION_NAME = "rag_demo"
EMBED_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 200                     # approximate characters per chunk
CHUNK_OVERLAP = 40                   # characters of overlap between chunks

# ---------------------------------------------------------------------------
# Sample corpus — replace with your own text or file loading logic.
# ---------------------------------------------------------------------------
DOCUMENTS = [
    {
        "source": "rag_overview.txt",
        "text": (
            "Retrieval-Augmented Generation (RAG) is a technique that combines "
            "information retrieval with text generation. Instead of relying solely "
            "on the parametric knowledge baked into a language model during training, "
            "RAG first retrieves relevant passages from an external corpus and injects "
            "them into the prompt. This grounding reduces hallucinations and allows "
            "the model to cite specific sources. RAG was introduced by Lewis et al. "
            "in 2020 and has since become one of the dominant architectures for "
            "question-answering over private or frequently updated knowledge bases."
        ),
    },
    {
        "source": "chromadb_guide.txt",
        "text": (
            "ChromaDB is an open-source embedding database designed for AI applications. "
            "It stores document embeddings alongside metadata and exposes a simple Python "
            "API for collection management, upsert, and query operations. Under the hood "
            "ChromaDB uses an HNSW (Hierarchical Navigable Small World) index for "
            "approximate nearest-neighbour search, which scales to millions of vectors "
            "while remaining fast on commodity hardware. Collections are persisted to "
            "a local SQLite file, so data survives process restarts without any server "
            "infrastructure."
        ),
    },
    {
        "source": "ollama_guide.txt",
        "text": (
            "Ollama is a tool for running large language models on your own hardware. "
            "It provides a unified REST API and a Python client library. Models are "
            "downloaded once and cached locally. Ollama supports quantised GGUF models, "
            "which fit on consumer GPUs or even CPU-only machines. The llama3.2 model "
            "family offers strong reasoning capability in a compact footprint. Because "
            "inference runs locally, no data leaves the machine — important for "
            "privacy-sensitive applications such as internal document Q&A."
        ),
    },
    {
        "source": "embeddings_primer.txt",
        "text": (
            "Embeddings are dense numerical representations of text. A sentence "
            "transformer model encodes a sentence into a fixed-length vector such that "
            "semantically similar sentences have vectors that are close together in "
            "the embedding space. The all-MiniLM-L6-v2 model produces 384-dimensional "
            "vectors and runs efficiently on CPU. Normalising vectors to unit length "
            "before storage means that cosine similarity reduces to a dot product, "
            "which is faster to compute. Good embeddings are the foundation of "
            "accurate retrieval: if similar meaning is not captured by the vectors, "
            "no retrieval algorithm can compensate."
        ),
    },
]


# ---------------------------------------------------------------------------
# Chunking helper
# ---------------------------------------------------------------------------
def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Split text into overlapping character-level chunks.

    A word-aware split avoids cutting mid-word: each chunk boundary is
    snapped to the nearest space after the target position.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end < len(text):
            # Snap to nearest space so we don't cut mid-word.
            snap = text.rfind(" ", start, end)
            if snap != -1:
                end = snap
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
        if start >= len(text):
            break
    return chunks


# ---------------------------------------------------------------------------
# Main ingest routine
# ---------------------------------------------------------------------------
def ingest():
    print(f"Connecting to ChromaDB at {CHROMA_PATH!r} …")
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # get_or_create_collection is idempotent — safe to re-run.
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},   # use cosine distance
    )

    embedder = SentenceTransformer(EMBED_MODEL)
    print(f"Loaded embedding model: {EMBED_MODEL}")

    all_chunks: list[str] = []
    all_ids: list[str] = []
    all_metadatas: list[dict] = []

    for doc in DOCUMENTS:
        source = doc["source"]
        chunks = chunk_text(doc["text"], CHUNK_SIZE, CHUNK_OVERLAP)
        print(f"  {source}: {len(chunks)} chunks")
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_ids.append(str(uuid.uuid4()))
            all_metadatas.append({"source": source, "chunk_index": i})

    print(f"\nEmbedding {len(all_chunks)} chunks …")
    embeddings = embedder.encode(all_chunks, normalize_embeddings=True).tolist()

    print("Upserting into ChromaDB …")
    collection.upsert(
        ids=all_ids,
        documents=all_chunks,
        embeddings=embeddings,
        metadatas=all_metadatas,
    )

    print(f"\nDone. Collection '{COLLECTION_NAME}' now has "
          f"{collection.count()} documents.")


if __name__ == "__main__":
    ingest()
```

Run the ingest step:

```bash
python ingest.py
```

You should see output similar to:

```
Connecting to ChromaDB at './chroma_db' …
Loaded embedding model: all-MiniLM-L6-v2
  rag_overview.txt: 5 chunks
  chromadb_guide.txt: 5 chunks
  ollama_guide.txt: 5 chunks
  embeddings_primer.txt: 5 chunks

Embedding 20 chunks …
Upserting into ChromaDB …

Done. Collection 'rag_demo' now has 20 documents.
```

A `chroma_db/` directory is created. You only need to run `ingest.py` again when your corpus changes.

## Step 4 — Write the query script

```python
# query.py
# Retrieve relevant chunks from ChromaDB and answer questions with Ollama.

import chromadb
from sentence_transformers import SentenceTransformer
import ollama

# ---------------------------------------------------------------------------
# Configuration — must match ingest.py
# ---------------------------------------------------------------------------
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "rag_demo"
EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K = 3
LLM_MODEL = "llama3.2"


# ---------------------------------------------------------------------------
# Initialise shared resources once.
# ---------------------------------------------------------------------------
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_collection(COLLECTION_NAME)
embedder = SentenceTransformer(EMBED_MODEL)


# ---------------------------------------------------------------------------
# Retrieve
# ---------------------------------------------------------------------------
def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    """
    Embed the query and return the top_k closest chunks with metadata.

    Returns a list of dicts with keys: 'text', 'source', 'chunk_index',
    'distance'.
    """
    query_embedding = embedder.encode([query], normalize_embeddings=True).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append({
            "text": doc,
            "source": meta.get("source", "unknown"),
            "chunk_index": meta.get("chunk_index", 0),
            "distance": dist,
        })
    return hits


# ---------------------------------------------------------------------------
# Generate
# ---------------------------------------------------------------------------
def answer(query: str) -> tuple[str, list[dict]]:
    """
    Retrieve context, call Ollama, and return (answer_text, hits).

    The caller can display the hits as citations.
    """
    hits = retrieve(query)

    context_parts = []
    for i, hit in enumerate(hits, start=1):
        context_parts.append(
            f"[{i}] (source: {hit['source']}, chunk {hit['chunk_index']})\n"
            f"{hit['text']}"
        )
    context = "\n\n".join(context_parts)

    prompt = (
        "You are a helpful assistant. Answer the question using ONLY the "
        "numbered context passages below. When you use information from a "
        "passage, cite it as [1], [2], etc.\n\n"
        f"{context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"], hits


# ---------------------------------------------------------------------------
# Interactive loop
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"ChromaDB RAG ready ({collection.count()} chunks indexed).")
    print("Type a question or 'quit' to exit.\n")
    while True:
        query = input("You: ").strip()
        if not query or query.lower() in {"quit", "exit", "q"}:
            break
        response_text, hits = answer(query)
        print(f"\nAssistant: {response_text}")
        print("\nSources used:")
        for i, hit in enumerate(hits, start=1):
            print(f"  [{i}] {hit['source']} (chunk {hit['chunk_index']}, "
                  f"distance={hit['distance']:.4f})")
        print()
```

Run the query script:

```bash
python query.py
```

Sample session:

```
ChromaDB RAG ready (20 chunks indexed).
Type a question or 'quit' to exit.

You: What is HNSW and why does ChromaDB use it?
```

## Step 5 — Understand the key design decisions

### Chunking with overlap

```python
start = end - overlap
```

Without overlap, a sentence that spans a chunk boundary gets split in two. Neither half carries the full meaning. The overlap window ensures that important phrases near boundaries appear in at least one complete chunk.

### `hnsw:space: cosine`

```python
collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
)
```

Setting the distance metric at collection creation time to `cosine` matches the normalised embeddings produced by sentence-transformers. If you forget this, ChromaDB defaults to L2 (Euclidean) distance and rankings may be slightly different.

### `upsert` instead of `add`

`upsert` inserts new documents and overwrites existing ones that share the same `id`. This makes `ingest.py` safe to re-run: duplicate chunks are replaced rather than duplicated. Using `uuid.uuid4()` for IDs means every run generates fresh IDs — fine for a demo but in production you would derive stable IDs from a hash of the content so re-runs are truly idempotent.

### Citations via metadata

```python
all_metadatas.append({"source": source, "chunk_index": i})
```

Metadata travels with every chunk through the index. At query time it comes back in `results["metadatas"]` so the UI can show exactly which file and which chunk each answer came from.

## Step 6 — Experiment

!!! tip "Things to try"
    - Change `TOP_K` from 3 to 5 and observe whether answer quality improves or gets noisier.
    - Reduce `CHUNK_SIZE` to 100 characters and re-ingest. Do retrieved chunks feel more or less focused?
    - Add a new document to `DOCUMENTS`, re-run `ingest.py`, and verify the collection count increases.
    - Ask a question that is not covered by any document and note how the model responds.

!!! warning "Re-ingesting after changes"
    If you change `CHUNK_SIZE` or `CHUNK_OVERLAP`, delete the `chroma_db/` directory before re-running `ingest.py`. Otherwise old chunks (with different sizes) will co-exist with new ones and retrieval quality will be inconsistent.

## Next steps

- [Tutorial 03 — Streamlit Chat App](03-streamlit-chat-app.md): wrap this pipeline in a browser-based chat interface.
- [Indexing Pipeline](../building-blocks/indexing-pipeline.md): learn about production-grade ingest patterns including metadata filtering and incremental updates.
