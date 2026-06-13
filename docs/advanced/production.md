# Production RAG

Moving a RAG prototype to production means hardening every component in the pipeline — caching, observability, security, freshness, and scale — while keeping latency and cost under control.

## What you'll learn

- Caching strategies to reduce redundant LLM and embedding calls
- Observability: what to log and how to trace a RAG request end-to-end
- Security concerns: prompt injection and PII in documents
- Document-level access control
- Incremental indexing for keeping your vector store fresh
- A production readiness checklist

## Caching

RAG has two expensive operations: embedding queries and calling the LLM. Both are deterministic for identical inputs and are prime candidates for caching.

```python
# caching.py
import hashlib, json
from functools import lru_cache
from pathlib import Path

CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(exist_ok=True)

def cache_key(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()

def cached_embed(text: str, embed_fn) -> list[float]:
    key = cache_key(text)
    path = CACHE_DIR / f"emb_{key}.json"
    if path.exists():
        return json.loads(path.read_text())
    embedding = embed_fn(text)
    path.write_text(json.dumps(embedding))
    return embedding

def cached_llm(prompt: str, llm_fn) -> str:
    key = cache_key(prompt)
    path = CACHE_DIR / f"llm_{key}.txt"
    if path.exists():
        return path.read_text()
    response = llm_fn(prompt)
    path.write_text(response)
    return response
```

!!! tip "Semantic cache"
    A more powerful approach: embed the incoming query, search a small "query cache" vector store for similar past queries (cosine sim > 0.97), and return the cached answer. This handles paraphrased repeats. Libraries like GPTCache implement this pattern.

## Observability and logging

Log every step of the pipeline so you can debug quality regressions:

```python
# observability.py
import time, logging, uuid

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("rag")

def traced_rag(query: str, retrieve_fn, generate_fn) -> dict:
    trace_id = str(uuid.uuid4())[:8]
    t0 = time.perf_counter()

    logger.info(f"[{trace_id}] query={query!r}")

    t1 = time.perf_counter()
    docs = retrieve_fn(query)
    retrieval_ms = (time.perf_counter() - t1) * 1000
    logger.info(f"[{trace_id}] retrieved={len(docs)} docs in {retrieval_ms:.1f}ms")

    t2 = time.perf_counter()
    answer = generate_fn(query, docs)
    gen_ms = (time.perf_counter() - t2) * 1000
    logger.info(f"[{trace_id}] generated in {gen_ms:.1f}ms")

    total_ms = (time.perf_counter() - t0) * 1000
    return {
        "trace_id": trace_id,
        "answer": answer,
        "retrieved_docs": docs,
        "latency_ms": {"retrieval": retrieval_ms, "generation": gen_ms, "total": total_ms},
    }
```

For structured tracing in production, integrate with OpenTelemetry, Langfuse, or Arize Phoenix — all support local deployment.

## Security

### Prompt injection

Users may craft inputs designed to override your system prompt or extract confidential context.

```python
# injection_guard.py
import re

INJECTION_PATTERNS = [
    r"ignore (all )?previous instructions",
    r"disregard (the )?system prompt",
    r"you are now",
    r"pretend (you are|to be)",
    r"forget (everything|all)",
]

def detect_injection(text: str) -> bool:
    text_lower = text.lower()
    return any(re.search(p, text_lower) for p in INJECTION_PATTERNS)

def safe_query(query: str) -> str:
    if detect_injection(query):
        raise ValueError("Potential prompt injection detected.")
    return query
```

!!! warning "PII in documents"
    If your corpus contains personal data, apply a PII detection and redaction step at index time (e.g., using spaCy's NER or Microsoft Presidio) before embedding. Never index raw PII you don't intend to expose to users.

### Document-level access control

Tag every document with metadata at index time, then filter at retrieval time:

```python
# access_control.py
import chromadb

client = chromadb.PersistentClient(path="./chroma_db")
col = client.get_or_create_collection("secure_docs")

# Index with ACL metadata
col.add(
    documents=["Top secret financials Q4.", "Public product overview."],
    ids=["d_private", "d_public"],
    metadatas=[{"access": "finance_team"}, {"access": "public"}],
)

def retrieve_for_user(query: str, user_role: str, top_k: int = 3):
    return col.query(
        query_texts=[query],
        n_results=top_k,
        where={"access": {"$in": [user_role, "public"]}},
    )
```

## Incremental indexing

Avoid full re-indexing on every document update. Track document hashes and only re-embed changed content:

```python
# incremental_index.py
import hashlib, json
from pathlib import Path

HASH_STORE = Path("doc_hashes.json")

def load_hashes() -> dict:
    return json.loads(HASH_STORE.read_text()) if HASH_STORE.exists() else {}

def save_hashes(hashes: dict):
    HASH_STORE.write_text(json.dumps(hashes, indent=2))

def hash_doc(content: str) -> str:
    return hashlib.md5(content.encode()).hexdigest()

def incremental_upsert(docs: dict[str, str], col) -> tuple[int, int]:
    """docs: {doc_id: content}. Returns (added, skipped)."""
    hashes = load_hashes()
    added, skipped = 0, 0
    for doc_id, content in docs.items():
        h = hash_doc(content)
        if hashes.get(doc_id) == h:
            skipped += 1
            continue
        col.upsert(documents=[content], ids=[doc_id])
        hashes[doc_id] = h
        added += 1
    save_hashes(hashes)
    return added, skipped
```

## Production readiness checklist

| Area | Item | Status |
|---|---|---|
| **Caching** | Embedding cache (exact or semantic) | |
| **Caching** | LLM response cache for repeated queries | |
| **Latency** | Retrieval P95 < 200 ms | |
| **Latency** | End-to-end P95 < 2 s | |
| **Observability** | Per-request trace with IDs logged | |
| **Observability** | Retrieval and generation latency tracked | |
| **Observability** | Retrieved doc IDs logged per query | |
| **Security** | Prompt injection detection on input | |
| **Security** | PII redacted at index time | |
| **Security** | Document-level ACL filtering at retrieval | |
| **Freshness** | Incremental / hash-based re-indexing | |
| **Freshness** | Deletion propagated to vector store | |
| **Scale** | Vector store persisted (not in-memory) | |
| **Scale** | Embedding model on GPU or batched | |
| **Evaluation** | Eval suite in CI on every pipeline change | |
| **Guardrails** | Output checked for hallucinations / off-topic | |

!!! note "Vector store scaling"
    For local deployments, ChromaDB's `PersistentClient` handles moderate scale (millions of vectors). For larger workloads, consider Qdrant (self-hosted), Weaviate, or Milvus — all support horizontal scaling.

## Guardrails

Add an output validation layer that checks the answer is grounded in context before returning it to the user:

```python
# guardrail.py
import ollama

def is_grounded(context: str, answer: str) -> bool:
    prompt = (
        f"Context:\n{context}\n\nAnswer:\n{answer}\n\n"
        "Is this answer fully supported by the context above? Reply YES or NO."
    )
    r = ollama.chat(model="llama3.2",
                    messages=[{"role": "user", "content": prompt}])
    return "YES" in r["message"]["content"].upper()
```

## Next steps

- [Evaluation](evaluation.md) — build the eval suite you run in CI
- [Vector stores](../tools/vector-stores.md) — choose the right store for your scale and deployment constraints
