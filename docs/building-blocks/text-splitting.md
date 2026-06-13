# Text Splitting

After loading a document you rarely feed the entire text to an embedder — you split it into overlapping chunks so each piece is small enough to embed accurately yet carries enough context to answer a question on its own.

## What you'll learn

- Why chunk size and overlap matter for retrieval quality
- How to implement fixed-size character splitting with overlap from scratch
- How recursive character splitting on semantic separators works
- How token-based splitting differs from character-based splitting
- How to preserve source metadata on every chunk

---

## Strategy comparison

| Strategy | Split on | Best for |
|----------|----------|----------|
| Fixed-size character | N characters | Simple baseline; predictable size |
| Recursive character | `\n\n`, `\n`, ` `, `""` | Prose, Markdown, mixed content |
| Token-based | Token count (tiktoken / HF tokenizer) | Matching a model's exact context window |

---

## 1. Fixed-size splitting with overlap

This is the simplest approach and a solid baseline.

```python
from __future__ import annotations


def split_fixed(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    metadata: dict | None = None,
) -> list[dict]:
    """Split *text* into fixed-size character chunks with overlap.

    Returns a list of {"text": str, "metadata": dict} dicts.
    """
    metadata = metadata or {}
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end]
        if chunk_text.strip():
            chunks.append({
                "text": chunk_text,
                "metadata": {**metadata, "chunk_index": len(chunks)},
            })
        # Slide forward by (chunk_size - overlap)
        start += chunk_size - chunk_overlap

    return chunks


# --- Example usage ---
with open("article.txt", encoding="utf-8") as f:
    raw = f.read()

chunks = split_fixed(raw, chunk_size=512, chunk_overlap=64,
                     metadata={"source": "article.txt"})

print(f"Total chunks: {len(chunks)}")
print(chunks[0])
# {'text': '...first 512 chars...', 'metadata': {'source': 'article.txt', 'chunk_index': 0}}
```

!!! tip "Choosing chunk_size and chunk_overlap"
    A common starting point is **512 characters / 64 overlap** for short factual documents and **1 024 characters / 128 overlap** for longer narrative text. Always evaluate retrieval quality on a few sample questions before committing.

---

## 2. Recursive character splitting

Recursive splitting tries to break on natural boundaries (paragraphs → sentences → words) before resorting to hard character cuts.

```python
from __future__ import annotations

_DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def split_recursive(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    separators: list[str] | None = None,
    metadata: dict | None = None,
) -> list[dict]:
    """Recursively split on decreasing-priority separators."""
    separators = separators or _DEFAULT_SEPARATORS
    metadata = metadata or {}

    def _split(text: str, seps: list[str]) -> list[str]:
        if not seps or len(text) <= chunk_size:
            return [text] if text.strip() else []
        sep, rest = seps[0], seps[1:]
        parts = text.split(sep)
        results: list[str] = []
        current = ""
        for part in parts:
            candidate = (current + sep + part).lstrip(sep) if current else part
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    results.append(current)
                current = part if len(part) <= chunk_size else ""
                if len(part) > chunk_size:
                    results.extend(_split(part, rest))
        if current:
            results.append(current)
        return results

    raw_chunks = _split(text, separators)

    # Apply overlap by prepending the tail of the previous chunk
    chunks = []
    for i, chunk_text in enumerate(raw_chunks):
        if i > 0 and chunk_overlap:
            prefix = raw_chunks[i - 1][-chunk_overlap:]
            chunk_text = prefix + chunk_text
        if chunk_text.strip():
            chunks.append({
                "text": chunk_text[:chunk_size],
                "metadata": {**metadata, "chunk_index": len(chunks)},
            })
    return chunks
```

---

## 3. Using LangChain's RecursiveCharacterTextSplitter

If you already have LangChain in your project, you can reach for its battle-tested implementation instead of rolling your own.

```bash
pip install langchain-text-splitters
```

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=64,
    separators=["\n\n", "\n", ". ", " ", ""],
)

# Works on a plain string
chunks = splitter.split_text(raw)

# Or on LangChain Document objects (preserves metadata automatically)
from langchain_core.documents import Document

doc = Document(page_content=raw, metadata={"source": "article.txt"})
split_docs = splitter.split_documents([doc])

for i, d in enumerate(split_docs[:3]):
    print(f"[{i}] len={len(d.page_content)}  meta={d.metadata}")
```

!!! note "LangChain vs. hand-rolled"
    The hand-rolled versions above give you full control and zero extra dependencies. LangChain's splitter is more convenient when you're already inside the LangChain ecosystem.

---

## 4. Token-based splitting

Character counts don't map cleanly to model token limits. Use a tokenizer when you need to guarantee a chunk fits inside a specific context window.

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def split_by_tokens(
    text: str,
    max_tokens: int = 128,
    overlap_tokens: int = 16,
    metadata: dict | None = None,
) -> list[dict]:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    start = 0
    while start < len(token_ids):
        end = start + max_tokens
        chunk_ids = token_ids[start:end]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        if chunk_text.strip():
            chunks.append({
                "text": chunk_text,
                "metadata": {**(metadata or {}), "chunk_index": len(chunks)},
            })
        start += max_tokens - overlap_tokens
    return chunks
```

---

## Next steps

- [Foundations — Chunking](../foundations/chunking.md) — the theory behind why chunk boundaries affect retrieval quality
- [Indexing Pipeline](indexing-pipeline.md) — plug your splitter into the full offline ingest workflow
