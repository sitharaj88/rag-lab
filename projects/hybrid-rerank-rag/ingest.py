"""
ingest.py — Load, chunk, embed, and store documents for Hybrid RAG.

What this script does:
1. Reads every *.md file from DATA_DIR.
2. Splits each file into overlapping chunks (simple character-level sliding window).
3. Encodes all chunks with the sentence-transformers embedding model.
4. Stores (id, embedding, document text, metadata) in a ChromaDB collection.
5. Writes chunks.json: {chunk_id: {"text": ..., "source": ...}}
   so BM25 can be built over the *exact same* corpus at query time.

Usage:
    python ingest.py           # ingest all docs in data/
    python ingest.py --reset   # delete existing collection first
"""

import argparse
import json
import pathlib
import sys
import time

import chromadb
from sentence_transformers import SentenceTransformer

from config import (
    CHROMA_DIR,
    CHUNKS_JSON,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    DATA_DIR,
    EMBEDDING_MODEL,
)


# ---------------------------------------------------------------------------
# Chunking helpers
# ---------------------------------------------------------------------------

def load_markdown_files(data_dir: pathlib.Path) -> list[dict]:
    """Return a list of {filename, text} dicts for every .md file in data_dir."""
    docs = []
    md_files = sorted(data_dir.glob("*.md"))
    if not md_files:
        print(f"[ingest] WARNING: no .md files found in {data_dir}")
    for path in md_files:
        text = path.read_text(encoding="utf-8")
        docs.append({"filename": path.name, "text": text})
        print(f"[ingest] Loaded {path.name} ({len(text)} chars)")
    return docs


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Split *text* into overlapping character-level windows.

    A simple sliding-window chunker.  Production systems would split on
    paragraph or sentence boundaries; character splits are used here for
    clarity and zero extra dependencies.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        # Advance by (chunk_size - overlap) so the next chunk shares `overlap`
        # characters with this one.
        start += chunk_size - overlap
    return chunks


def build_chunks(docs: list[dict]) -> dict[str, dict]:
    """
    Chunk every document and return a flat mapping:
        chunk_id -> {"text": ..., "source": ...}

    IDs are formatted as  "<filename>_chunk_<index>"  for easy tracing.
    """
    corpus: dict[str, dict] = {}
    for doc in docs:
        raw_chunks = chunk_text(doc["text"], CHUNK_SIZE, CHUNK_OVERLAP)
        for i, chunk_text_val in enumerate(raw_chunks):
            chunk_id = f"{doc['filename']}_chunk_{i:04d}"
            corpus[chunk_id] = {
                "text": chunk_text_val,
                "source": doc["filename"],
            }
    return corpus


# ---------------------------------------------------------------------------
# Embedding + ChromaDB ingestion
# ---------------------------------------------------------------------------

def ingest(reset: bool = False) -> None:
    """Run the full ingestion pipeline."""

    # -- 1. Load documents --------------------------------------------------
    docs = load_markdown_files(DATA_DIR)
    if not docs:
        sys.exit(1)

    # -- 2. Build chunks and persist them for BM25 --------------------------
    print(f"\n[ingest] Chunking (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}) …")
    corpus = build_chunks(docs)
    print(f"[ingest] Total chunks: {len(corpus)}")

    CHUNKS_JSON.write_text(
        json.dumps(corpus, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[ingest] chunks.json written → {CHUNKS_JSON}")

    # -- 3. Load embedding model --------------------------------------------
    print(f"\n[ingest] Loading embedding model: {EMBEDDING_MODEL}")
    t0 = time.time()
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    print(f"[ingest] Model loaded in {time.time() - t0:.1f}s")

    # -- 4. Connect to ChromaDB (PersistentClient) --------------------------
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"[ingest] Deleted existing collection '{COLLECTION_NAME}'")
        except Exception:
            pass  # collection did not exist yet

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # -- 5. Embed and upsert in batches -------------------------------------
    ids = list(corpus.keys())
    texts = [corpus[i]["text"] for i in ids]
    sources = [corpus[i]["source"] for i in ids]

    BATCH_SIZE = 64
    print(f"\n[ingest] Embedding {len(ids)} chunks in batches of {BATCH_SIZE} …")
    t0 = time.time()

    all_embeddings = []
    for batch_start in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[batch_start : batch_start + BATCH_SIZE]
        batch_vecs = embedder.encode(batch_texts, show_progress_bar=False)
        all_embeddings.extend(batch_vecs.tolist())
        print(
            f"[ingest]   embedded {min(batch_start + BATCH_SIZE, len(texts))}"
            f" / {len(texts)}"
        )

    print(f"[ingest] Embedding done in {time.time() - t0:.1f}s")

    # Upsert everything into ChromaDB
    collection.upsert(
        ids=ids,
        embeddings=all_embeddings,
        documents=texts,
        metadatas=[{"source": s} for s in sources],
    )
    print(f"[ingest] ChromaDB collection '{COLLECTION_NAME}' has "
          f"{collection.count()} documents.")
    print("\n[ingest] Done.  Run `python rag.py` to start chatting.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest markdown docs into ChromaDB for hybrid RAG."
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete the existing ChromaDB collection before ingesting.",
    )
    args = parser.parse_args()
    ingest(reset=args.reset)
