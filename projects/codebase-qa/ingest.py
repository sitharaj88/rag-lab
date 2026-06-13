# ingest.py — embed source-code chunks and store them in ChromaDB.
#
# Usage:
#   python ingest.py --path /path/to/your/repo
#   python ingest.py --path . --reset          # wipe and re-index
#
# The script:
#   1. Walks the target directory with walker.load_chunks()
#   2. Encodes chunk text with sentence-transformers
#   3. Upserts vectors + metadata into a ChromaDB PersistentClient collection
#
# IDs are deterministic: "<abs_path>::<start>-<end>", so re-running without
# --reset simply overwrites existing chunks (safe for incremental updates).

from __future__ import annotations

import argparse
import sys
from typing import List

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

import config
from walker import Chunk, load_chunks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_client() -> chromadb.PersistentClient:
    """Return a ChromaDB PersistentClient pointed at config.CHROMA_DIR."""
    return chromadb.PersistentClient(
        path=config.CHROMA_DIR,
        settings=Settings(anonymized_telemetry=False),
    )


def get_or_create_collection(
    client: chromadb.PersistentClient,
    reset: bool = False,
) -> chromadb.Collection:
    """Return the collection, optionally wiping it first."""
    if reset:
        try:
            client.delete_collection(config.COLLECTION_NAME)
            print(f"[ingest] Collection '{config.COLLECTION_NAME}' deleted.")
        except Exception:
            pass  # collection didn't exist yet — that's fine

    # cosine distance is ideal for normalised embeddings
    collection = client.get_or_create_collection(
        name=config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def chunk_id(chunk: Chunk) -> str:
    """Deterministic document ID so upsert is idempotent."""
    return f"{chunk.path}::{chunk.start_line}-{chunk.end_line}"


# ---------------------------------------------------------------------------
# Embedding + storage
# ---------------------------------------------------------------------------

def embed_and_store(
    chunks: List[Chunk],
    collection: chromadb.Collection,
    model: SentenceTransformer,
    batch_size: int = 64,
) -> None:
    """Encode chunks in batches and upsert into the collection."""
    total = len(chunks)
    for batch_start in range(0, total, batch_size):
        batch = chunks[batch_start : batch_start + batch_size]

        texts = [c.text for c in batch]
        ids = [chunk_id(c) for c in batch]
        metadatas = [
            {
                "path": c.path,
                "start_line": c.start_line,
                "end_line": c.end_line,
            }
            for c in batch
        ]

        # normalize_embeddings=True → unit vectors → cosine ≡ dot product
        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

        done = min(batch_start + batch_size, total)
        print(f"[ingest] Stored {done}/{total} chunks …", end="\r")

    print()  # newline after progress line


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Index a codebase into ChromaDB for Q&A."
    )
    parser.add_argument(
        "--path",
        default=config.TARGET_DIR,
        help="Root directory of the codebase to index (default: '.').",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete the existing collection before indexing.",
    )
    args = parser.parse_args()

    # 1. Walk + chunk
    print(f"[ingest] Walking '{args.path}' …")
    chunks = load_chunks(args.path)
    if not chunks:
        print("[ingest] No indexable files found. Check INCLUDE_EXTENSIONS / SKIP_DIRS in config.py.")
        sys.exit(1)

    # 2. Load embedding model (downloads once, then cached locally)
    print(f"[ingest] Loading embedding model '{config.EMBEDDING_MODEL}' …")
    model = SentenceTransformer(config.EMBEDDING_MODEL)

    # 3. Connect to ChromaDB
    client = build_client()
    collection = get_or_create_collection(client, reset=args.reset)

    # 4. Embed and store
    print(f"[ingest] Embedding and storing {len(chunks)} chunks …")
    embed_and_store(chunks, collection, model)

    # 5. Summary
    print(
        f"\n[ingest] Done! Collection '{config.COLLECTION_NAME}' now has "
        f"{collection.count()} vectors in '{config.CHROMA_DIR}'."
    )
    print("[ingest] Run `python rag.py` or `streamlit run app.py` to query.")


if __name__ == "__main__":
    main()
