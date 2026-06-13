"""Ingest documents into the local ChromaDB vector store.

Usage:
    python ingest.py            # ingest everything in ./data
    python ingest.py --reset    # wipe the collection first, then ingest

This is the "indexing" half of RAG: load -> chunk -> embed -> store.
"""
from __future__ import annotations

import argparse

import chromadb
from sentence_transformers import SentenceTransformer

import config
from chunking import load_chunks


def get_collection(reset: bool = False) -> chromadb.Collection:
    """Open (or create) the persistent Chroma collection."""
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    if reset:
        try:
            client.delete_collection(config.COLLECTION_NAME)
        except Exception:
            pass  # collection didn't exist yet — fine
    # cosine distance matches normalized sentence-transformer embeddings well
    return client.get_or_create_collection(
        name=config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def main(reset: bool) -> None:
    print(f"Loading documents from {config.DATA_DIR} ...")
    chunks = load_chunks()
    if not chunks:
        print("No documents found. Add .txt/.md/.pdf files to the data/ folder and retry.")
        return
    print(f"Loaded {len(chunks)} chunks from disk.")

    print(f"Loading embedding model: {config.EMBEDDING_MODEL} ...")
    embedder = SentenceTransformer(config.EMBEDDING_MODEL)

    print("Embedding chunks (first run downloads the model) ...")
    embeddings = embedder.encode(
        [c.text for c in chunks],
        show_progress_bar=True,
        normalize_embeddings=True,
    ).tolist()

    collection = get_collection(reset=reset)
    collection.add(
        ids=[f"{c.source}::{c.chunk_index}" for c in chunks],
        documents=[c.text for c in chunks],
        embeddings=embeddings,
        metadatas=[{"source": c.source, "chunk_index": c.chunk_index} for c in chunks],
    )
    print(f"Done. Collection '{config.COLLECTION_NAME}' now holds {collection.count()} chunks.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents into ChromaDB.")
    parser.add_argument("--reset", action="store_true", help="wipe the collection before ingesting")
    args = parser.parse_args()
    main(reset=args.reset)
