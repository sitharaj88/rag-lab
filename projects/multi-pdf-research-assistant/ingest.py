"""
ingest.py — Build and populate the ChromaDB vector store.

Usage
-----
    python ingest.py           # ingest all PDFs in pdfs/
    python ingest.py --reset   # wipe the collection first, then re-ingest

What it does
------------
1. Loads every PDF in DATA_DIR, splitting each page into overlapping chunks.
2. Embeds each chunk with sentence-transformers (all-MiniLM-L6-v2).
3. Upserts chunks into a ChromaDB collection configured for cosine similarity.
"""

from __future__ import annotations

import argparse
import logging
import sys

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

import config
from loader import Chunk, load_chunks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ChromaDB helpers
# ---------------------------------------------------------------------------


def get_collection(reset: bool = False) -> chromadb.Collection:
    """Return (or recreate) the ChromaDB collection.

    Parameters
    ----------
    reset:
        When True the existing collection is deleted before a fresh one is
        created.  Useful when you want to re-ingest everything from scratch.

    Returns
    -------
    chromadb.Collection
        A collection configured to use cosine distance for nearest-neighbour
        search.
    """
    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(
        path=str(config.CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )

    if reset:
        try:
            client.delete_collection(config.COLLECTION_NAME)
            logger.info("Deleted existing collection '%s'.", config.COLLECTION_NAME)
        except Exception:
            pass  # collection may not exist yet — that is fine

    collection = client.get_or_create_collection(
        name=config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


# ---------------------------------------------------------------------------
# Embedding helper
# ---------------------------------------------------------------------------


def embed_chunks(
    model: SentenceTransformer,
    chunks: list[Chunk],
) -> list[list[float]]:
    """Return a list of embedding vectors for *chunks*.

    Parameters
    ----------
    model:
        A loaded SentenceTransformer instance.
    chunks:
        The Chunk objects whose .text fields will be embedded.

    Returns
    -------
    list[list[float]]
        One float vector per chunk, in the same order as *chunks*.
    """
    texts = [c.text for c in chunks]
    # normalize_embeddings=True makes cosine similarity equivalent to dot product,
    # which is what ChromaDB's "hnsw:space": "cosine" expects.
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    return [v.tolist() for v in vectors]


# ---------------------------------------------------------------------------
# Main ingest routine
# ---------------------------------------------------------------------------


def ingest(reset: bool = False) -> None:
    """Load all PDFs, embed their chunks, and upsert into ChromaDB.

    Parameters
    ----------
    reset:
        Passed through to get_collection(); clears existing data when True.
    """
    # ---- guard: check the PDF directory exists and has content ----
    if not config.DATA_DIR.exists():
        logger.error(
            "PDF directory not found: %s  — create it and drop PDFs inside.",
            config.DATA_DIR,
        )
        sys.exit(1)

    pdf_files = list(config.DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        logger.warning(
            "No PDF files found in %s.  Nothing to ingest.", config.DATA_DIR
        )
        return

    logger.info("Found %d PDF file(s) in %s.", len(pdf_files), config.DATA_DIR)

    # ---- load and chunk ----
    all_chunks: list[Chunk] = list(
        load_chunks(
            config.DATA_DIR,
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
        )
    )
    if not all_chunks:
        logger.warning("No text extracted from the PDFs.  Aborting.")
        return

    logger.info("Extracted %d chunk(s) from all PDFs.", len(all_chunks))

    # ---- load the embedding model ----
    logger.info("Loading embedding model '%s' …", config.EMBEDDING_MODEL)
    model = SentenceTransformer(config.EMBEDDING_MODEL)

    # ---- embed in one pass ----
    logger.info("Embedding %d chunk(s) …", len(all_chunks))
    embeddings = embed_chunks(model, all_chunks)

    # ---- connect to / reset the collection ----
    collection = get_collection(reset=reset)
    logger.info(
        "Using ChromaDB collection '%s' at %s.",
        config.COLLECTION_NAME,
        config.CHROMA_DIR,
    )

    # ---- build IDs and metadata ----
    # ID format:  "report.pdf::p3::2"  (file, page, chunk index — all unique)
    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict] = []

    for chunk in all_chunks:
        chunk_id = f"{chunk.source}::p{chunk.page}::{chunk.chunk_index}"
        ids.append(chunk_id)
        documents.append(chunk.text)
        metadatas.append(
            {
                "source": chunk.source,
                "page": chunk.page,
                "chunk_index": chunk.chunk_index,
            }
        )

    # ---- upsert in batches to stay within ChromaDB limits ----
    BATCH_SIZE = 512
    total_upserted = 0

    for batch_start in range(0, len(ids), BATCH_SIZE):
        batch_end = batch_start + BATCH_SIZE
        collection.upsert(
            ids=ids[batch_start:batch_end],
            embeddings=embeddings[batch_start:batch_end],
            documents=documents[batch_start:batch_end],
            metadatas=metadatas[batch_start:batch_end],
        )
        total_upserted += len(ids[batch_start:batch_end])
        logger.info("Upserted %d / %d chunks …", total_upserted, len(ids))

    logger.info(
        "Done!  %d chunk(s) stored in collection '%s'.",
        total_upserted,
        config.COLLECTION_NAME,
    )
    logger.info(
        "Collection now holds %d total document(s).", collection.count()
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest PDFs into the ChromaDB vector store."
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete the existing collection before ingesting (full re-build).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    ingest(reset=args.reset)
