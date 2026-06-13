"""
config.py — Central configuration for the Multi-PDF Research Assistant.

All paths, model names, and tunable parameters live here so every other
module can import from a single source of truth.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Root of this project (the directory that contains this file).
PROJECT_ROOT = Path(__file__).parent.resolve()

# Drop your PDF files into this folder before running ingest.py.
DATA_DIR = PROJECT_ROOT / "pdfs"

# ChromaDB will persist its on-disk database here.
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

# ---------------------------------------------------------------------------
# ChromaDB
# ---------------------------------------------------------------------------

# Name of the collection that holds all ingested chunks.
COLLECTION_NAME = "multi_pdf_docs"

# ---------------------------------------------------------------------------
# Embedding model
# ---------------------------------------------------------------------------

# all-MiniLM-L6-v2 is fast, lightweight, and works well for semantic search.
# sentence-transformers will download it automatically on first use (~80 MB).
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

# Maximum number of characters per chunk.
CHUNK_SIZE = 900

# Number of characters that overlap between consecutive chunks on the same
# page.  Overlap helps preserve context at chunk boundaries.
CHUNK_OVERLAP = 150

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

# How many chunks to retrieve per query before sending them to the LLM.
TOP_K = 5

# ---------------------------------------------------------------------------
# Ollama / LLM
# ---------------------------------------------------------------------------

# The Ollama model tag to use for answer generation.
# Pull it once with:  ollama pull llama3.2
OLLAMA_MODEL = "llama3.2"

# Lower temperature → more factual, less creative answers.
OLLAMA_TEMPERATURE = 0.1
