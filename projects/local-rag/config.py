"""Central configuration for the local RAG sample app.

Everything is local: ChromaDB persists to disk, embeddings run via
sentence-transformers, and generation runs through Ollama. Tune these values
to trade off speed vs. quality.
"""
from pathlib import Path

# --- Paths ---------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"               # drop your .txt / .md / .pdf files here
CHROMA_DIR = PROJECT_ROOT / "chroma_db"         # on-disk vector store (git-ignored)
COLLECTION_NAME = "rag_lab_docs"

# --- Embeddings ----------------------------------------------------------
# A small, fast, well-rounded sentence-embedding model (384 dims).
# Downloaded automatically on first run and cached locally.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# --- Chunking ------------------------------------------------------------
CHUNK_SIZE = 800          # characters per chunk
CHUNK_OVERLAP = 120       # characters shared between adjacent chunks

# --- Retrieval -----------------------------------------------------------
TOP_K = 4                 # number of chunks to retrieve per query

# --- Generation (Ollama) -------------------------------------------------
# Pull this model first:  ollama pull llama3.2
OLLAMA_MODEL = "llama3.2"
OLLAMA_TEMPERATURE = 0.1  # low temperature = more grounded, less creative
