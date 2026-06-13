"""
config.py — Central configuration for Hybrid Search + Reranking RAG.

All paths and hyperparameters live here so every other module imports from
a single source of truth.  Change values here; nothing else needs editing.
"""

import pathlib

# ---------------------------------------------------------------------------
# Filesystem layout
# ---------------------------------------------------------------------------

# Root of this project (the directory that contains config.py)
PROJECT_DIR = pathlib.Path(__file__).parent.resolve()

# Raw knowledge documents (Markdown files)
DATA_DIR = PROJECT_DIR / "data"

# ChromaDB persistence directory
CHROMA_DIR = PROJECT_DIR / "chroma_db"

# JSON file that stores {chunk_id: {"text": ..., "source": ...}} for BM25
CHUNKS_JSON = PROJECT_DIR / "chunks.json"

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

CHUNK_SIZE = 700        # characters per chunk
CHUNK_OVERLAP = 120     # character overlap between consecutive chunks

# ---------------------------------------------------------------------------
# Embedding model (dense retrieval)
# all-MiniLM-L6-v2 is fast, small (~80 MB), and works great for RAG demos.
# ---------------------------------------------------------------------------

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ChromaDB collection name
COLLECTION_NAME = "hybrid_rag_docs"

# ---------------------------------------------------------------------------
# Retrieval hyperparameters
# ---------------------------------------------------------------------------

# Number of candidates each retriever (BM25 and dense) returns BEFORE fusion.
# More candidates = better recall for fusion; fewer = faster.
CANDIDATES_K = 20

# Reciprocal Rank Fusion constant.  Larger k reduces the influence of rank
# differences near the top.  60 is a well-established default.
RRF_K = 60

# Final number of chunks passed to the LLM after reranking.
TOP_K = 4

# ---------------------------------------------------------------------------
# Reranking model (cross-encoder)
# ms-marco-MiniLM-L-6-v2 is purpose-trained for passage relevance ranking.
# Downloaded automatically on first run (~70 MB).
# ---------------------------------------------------------------------------

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ---------------------------------------------------------------------------
# Ollama / generation
# ---------------------------------------------------------------------------

OLLAMA_MODEL = "llama3.2"
OLLAMA_TEMPERATURE = 0.1   # Low temperature for grounded, factual answers
