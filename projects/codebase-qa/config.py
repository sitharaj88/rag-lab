# config.py — central configuration for the Codebase Q&A project.
# Tweak these constants to adapt the pipeline to your codebase and hardware.

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Root directory to index when no --path flag is given to ingest.py.
# Default "." means "index this very repo" — useful for demos.
TARGET_DIR: str = "."

# Where ChromaDB stores its on-disk database.
CHROMA_DIR: str = os.path.join(os.path.dirname(__file__), "chroma_db")

# Name of the ChromaDB collection.
COLLECTION_NAME: str = "codebase_qa"

# ---------------------------------------------------------------------------
# Embedding model
# ---------------------------------------------------------------------------

# Small, fast, and CPU-friendly sentence-transformer model.
# normalize_embeddings=True makes cosine similarity equivalent to dot product.
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

# Number of source-code lines per chunk.
# 60 lines gives enough context without blowing the LLM context window.
LINES_PER_CHUNK: int = 60

# Lines of overlap between consecutive chunks so cross-boundary logic is captured.
LINE_OVERLAP: int = 10

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

# Number of chunks to retrieve and feed into the LLM prompt.
TOP_K: int = 6

# ---------------------------------------------------------------------------
# LLM (via Ollama)
# ---------------------------------------------------------------------------

# Ollama model tag.  Run `ollama pull llama3.2` once before using this project.
OLLAMA_MODEL: str = "llama3.2"

# Low temperature keeps answers grounded and deterministic.
OLLAMA_TEMPERATURE: float = 0.1

# ---------------------------------------------------------------------------
# File-walking filters
# ---------------------------------------------------------------------------

# Only index files with these extensions.
INCLUDE_EXTENSIONS: set = {
    ".py", ".js", ".ts", ".tsx", ".jsx",
    ".java", ".go", ".rs", ".rb", ".php",
    ".c", ".cpp", ".h", ".hpp", ".cs",
    ".md", ".txt", ".rst",
    ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg",
    ".sh", ".bash", ".zsh",
    ".sql",
    ".html", ".css",
    ".env.example",  # safe example files — never real .env
}

# Skip these directory names anywhere in the tree.
SKIP_DIRS: set = {
    ".git", ".github",
    "node_modules",
    ".venv", "venv", "env", ".env",
    "__pycache__",
    "dist", "build", "out",
    "site",                   # MkDocs output
    ".chroma", "chroma_db",   # our own vector store
    ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "coverage", ".coverage",
    "target",                 # Rust/Java build output
    ".idea", ".vscode",
}

# Skip files larger than this (bytes).  Avoids embedding minified bundles,
# lock files, or generated code that would flood the index with noise.
MAX_FILE_BYTES: int = 250_000  # 250 KB
