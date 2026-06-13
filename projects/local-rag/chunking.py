"""Document loading and chunking utilities.

Kept deliberately dependency-light and readable so you can see exactly how
raw files become the chunks that get embedded. In production you'd likely
reach for LangChain's or LlamaIndex's splitters, but the idea is identical.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

import config


@dataclass
class Chunk:
    """A piece of a source document, ready to embed."""
    text: str
    source: str       # file name the chunk came from
    chunk_index: int  # position of this chunk within its source


def read_file(path: Path) -> str:
    """Read a .txt, .md, or .pdf file into plain text."""
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md", ".markdown"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    raise ValueError(f"Unsupported file type: {path.name}")


def split_text(text: str, size: int, overlap: int) -> list[str]:
    """Split text into overlapping character windows.

    Overlap preserves context that would otherwise be cut at a chunk boundary,
    so an answer spanning the seam between two chunks is still retrievable.
    """
    text = text.strip()
    if not text:
        return []
    if size <= overlap:
        raise ValueError("CHUNK_SIZE must be greater than CHUNK_OVERLAP")

    chunks: list[str] = []
    start = 0
    step = size - overlap
    while start < len(text):
        chunk = text[start : start + size].strip()
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks


def load_chunks(data_dir: Path | None = None) -> list[Chunk]:
    """Load every supported file in ``data_dir`` and return a flat list of chunks."""
    data_dir = data_dir or config.DATA_DIR
    supported = {".txt", ".md", ".markdown", ".pdf"}
    chunks: list[Chunk] = []

    files = sorted(p for p in data_dir.rglob("*") if p.suffix.lower() in supported)
    for path in files:
        text = read_file(path)
        for i, piece in enumerate(split_text(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)):
            chunks.append(Chunk(text=piece, source=path.name, chunk_index=i))
    return chunks
