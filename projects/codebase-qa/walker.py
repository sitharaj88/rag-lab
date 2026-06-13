# walker.py — directory walker and line-based chunker.
#
# Design decisions:
#   * Chunking by LINES (not characters) keeps citations human-readable:
#     "src/server.py:60-120" is immediately useful to a developer.
#   * Overlap prevents missing logic that straddles a chunk boundary.
#   * We store path, start_line, end_line in each chunk so ingest.py can
#     round-trip that metadata into ChromaDB.

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Generator, List

import config


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """One slice of a source file, ready to embed."""
    path: str       # absolute or relative path to the source file
    start_line: int # 1-based, inclusive
    end_line: int   # 1-based, inclusive
    text: str       # raw text of the chunk (the lines joined)


# ---------------------------------------------------------------------------
# Directory walker
# ---------------------------------------------------------------------------

def iter_source_files(root: str) -> Generator[str, None, None]:
    """Yield absolute paths of indexable files under *root*.

    Skips:
    - Directories whose basename is in config.SKIP_DIRS
    - Files whose extension is not in config.INCLUDE_EXTENSIONS
    - Files larger than config.MAX_FILE_BYTES
    """
    root = os.path.abspath(root)

    for dirpath, dirnames, filenames in os.walk(root):
        # Prune skip-dirs in-place so os.walk doesn't descend into them.
        # We iterate a copy so we can modify dirnames safely.
        dirnames[:] = [
            d for d in dirnames
            if d not in config.SKIP_DIRS and not d.startswith(".")
            # Note: SKIP_DIRS already includes entries starting with "." that
            # we care about; the startswith check catches arbitrary hidden dirs.
        ]

        for filename in filenames:
            # Extension filter
            _, ext = os.path.splitext(filename)
            if ext.lower() not in config.INCLUDE_EXTENSIONS:
                # Also accept extensionless files named exactly ".env.example"
                if filename != ".env.example":
                    continue

            filepath = os.path.join(dirpath, filename)

            # Size filter — skip giant generated / minified files
            try:
                if os.path.getsize(filepath) > config.MAX_FILE_BYTES:
                    continue
            except OSError:
                continue

            yield filepath


# ---------------------------------------------------------------------------
# Line-based chunker
# ---------------------------------------------------------------------------

def chunk_file(filepath: str) -> List[Chunk]:
    """Split a single file into overlapping line-range chunks.

    Returns an empty list if the file cannot be read (binary / encoding error).
    """
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
    except OSError:
        return []

    if not lines:
        return []

    chunks: List[Chunk] = []
    step = config.LINES_PER_CHUNK - config.LINE_OVERLAP  # advance per chunk
    total = len(lines)

    start = 0  # 0-based index into `lines`
    while start < total:
        end = min(start + config.LINES_PER_CHUNK, total)  # exclusive
        chunk_lines = lines[start:end]
        text = "".join(chunk_lines).rstrip()

        if text:  # skip empty chunks (e.g. trailing blank lines)
            chunks.append(Chunk(
                path=filepath,
                start_line=start + 1,      # convert to 1-based
                end_line=end,              # end is already the last line number
                text=text,
            ))

        if end == total:
            break  # reached end of file
        start += step

    return chunks


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------

def load_chunks(root: str) -> List[Chunk]:
    """Walk *root* and return all chunks from all indexable files.

    Prints a progress summary to stdout.
    """
    all_chunks: List[Chunk] = []
    file_count = 0

    for filepath in iter_source_files(root):
        file_chunks = chunk_file(filepath)
        if file_chunks:
            all_chunks.extend(file_chunks)
            file_count += 1

    print(f"[walker] Scanned {file_count} files → {len(all_chunks)} chunks")
    return all_chunks


# ---------------------------------------------------------------------------
# Quick self-test (python walker.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    chunks = load_chunks(root)
    if chunks:
        sample = chunks[0]
        print(f"\nSample chunk: {sample.path}:{sample.start_line}-{sample.end_line}")
        print(sample.text[:300])
