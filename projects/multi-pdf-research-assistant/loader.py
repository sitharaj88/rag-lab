"""
loader.py — PDF loading and text chunking utilities.

Public API
----------
read_pdf_pages(pdf_path)   -> generator of (page_number, text) tuples
chunk_page(text, source, page, chunk_size, chunk_overlap)
                           -> list of Chunk dataclass instances
load_chunks(data_dir)      -> generator of all Chunk objects from every PDF
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pypdf

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Chunk:
    """A single text window extracted from one page of one PDF.

    Attributes
    ----------
    text:        The raw chunk text.
    source:      Filename of the originating PDF (e.g. "report.pdf").
    page:        1-based page number within that PDF.
    chunk_index: 0-based position of this chunk among all chunks on the page.
    """

    text: str
    source: str
    page: int
    chunk_index: int


# ---------------------------------------------------------------------------
# Low-level PDF reader
# ---------------------------------------------------------------------------


def read_pdf_pages(pdf_path: Path):
    """Yield (page_number, text) pairs for every page in *pdf_path*.

    Parameters
    ----------
    pdf_path:
        Absolute or relative path to a PDF file.

    Yields
    ------
    tuple[int, str]
        A 1-based page number and the extracted text for that page.
        Pages whose text is empty or whitespace-only are skipped.
    """
    with open(pdf_path, "rb") as fh:
        reader = pypdf.PdfReader(fh)
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            text = text.strip()
            if text:
                yield page_num, text
            else:
                logger.debug(
                    "Skipping blank page %d in %s", page_num, pdf_path.name
                )


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------


def chunk_page(
    text: str,
    source: str,
    page: int,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    """Split *text* into overlapping character windows.

    Parameters
    ----------
    text:
        The full text of a single PDF page.
    source:
        Filename of the source PDF (used in metadata).
    page:
        The 1-based page number (used in metadata).
    chunk_size:
        Maximum number of characters per chunk.
    chunk_overlap:
        Number of characters that consecutive chunks share at boundaries.

    Returns
    -------
    list[Chunk]
        Ordered list of Chunk objects for this page.  Returns an empty list
        when *text* is empty.
    """
    if not text:
        return []

    chunks: list[Chunk] = []
    start = 0
    chunk_index = 0
    step = chunk_size - chunk_overlap  # how far to advance the window each time

    # Guard against degenerate configuration (overlap >= size).
    if step <= 0:
        step = max(1, chunk_size // 2)

    while start < len(text):
        end = start + chunk_size
        window = text[start:end].strip()
        if window:
            chunks.append(
                Chunk(
                    text=window,
                    source=source,
                    page=page,
                    chunk_index=chunk_index,
                )
            )
            chunk_index += 1
        start += step

    return chunks


# ---------------------------------------------------------------------------
# Top-level loader
# ---------------------------------------------------------------------------


def load_chunks(data_dir: Path, chunk_size: int = 900, chunk_overlap: int = 150):
    """Yield all Chunk objects produced by loading every PDF in *data_dir*.

    Parameters
    ----------
    data_dir:
        Directory that contains *.pdf files.
    chunk_size:
        Forwarded to chunk_page.
    chunk_overlap:
        Forwarded to chunk_page.

    Yields
    ------
    Chunk
        One chunk at a time, in the order: file → page → chunk_index.
    """
    pdf_files = sorted(data_dir.glob("*.pdf"))
    if not pdf_files:
        logger.warning("No PDF files found in %s", data_dir)
        return

    for pdf_path in pdf_files:
        logger.info("Loading %s", pdf_path.name)
        try:
            for page_num, page_text in read_pdf_pages(pdf_path):
                for chunk in chunk_page(
                    text=page_text,
                    source=pdf_path.name,
                    page=page_num,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                ):
                    yield chunk
        except Exception as exc:
            logger.error("Failed to load %s: %s", pdf_path.name, exc)
