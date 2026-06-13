# Data Loaders

Before you can index documents into a vector store, you need to extract clean text and metadata from raw files. This guide covers three essential Python libraries — `pypdf`, `unstructured`, and `pandas` — and shows how their output feeds directly into the indexing pipeline.

## What you'll learn

- Extracting text and page metadata from PDFs with `pypdf`
- Handling complex and mixed-layout documents with `unstructured`
- Loading tabular data from CSV files with `pandas`
- Understanding OCR limitations for scanned PDFs
- How loaded content connects to the chunking and indexing pipeline

## Install

```bash
pip install pypdf unstructured pandas
```

!!! note "Optional extras for unstructured"
    For PDF parsing with layout detection and OCR:
    ```bash
    pip install "unstructured[pdf]"
    ```
    OCR support additionally requires `tesseract` and `poppler` to be installed at the system level.

## pypdf — simple PDF extraction

`pypdf` is the go-to choice for text-layer PDFs (i.e., PDFs where the text is embedded as selectable characters, not rendered as images).

```python
from pypdf import PdfReader
import pathlib

def load_pdf(path: str) -> list[dict]:
    """Return a list of page dicts with 'text', 'page', and 'source' keys."""
    reader = PdfReader(path)
    pages = []

    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append({
            "text": text.strip(),
            "page": page_num,
            "source": pathlib.Path(path).name,
        })

    return pages


# ── Usage ─────────────────────────────────────────────────────────────────────
pages = load_pdf("my_document.pdf")

for p in pages[:3]:
    print(f"[page {p['page']}] {p['text'][:120]}…")
```

Each dict in `pages` is a self-contained unit you can pass to a text splitter or embed directly.

!!! warning "Scanned PDFs — OCR limitation"
    `pypdf` extracts the text layer only. If a PDF was created by scanning a physical document (images of pages), `page.extract_text()` returns an empty string. Use `unstructured` with an OCR backend for those files.

## unstructured — complex and mixed layouts

`unstructured` handles PDFs with tables, headers, multi-column layouts, and mixed content. It returns a list of `Element` objects, each tagged with a type (`Title`, `NarrativeText`, `Table`, `ListItem`, etc.).

```python
from unstructured.partition.pdf import partition_pdf

def load_pdf_structured(path: str) -> list[dict]:
    """Use unstructured to extract elements with type metadata."""
    elements = partition_pdf(filename=path)
    results = []

    for elem in elements:
        results.append({
            "text": str(elem),
            "type": type(elem).__name__,      # e.g. 'NarrativeText', 'Title', 'Table'
            "source": path,
        })

    return results


# ── Usage ─────────────────────────────────────────────────────────────────────
elements = load_pdf_structured("complex_report.pdf")

for e in elements[:5]:
    print(f"[{e['type']}] {e['text'][:100]}")
```

For non-PDF files (Word, HTML, email, Markdown) use the generic `partition` function:

```python
from unstructured.partition.auto import partition

elements = partition(filename="document.docx")
```

!!! tip "Filtering by element type"
    You can keep only narrative content and skip headers/footers:
    ```python
    narrative = [e for e in elements if e["type"] == "NarrativeText"]
    ```

!!! warning "OCR for scanned documents"
    To process scanned PDFs, pass `strategy="ocr_only"` or `strategy="hi_res"` to `partition_pdf`.
    This requires `tesseract` and `poppler` installed on your system, and is significantly slower.
    ```python
    elements = partition_pdf(filename="scanned.pdf", strategy="hi_res")
    ```

## pandas — CSV and tabular data

Tabular data (spreadsheets, CSVs, exports) is often overlooked in RAG pipelines. Converting rows to text strings makes them embeddable.

```python
import pandas as pd

def load_csv(path: str, text_columns: list[str], metadata_columns: list[str] | None = None) -> list[dict]:
    """
    Load a CSV and produce one document dict per row.

    text_columns: columns to concatenate as the searchable text
    metadata_columns: columns to keep as payload metadata
    """
    df = pd.read_csv(path)
    metadata_columns = metadata_columns or []
    results = []

    for _, row in df.iterrows():
        text = " | ".join(str(row[col]) for col in text_columns if col in df.columns)
        metadata = {col: row[col] for col in metadata_columns if col in df.columns}
        metadata["source"] = path
        results.append({"text": text, **metadata})

    return results


# ── Usage ─────────────────────────────────────────────────────────────────────
records = load_csv(
    "products.csv",
    text_columns=["name", "description"],
    metadata_columns=["category", "price"],
)

for r in records[:3]:
    print(r)
```

!!! tip "Handling large CSV files"
    For very large files, use `pd.read_csv(path, chunksize=1000)` and process each chunk separately to avoid loading the entire file into memory.

## Normalising output for the pipeline

Regardless of which loader you use, normalise output to the same structure before passing it to a text splitter or embedder:

```python
from typing import TypedDict

class Document(TypedDict):
    text: str        # the raw extracted content
    source: str      # filename or URL
    page: int        # page number (0 if not applicable)
    extra: dict      # any additional metadata

def to_document(text: str, source: str, page: int = 0, **kwargs) -> Document:
    return Document(text=text, source=source, page=page, extra=kwargs)
```

## Choosing a loader

| Loader | Best for |
|---|---|
| `pypdf` | Standard text-layer PDFs, fast, minimal dependencies |
| `unstructured` | Complex PDFs, tables, mixed layouts, scanned docs (with OCR), non-PDF file types |
| `pandas` | CSV, Excel, tabular exports |

For most RAG pipelines, you will combine all three: `pypdf` for clean PDFs, `unstructured` for edge cases, and `pandas` for any tabular sources.

## Next steps

- Learn how loaded text is split into chunks: [Document Loaders](../building-blocks/document-loaders.md)
- See how chunks flow into a vector store: [Indexing Pipeline](../building-blocks/indexing-pipeline.md)
- Build a complete PDF Q&A app: [Tutorial 04 — PDF Q&A](../tutorials/04-pdf-qa.md)
