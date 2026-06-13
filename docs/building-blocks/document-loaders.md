# Document Loaders

Document loaders are the entry point of any RAG pipeline — they read raw files from disk (or the web) and return clean text plus structured metadata that travels with every chunk downstream.

## What you'll learn

- How to load `.txt`, `.md`, `.pdf`, `.html`, and `.docx` files into Python strings
- How to extract clean text and handle encoding edge cases
- How to attach source metadata (file path, page number) to each document
- How to build a reusable `read_file` helper that covers the most common formats
- Which file types are genuinely hard and why

---

## Supported formats at a glance

| Format | Library | Notes |
|--------|---------|-------|
| `.txt` / `.md` | built-in | Fast; watch for encoding |
| `.pdf` | `pypdf` | Good for digital PDFs; scanned = hard |
| `.html` | `beautifulsoup4` | Strip tags; beware boilerplate |
| `.docx` | `python-docx` | Paragraphs + tables; images skipped |

Install the optional dependencies once:

```bash
pip install pypdf beautifulsoup4 python-docx
```

---

## A reusable `read_file` function

The function below returns a `dict` with `text` and `metadata` keys so that every loader speaks the same interface.

```python
from __future__ import annotations
import pathlib


def read_file(path: str | pathlib.Path) -> dict:
    """Load a file and return {"text": str, "metadata": dict}.

    Supported extensions: .txt, .md, .pdf, .html, .htm, .docx
    """
    path = pathlib.Path(path)
    ext = path.suffix.lower()
    metadata = {"source": str(path)}

    if ext in {".txt", ".md"}:
        text = path.read_text(encoding="utf-8", errors="replace")

    elif ext == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        pages = []
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            pages.append(page_text)
        text = "\n".join(pages)
        metadata["page_count"] = len(reader.pages)

    elif ext in {".html", ".htm"}:
        from bs4 import BeautifulSoup

        raw = path.read_text(encoding="utf-8", errors="replace")
        soup = BeautifulSoup(raw, "html.parser")
        # Remove script / style noise
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        text = soup.get_text(separator="\n")

    elif ext == ".docx":
        from docx import Document

        doc = Document(str(path))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        text = "\n".join(paragraphs)

    else:
        raise ValueError(f"Unsupported file type: {ext}")

    # Normalise whitespace
    text = "\n".join(line.rstrip() for line in text.splitlines())
    text = "\n".join(chunk for chunk in text.split("\n\n") if chunk.strip())

    return {"text": text, "metadata": metadata}
```

### Quick smoke-test

```python
doc = read_file("report.pdf")
print(doc["metadata"])          # {'source': 'report.pdf', 'page_count': 12}
print(doc["text"][:200])        # first 200 characters of extracted text
```

---

## Attaching richer metadata

Metadata you attach here flows all the way to the retrieval results, so add everything that helps users verify a cited source.

```python
import datetime

def read_file_with_meta(path: str) -> dict:
    doc = read_file(path)
    p = pathlib.Path(path)
    doc["metadata"].update({
        "filename": p.name,
        "file_size_kb": round(p.stat().st_size / 1024, 1),
        "loaded_at": datetime.datetime.utcnow().isoformat(),
    })
    return doc
```

---

## Hard cases to know about

!!! warning "Scanned PDFs"
    `pypdf` extracts the text layer of a PDF. Scanned documents have **no text layer** — you get an empty string. To handle scans you need an OCR step (e.g. `pytesseract` + `pdf2image`) before calling `read_file`.

!!! warning "Complex tables in PDFs"
    Tabular data in PDFs is often extracted as garbled columns. Consider `pdfplumber` or `camelot-py` when tables are critical to your corpus.

!!! note "Encoding"
    Always pass `errors="replace"` when reading arbitrary text files so a single bad byte does not crash the whole pipeline.

---

## Next steps

- [Text Splitting](text-splitting.md) — once you have clean text, learn how to chunk it for embedding
- [Tutorial 04 — PDF Q&A](../tutorials/04-pdf-qa.md) — end-to-end walkthrough using `read_file` on a real PDF
- [Indexing Pipeline](indexing-pipeline.md) — wire loaders into the full offline ingest script
