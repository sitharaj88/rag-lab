# PDF Q&A with Page Citations

Load one or more PDF files, chunk them with page-aware metadata, index into ChromaDB, and answer questions with precise citations that include the source filename and page number.

## What you'll learn

- How to extract text from PDFs using `pypdf` with per-page awareness
- How to attach page-level metadata to every chunk so citations are accurate
- How to build a full ingest-and-query pipeline for PDF corpora
- How to display page citations alongside answers
- The limitations of text-based PDF extraction and when OCR is needed

## Prerequisites

- [Environment Setup](../getting-started/environment-setup.md)
- [Running a Local LLM with Ollama](../getting-started/local-llm-ollama.md)
- Completed [Tutorial 02 — RAG with Ollama + ChromaDB](02-rag-with-ollama-chroma.md)
- `pip install pypdf sentence-transformers chromadb ollama`

## Step 1 — How PDF loading differs from plain text

When you load plain text you know exactly where each document boundary is. PDFs add structure: a document may have dozens of pages, and a good citation tells the user not just which file but which page. `pypdf` gives access to text on a per-page basis, so you can record `{"source": "report.pdf", "page": 3}` as metadata on every chunk that came from page 3.

!!! warning "Scanned PDFs and OCR"
    `pypdf` extracts the text layer embedded in a PDF. If the PDF was created by scanning a physical page and no OCR was run, `pypdf` will return empty strings for those pages. To handle scanned PDFs you need an OCR tool such as `pytesseract` with `pdf2image`, or a managed service. This tutorial covers text-based PDFs only.

## Step 2 — Project layout

```
pdf_qa/
├── pdfs/           # put your PDF files here
│   └── sample.pdf  # created in Step 3 below for testing
├── pdf_ingest.py   # load, chunk, embed, store
└── pdf_query.py    # retrieve, cite, answer
```

Create the directory:

```bash
mkdir pdf_qa && mkdir pdf_qa/pdfs
```

## Step 3 — Create a sample PDF for testing

If you do not have a PDF handy, create a minimal one with `pypdf`:

```python
# make_sample_pdf.py — run once to create pdf_qa/pdfs/sample.pdf
from pypdf import PdfWriter
import os

writer = PdfWriter()

pages = [
    (
        "Introduction to Retrieval-Augmented Generation\n\n"
        "Retrieval-Augmented Generation (RAG) is an AI framework that "
        "enhances large language model outputs by retrieving relevant "
        "documents from an external knowledge base before generating a "
        "response. This approach reduces hallucinations and allows the "
        "model to cite specific sources, making it particularly valuable "
        "for question-answering over private or domain-specific corpora."
    ),
    (
        "Vector Databases\n\n"
        "A vector database stores numerical embeddings alongside metadata "
        "and provides fast approximate nearest-neighbour search. ChromaDB "
        "is a popular open-source option that persists data to a local "
        "SQLite-backed HNSW index. Other options include Pinecone, Weaviate, "
        "Qdrant, and pgvector. The choice of vector store affects scalability, "
        "query latency, and the richness of metadata filtering available."
    ),
    (
        "Chunking Strategies\n\n"
        "Chunking splits documents into smaller passages before embedding. "
        "The chunk size controls the granularity of retrieval: small chunks "
        "are precise but may lack context; large chunks are richer but may "
        "include irrelevant content. A common strategy is to chunk by "
        "sentence or paragraph with a fixed overlap to avoid splitting "
        "important phrases across boundaries."
    ),
]

os.makedirs("pdf_qa/pdfs", exist_ok=True)

for text in pages:
    page = writer.add_blank_page(width=612, height=792)
    # pypdf's add_blank_page does not support text directly;
    # write as a simple annotation instead.
    writer.add_page(page)

# For a proper test PDF with real text, use reportlab or fpdf2:
try:
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Helvetica", size=11)
    for text in pages:
        pdf.add_page()
        pdf.multi_cell(0, 8, text)
    pdf.output("pdf_qa/pdfs/sample.pdf")
    print("Created pdf_qa/pdfs/sample.pdf with fpdf2.")
except ImportError:
    print(
        "fpdf2 not installed. Install it with: pip install fpdf2\n"
        "Or place any existing PDF at pdf_qa/pdfs/sample.pdf."
    )
```

```bash
pip install fpdf2
python make_sample_pdf.py
```

## Step 4 — Write the ingest script

```python
# pdf_ingest.py
# Load PDFs page by page, chunk with overlap, embed, and store in ChromaDB.

import os
import uuid
import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PDF_DIR = "./pdfs"
CHROMA_PATH = "./chroma_db_pdf"
COLLECTION_NAME = "pdf_qa"
EMBED_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 300        # characters
CHUNK_OVERLAP = 60      # characters


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping character-level chunks, snapping to spaces."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end < len(text):
            snap = text.rfind(" ", start, end)
            if snap != -1:
                end = snap
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
        if start >= len(text):
            break
    return chunks


# ---------------------------------------------------------------------------
# PDF loading — page-aware
# ---------------------------------------------------------------------------
def load_pdf(pdf_path: str) -> list[dict]:
    """
    Extract text from each page of a PDF.

    Returns a list of dicts with keys 'text', 'source', and 'page'
    (1-indexed to match what a human sees in a PDF viewer).
    """
    reader = PdfReader(pdf_path)
    filename = os.path.basename(pdf_path)
    pages = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            pages.append({
                "text": text,
                "source": filename,
                "page": page_num,
            })
        else:
            print(
                f"  Warning: page {page_num} of {filename!r} returned no text. "
                "This page may be scanned — OCR is required for scanned pages."
            )
    return pages


# ---------------------------------------------------------------------------
# Main ingest routine
# ---------------------------------------------------------------------------
def ingest():
    pdf_files = [
        os.path.join(PDF_DIR, f)
        for f in os.listdir(PDF_DIR)
        if f.lower().endswith(".pdf")
    ]
    if not pdf_files:
        print(f"No PDF files found in {PDF_DIR!r}. Add PDFs and re-run.")
        return

    print(f"Found {len(pdf_files)} PDF file(s).")

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    embedder = SentenceTransformer(EMBED_MODEL)

    all_chunks: list[str] = []
    all_ids: list[str] = []
    all_metadatas: list[dict] = []

    for pdf_path in pdf_files:
        print(f"\nProcessing {pdf_path!r} …")
        pdf_pages = load_pdf(pdf_path)
        for page_data in pdf_pages:
            chunks = chunk_text(page_data["text"], CHUNK_SIZE, CHUNK_OVERLAP)
            print(
                f"  Page {page_data['page']}: "
                f"{len(page_data['text'])} chars → {len(chunks)} chunks"
            )
            for chunk_idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_ids.append(str(uuid.uuid4()))
                all_metadatas.append({
                    "source": page_data["source"],
                    "page": page_data["page"],
                    "chunk_index": chunk_idx,
                })

    if not all_chunks:
        print("No text extracted. Check that your PDFs are not scanned images.")
        return

    print(f"\nEmbedding {len(all_chunks)} chunks …")
    embeddings = embedder.encode(all_chunks, normalize_embeddings=True).tolist()

    print("Upserting into ChromaDB …")
    # Batch in groups of 500 to stay within ChromaDB's default batch limit.
    batch_size = 500
    for i in range(0, len(all_chunks), batch_size):
        collection.upsert(
            ids=all_ids[i:i + batch_size],
            documents=all_chunks[i:i + batch_size],
            embeddings=embeddings[i:i + batch_size],
            metadatas=all_metadatas[i:i + batch_size],
        )

    print(
        f"\nDone. Collection '{COLLECTION_NAME}' now contains "
        f"{collection.count()} chunks from {len(pdf_files)} PDF(s)."
    )


if __name__ == "__main__":
    ingest()
```

Run the ingest step:

```bash
cd pdf_qa
python pdf_ingest.py
```

## Step 5 — Write the query script

```python
# pdf_query.py
# Retrieve chunks from the PDF index and answer questions with page citations.

import chromadb
from sentence_transformers import SentenceTransformer
import ollama

# ---------------------------------------------------------------------------
# Configuration — must match pdf_ingest.py
# ---------------------------------------------------------------------------
CHROMA_PATH = "./chroma_db_pdf"
COLLECTION_NAME = "pdf_qa"
EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama3.2"
TOP_K = 4


# ---------------------------------------------------------------------------
# Initialise shared resources
# ---------------------------------------------------------------------------
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_collection(COLLECTION_NAME)
embedder = SentenceTransformer(EMBED_MODEL)


# ---------------------------------------------------------------------------
# Retrieve
# ---------------------------------------------------------------------------
def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    """Return the top_k most relevant chunks with source and page metadata."""
    query_embedding = embedder.encode([query], normalize_embeddings=True).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append({
            "text": doc,
            "source": meta.get("source", "unknown"),
            "page": meta.get("page", "?"),
            "chunk_index": meta.get("chunk_index", 0),
            "distance": dist,
        })
    return hits


# ---------------------------------------------------------------------------
# Generate
# ---------------------------------------------------------------------------
def answer(query: str) -> tuple[str, list[dict]]:
    """Retrieve context and generate a cited answer."""
    hits = retrieve(query)

    context_parts = []
    for i, hit in enumerate(hits, start=1):
        context_parts.append(
            f"[{i}] {hit['source']}, page {hit['page']}\n{hit['text']}"
        )
    context = "\n\n".join(context_parts)

    prompt = (
        "You are a helpful assistant that answers questions based on PDF "
        "documents. Answer using ONLY the numbered passages below. "
        "Cite each passage you use as [1], [2], etc. "
        "Include the filename and page number in your citations where possible. "
        "If the context does not contain the answer, say so.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\nAnswer:"
    )

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"], hits


# ---------------------------------------------------------------------------
# Interactive loop
# ---------------------------------------------------------------------------
def format_citation(hit: dict, index: int) -> str:
    return (
        f"  [{index}] {hit['source']}, page {hit['page']} "
        f"(chunk {hit['chunk_index']}, distance={hit['distance']:.4f})"
    )


if __name__ == "__main__":
    total = collection.count()
    print(f"PDF Q&A ready — {total} chunks indexed.")
    print("Type a question or 'quit' to exit.\n")
    while True:
        query = input("You: ").strip()
        if not query or query.lower() in {"quit", "exit", "q"}:
            break
        response_text, hits = answer(query)
        print(f"\nAssistant: {response_text}")
        print("\nPage citations:")
        for i, hit in enumerate(hits, start=1):
            print(format_citation(hit, i))
        print()
```

Run the query script:

```bash
python pdf_query.py
```

Sample session:

```
PDF Q&A ready — 12 chunks indexed.
Type a question or 'quit' to exit.

You: What is RAG and why does it reduce hallucinations?
```

## Step 6 — Understand the key design decisions

### Page-aware chunking

```python
all_metadatas.append({
    "source": page_data["source"],
    "page": page_data["page"],
    "chunk_index": chunk_idx,
})
```

By processing pages individually and storing the page number in metadata, every chunk carries a precise citation. When the answer references `[2]`, the user sees `report.pdf, page 7` — enough to verify the claim in the original document.

### Why chunk within a page, not across pages?

Chunking text that spans a page boundary would lose the page-number signal. Chunking within each page means you always know which page a chunk came from. The trade-off is that context near the bottom of one page and the top of the next is never in the same chunk, but in practice page breaks rarely split a single coherent idea.

### Batch upsert

```python
batch_size = 500
for i in range(0, len(all_chunks), batch_size):
    collection.upsert(...)
```

ChromaDB has a default maximum batch size. For large PDFs (hundreds of pages, thousands of chunks) a single `upsert` call with all chunks can exceed this limit. Batching in groups of 500 keeps the ingest robust regardless of corpus size.

### `TOP_K = 4` for PDFs

PDFs are often denser than plain text. Using `TOP_K = 4` instead of 2 or 3 gives the model slightly more context to work with, which helps when an answer spans multiple sections of a long document.

## Step 7 — Handling multiple PDFs

The ingest script already handles multiple files — it iterates over every `.pdf` in the `pdfs/` directory. To index a new PDF, copy it into `pdfs/` and re-run:

```bash
python pdf_ingest.py
```

!!! tip "Stable chunk IDs for incremental updates"
    The current script generates random UUIDs, so re-running ingest always adds new chunks. For large corpora you may want stable IDs derived from a hash of `(filename, page, chunk_index)` so that re-running replaces existing chunks in place rather than duplicating them.

## Step 8 — OCR for scanned PDFs

If `pypdf` returns empty text for pages, the PDF is likely scanned. You have two options:

**Option A — `pytesseract` (free, local):**

```bash
pip install pytesseract pdf2image
# Also install Tesseract OCR: https://github.com/tesseract-ocr/tesseract
```

```python
from pdf2image import convert_from_path
import pytesseract

def extract_text_with_ocr(pdf_path: str) -> list[dict]:
    images = convert_from_path(pdf_path, dpi=300)
    pages = []
    for page_num, image in enumerate(images, start=1):
        text = pytesseract.image_to_string(image).strip()
        if text:
            pages.append({
                "text": text,
                "source": os.path.basename(pdf_path),
                "page": page_num,
            })
    return pages
```

**Option B — Managed OCR API:** Services like AWS Textract or Azure Document Intelligence handle complex layouts (tables, columns, headers) more accurately than Tesseract but require an internet connection and API key.

!!! warning "OCR quality affects retrieval quality"
    OCR errors (garbled characters, missed words) propagate into embeddings and degrade retrieval accuracy. Always inspect a sample of extracted text before ingesting a large scanned corpus.

## Next steps

- [Document Loaders](../building-blocks/document-loaders.md): learn about loaders for other file types (Word, HTML, Markdown) and how to normalise text before chunking.
- [Evaluation](../advanced/evaluation.md): measure retrieval precision and answer faithfulness to know when your pipeline is good enough.
