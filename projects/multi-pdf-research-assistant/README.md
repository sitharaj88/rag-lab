# Multi-PDF Research Assistant

A fully local RAG (Retrieval-Augmented Generation) application that lets you
ask natural-language questions across a collection of PDF files and receive
answers with **per-file and per-page citations** — no API keys, no internet
connection required after the first model download.

---

## What it does

1. **Ingests** any number of PDFs by splitting each page into overlapping text
   chunks and storing their embeddings in a local ChromaDB vector database.
2. **Retrieves** the most relevant chunks for your question using cosine
   similarity on sentence-transformer embeddings.
3. **Answers** your question using a locally-running Llama 3.2 model via
   Ollama, grounded strictly on the retrieved context.
4. **Cites** every claim with `filename.pdf (p.N)` tags so you can trace
   answers back to the exact page.

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.10 or later | `python --version` |
| [Ollama](https://ollama.com) installed and running | `ollama serve` |
| Llama 3.2 model pulled | `ollama pull llama3.2` |
| ~2 GB disk (model + embeddings) | First run downloads `all-MiniLM-L6-v2` (~80 MB) |

---

## Installation

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
```

---

## Quickstart

### Step 1 — Add your PDFs

Drop any number of PDF files into the `pdfs/` folder:

```
pdfs/
  annual-report-2023.pdf
  research-paper.pdf
  product-manual.pdf
```

### Step 2 — Ingest

Build the vector index (run once, or after adding new PDFs):

```bash
python ingest.py
```

To wipe the existing index and start from scratch:

```bash
python ingest.py --reset
```

### Step 3 — Ask questions

**Option A — Streamlit web UI (recommended):**

```bash
streamlit run app.py
```

Open the URL printed in the terminal (usually `http://localhost:8501`).

**Option B — Command-line REPL:**

```bash
python rag.py
```

Type your question and press Enter.  Type `quit` to exit.

---

## Example questions

These are illustrative — replace them with questions relevant to your own PDFs:

* "What were the key revenue figures for Q3?"
* "Summarise the methodology used in the study."
* "What safety precautions does the manual recommend?"
* "Compare the conclusions across all the documents."

---

## Configuration

All tunable parameters live in `config.py`:

| Constant | Default | Effect |
|---|---|---|
| `DATA_DIR` | `./pdfs` | Where to look for PDF files |
| `CHROMA_DIR` | `./chroma_db` | Where ChromaDB persists its data |
| `COLLECTION_NAME` | `multi_pdf_docs` | Name of the ChromaDB collection |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformers model |
| `CHUNK_SIZE` | `900` | Characters per chunk |
| `CHUNK_OVERLAP` | `150` | Overlap characters between chunks |
| `TOP_K` | `5` | Chunks retrieved per query |
| `OLLAMA_MODEL` | `llama3.2` | Ollama model tag |
| `OLLAMA_TEMPERATURE` | `0.1` | LLM sampling temperature |

---

## File overview

| File | Purpose |
|---|---|
| `config.py` | All path and model constants |
| `loader.py` | PDF page extraction and text chunking |
| `ingest.py` | Embed chunks and upsert into ChromaDB |
| `rag.py` | Retrieve + LLM pipeline; CLI REPL |
| `app.py` | Streamlit chat UI |
| `pdfs/` | Drop your PDF files here |
| `chroma_db/` | Auto-created by `ingest.py` |

---

## Troubleshooting

**`ollama: command not found`**
Install Ollama from https://ollama.com and make sure it is running
(`ollama serve`).

**`model "llama3.2" not found`**
Run `ollama pull llama3.2` to download the model.

**`Could not connect to the vector store`**
You need to run `python ingest.py` before starting the UI or REPL.

**Answers seem wrong or generic**
Try lowering `OLLAMA_TEMPERATURE` toward `0.0` in `config.py` for more
deterministic responses, or increase `TOP_K` to provide more context.

**Very slow on first run**
The sentence-transformers library downloads `all-MiniLM-L6-v2` (~80 MB) on
first use.  Subsequent runs use the cached model and are much faster.

**PDF text is garbled or empty**
Some PDFs are image-only scans.  pypdf cannot extract text from scanned images
without an OCR step.  Consider using a tool like `ocrmypdf` to add a text
layer before ingesting.
