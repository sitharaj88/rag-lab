# Multi-PDF Research Assistant

Build a completely local RAG system that ingests many PDFs at once, embeds
them page-by-page into a vector store, and answers questions with citations
that identify not just which file the answer came from but which exact page.

---

## What you'll learn

* How to extract text **page-by-page** from PDFs with pypdf so page numbers
  survive into the vector store as metadata.
* How to chunk text with an **overlapping character window** to preserve
  cross-sentence context at boundaries.
* How to configure ChromaDB with **cosine similarity** and attach structured
  metadata to every document.
* How to write a **grounded system prompt** that forces the LLM to cite
  sources and admit ignorance when the context does not contain an answer.
* How to surface per-file, per-page citations in a **Streamlit chat UI**.
* How to keep the whole stack **100% local** — no API keys, no cloud services.

---

## Architecture

```mermaid
flowchart LR
    subgraph Ingest ["Ingest (run once)"]
        A[pdfs/*.pdf] --> B(loader.py\nread pages + chunk)
        B --> C(sentence-transformers\nembed chunks)
        C --> D[(ChromaDB\npersistent store)]
    end

    subgraph Query ["Query (per question)"]
        E[User question] --> F(embed question)
        F --> G{ChromaDB\ntop-k cosine search}
        G --> H(retrieved chunks\n+ source + page)
        H --> I(build_prompt\n[file.pdf p.N] tags)
        I --> J(Ollama llama3.2)
        J --> K[Answer + citations]
    end

    D --> G
```

---

## File-by-file explanation

### `config.py`

Central constant store.  Every other module imports from here so there is a
single place to tune chunk size, model names, retrieval depth, and directory
paths.

```python
PROJECT_ROOT = Path(__file__).parent.resolve()
DATA_DIR     = PROJECT_ROOT / "pdfs"          # drop PDFs here
CHROMA_DIR   = PROJECT_ROOT / "chroma_db"     # auto-created by ingest
CHUNK_SIZE   = 900    # characters per chunk
CHUNK_OVERLAP = 150   # overlap between consecutive chunks
TOP_K        = 5      # chunks to retrieve per query
OLLAMA_MODEL = "llama3.2"
```

---

### `loader.py`

Two responsibilities:

1. **`read_pdf_pages(pdf_path)`** — opens a PDF with pypdf and yields
   `(page_number, text)` pairs.  Pages that contain no extractable text are
   silently skipped so downstream code never receives empty strings.

2. **`chunk_page(text, source, page, ...)`** — slices a page's text into
   overlapping windows.  The step size is `chunk_size - chunk_overlap`, so
   each successive window reuses the last `chunk_overlap` characters of the
   previous one.  This prevents a sentence that straddles a boundary from
   being cut in half.

3. **`load_chunks(data_dir)`** — iterates every `*.pdf` in the directory,
   calling the two functions above, and yields `Chunk` dataclass instances
   carrying `text`, `source` (filename), `page`, and `chunk_index`.

```python
@dataclass
class Chunk:
    text: str
    source: str       # e.g. "annual-report.pdf"
    page: int         # 1-based
    chunk_index: int  # 0-based within the page
```

---

### `ingest.py`

Orchestrates the full build pipeline:

1. Calls `load_chunks()` to get every chunk from every PDF.
2. Loads the `all-MiniLM-L6-v2` sentence-transformer model.
3. Embeds all chunks in a single batched call with
   `normalize_embeddings=True` — mandatory for cosine similarity to work
   correctly in ChromaDB.
4. Creates (or resets) a `PersistentClient` collection configured with
   `"hnsw:space": "cosine"`.
5. Upserts in batches of 512 using IDs of the form
   `"filename.pdf::p3::2"` so re-running the script is idempotent.

The `--reset` flag drops the collection before rebuilding — useful when you
want to remove PDFs that were previously ingested.

```bash
python ingest.py           # add / update
python ingest.py --reset   # full rebuild
```

---

### `rag.py`

Contains the `RAGPipeline` class and a standalone REPL.

**`retrieve(query, top_k)`**
Embeds the query with the same model used at ingest time, runs a
`collection.query()` call, and converts ChromaDB's cosine *distance* values
(range 0–2) to a similarity *score* in [0, 1]:

```python
score = max(0.0, 1.0 - distance / 2.0)
```

**`build_prompt(query, chunks)`**
Prefixes each chunk with a citation tag so the LLM sees exactly where each
passage came from:

```
--- Context 1 [annual-report.pdf p.7] ---
Revenue for Q3 was …
```

**`answer(query)`**
Calls `ollama.chat()` with a grounded system prompt that instructs the model
to:

* Answer **only** from the provided context.
* Say "I don't have enough information" when the answer is absent.
* Cite every claim using `[filename.pdf p.N]` tags.

Returns `(answer_text, list[RetrievedChunk])` so callers can render rich
citation UI.

---

### `app.py`

A Streamlit chat application with three key design choices:

* **`@st.cache_resource`** — the `RAGPipeline` is created once and shared
  across all user interactions in the session, preventing the embedding model
  from reloading on every question.
* **`st.session_state["messages"]`** — chat history accumulates across turns
  and is re-rendered on every Streamlit re-run, giving the appearance of a
  persistent conversation.
* **Sources expander** — each assistant message includes a collapsible panel
  listing `filename.pdf (p.N) — similarity 0.xx` plus a 280-character snippet
  for each retrieved chunk.

---

## Setup steps

### 1 — Prerequisites

* Python 3.10 or later
* [Ollama](https://ollama.com) installed and the daemon running (`ollama serve`)
* Llama 3.2 downloaded: `ollama pull llama3.2`

### 2 — Create a virtual environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### 4 — Add PDFs

Copy your PDF files into the `pdfs/` directory.  There is no limit on the
number of files.

### 5 — Build the index

```bash
python ingest.py
```

Watch the progress bar — sentence-transformers will download the embedding
model (~80 MB) on the first run.

### 6 — Run the assistant

```bash
# Web UI
streamlit run app.py

# Command-line REPL
python rag.py
```

---

## Configuration knobs

All constants are in `config.py`.  The most impactful ones:

| Knob | Effect of increasing |
|---|---|
| `CHUNK_SIZE` | Larger context per chunk; fewer chunks per page |
| `CHUNK_OVERLAP` | More repeated content at boundaries; better cross-sentence recall |
| `TOP_K` | More context passed to the LLM; higher token cost per query |
| `OLLAMA_TEMPERATURE` | More creative answers; less factual precision |

A good starting sweep for a new document set: keep `CHUNK_SIZE` around 800–1000
characters and `CHUNK_OVERLAP` at 15–20% of `CHUNK_SIZE`.

---

## How to extend this project

### Add a re-ranker

After the initial vector retrieval, a cross-encoder re-ranker scores each
`(query, chunk)` pair together rather than independently, which dramatically
improves precision for ambiguous queries.

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
pairs = [[query, c.text] for c in chunks]
scores = reranker.predict(pairs)
chunks = [c for _, c in sorted(zip(scores, chunks), reverse=True)]
```

See [../advanced/reranking.md](../advanced/reranking.md) for a full walkthrough.

### Swap the LLM

The `ollama.chat()` call in `rag.py` can be replaced with any compatible API.
Change `OLLAMA_MODEL` in `config.py` to any model you have pulled locally, for
example `mistral` or `phi3`.

### Add observability

Instrument `RAGPipeline.answer()` to log query, retrieved chunk IDs,
similarity scores, and answer length.  See
[../advanced/observability.md](../advanced/observability.md) for patterns.

### Handle scanned PDFs

pypdf cannot extract text from image-only PDFs.  Pre-process them with
[`ocrmypdf`](https://ocrmypdf.readthedocs.io/) to add a text layer, then
re-ingest normally.

### Persist chat history

Replace Streamlit's `st.session_state` with a SQLite-backed store so
conversations survive page refreshes.

---

## Troubleshooting

> **`model "llama3.2" not found`**
>
> Run `ollama pull llama3.2`.  Make sure `ollama serve` is running in a
> separate terminal or as a background service.

> **`Could not connect to the vector store`**
>
> The ChromaDB collection does not exist yet.  Run `python ingest.py` before
> starting the UI.

> **Empty or garbled PDF text**
>
> The PDF is likely a scanned image.  Use `ocrmypdf` to add a text layer
> before ingesting.

> **Slow first run**
>
> `sentence-transformers` downloads `all-MiniLM-L6-v2` (~80 MB) once and
> caches it locally.  Subsequent runs skip the download.

> **Answers ignore the documents**
>
> Check that `OLLAMA_TEMPERATURE` is low (0.0–0.2) and that `TOP_K` is high
> enough that the relevant passage is actually being retrieved.  Use the
> sources expander in the UI to inspect exactly which chunks were passed to
> the model.

---

## Next steps

* [local-rag.md](local-rag.md) — the simpler single-file RAG starter this
  project is based on.
* [../tutorials/04-pdf-qa.md](../tutorials/04-pdf-qa.md) — step-by-step
  tutorial for building a single-PDF Q&A pipeline from scratch.
* [../building-blocks/document-loaders.md](../building-blocks/document-loaders.md) — overview of
  document loading strategies (PDFs, DOCX, HTML, plain text).
* [../advanced/reranking.md](../advanced/reranking.md) — improve retrieval
  precision with a cross-encoder re-ranker.
* [../advanced/observability.md](../advanced/observability.md) — trace
  queries, log latency, and evaluate answer quality.
