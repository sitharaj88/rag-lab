# Codebase Q&A

A 100%-local RAG system that indexes any source-code directory and answers
questions about it — with **file-path + line-range citations** and no API keys.

```
You: Where is the database connection configured?
Assistant: The database connection is set up in src/db/connection.py:1-45.
           The DSN is read from the environment variable DATABASE_URL …
Sources:
  [1] src/db/connection.py:1-45   (score=0.87)
  [2] src/config/settings.py:20-55  (score=0.74)
```

---

## Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| Python | ≥ 3.10 | python.org |
| Ollama | latest | [ollama.com](https://ollama.com) |
| llama3.2 model | — | `ollama pull llama3.2` |

> **No API keys required.** Everything runs locally.

---

## Install

```bash
# 1. Clone / navigate to this project
cd projects/codebase-qa

# 2. Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Pull the Ollama model (once)
ollama pull llama3.2
```

---

## Index a codebase

```bash
# Index a specific repository
python ingest.py --path /path/to/your/repo

# Index this project itself (great for demos)
python ingest.py --path .

# Wipe the index and start fresh
python ingest.py --path /path/to/your/repo --reset
```

Sample output:

```
[ingest] Walking '/path/to/your/repo' …
[walker] Scanned 83 files → 412 chunks
[ingest] Loading embedding model 'all-MiniLM-L6-v2' …
[ingest] Embedding and storing 412 chunks …
[ingest] Stored 412/412 chunks …

[ingest] Done! Collection 'codebase_qa' now has 412 vectors in './chroma_db'.
[ingest] Run `python rag.py` or `streamlit run app.py` to query.
```

---

## Query

### Interactive CLI (REPL)

```bash
python rag.py

# Increase retrieved chunks for complex questions
python rag.py --top-k 10
```

### Streamlit web app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Example questions

- "Where is X configured?" → finds config files and constants
- "How does the authentication middleware work?"
- "What does the `process_payment` function do?"
- "Which files import the `database` module?"
- "How is error handling structured in the API layer?"
- "What environment variables does this project use?"

---

## Re-indexing

The index is idempotent — re-running `python ingest.py --path <dir>` without
`--reset` safely **upserts** (updates existing chunks, adds new ones).  Use
`--reset` only when you want to completely rebuild, e.g. after large refactors.

---

## Project structure

```
codebase-qa/
├── config.py        ← all tunable constants
├── walker.py        ← directory walker + line-based chunker
├── ingest.py        ← embed chunks → store in ChromaDB
├── rag.py           ← retrieve → prompt → answer (+ REPL)
├── app.py           ← Streamlit chat UI
├── requirements.txt
└── chroma_db/       ← created at runtime (git-ignored)
```

---

## Troubleshooting

**`Collection 'codebase_qa' does not exist`**
→ You haven't run `ingest.py` yet, or the `chroma_db/` directory was deleted.
Run `python ingest.py --path <dir>`.

**`ollama: command not found` / connection error**
→ Start Ollama: `ollama serve` (it runs in the background by default on macOS/
Linux; on Windows start the Ollama app).

**Very slow embedding on first run**
→ `all-MiniLM-L6-v2` is ~90 MB and is downloaded once into `~/.cache/
huggingface/hub/`.  Subsequent runs use the local cache.

**Answers don't mention the right files**
→ The target directory may not be indexed.  Re-run `ingest.py --path <correct-dir>`.
Also try increasing `TOP_K` in `config.py` (or `--top-k` flag).

**Out of memory during embedding**
→ Lower the batch size in `ingest.py`'s `embed_and_store()` call
(`batch_size=32` or `batch_size=16`).
