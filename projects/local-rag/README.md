# Local RAG — sample project

A complete, **100% local** Retrieval-Augmented Generation app:

- **Embeddings** — `sentence-transformers/all-MiniLM-L6-v2` (runs on CPU)
- **Vector store** — ChromaDB (persists to `./chroma_db`)
- **LLM** — Llama 3.2 via [Ollama](https://ollama.com)
- **UI** — a Streamlit chat app (plus a terminal REPL)

No API keys. No cloud. Everything runs on your machine.

## 1. Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/download) installed and running

```bash
ollama pull llama3.2     # one-time model download (~2 GB)
ollama serve             # if it isn't already running as a service
```

## 2. Install

```bash
cd projects/local-rag
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 3. Add your documents

Drop `.txt`, `.md`, or `.pdf` files into the [`data/`](data/) folder.
Two sample documents are included so you can try it immediately.

## 4. Index them

```bash
python ingest.py             # load -> chunk -> embed -> store
# python ingest.py --reset   # wipe and re-index from scratch
```

## 5. Ask questions

Terminal:

```bash
python rag.py
```

Or the chat UI:

```bash
streamlit run app.py
```

Try: *"What are the two phases of RAG?"* or *"Why does chunk overlap matter?"*

## How it maps to the tutorials

| File | Concept | Read more |
|------|---------|-----------|
| `chunking.py` | Loading & splitting documents | [Chunking](../../docs/foundations/chunking.md) |
| `ingest.py` | The indexing pipeline | [Indexing pipeline](../../docs/building-blocks/indexing-pipeline.md) |
| `rag.py` | Retrieval + grounded generation | [Retrieval](../../docs/foundations/retrieval.md) |
| `app.py` | A chat interface | [Streamlit chat app](../../docs/tutorials/03-streamlit-chat-app.md) |

## Troubleshooting

- **`ConnectionError` / `model not found`** — make sure `ollama serve` is running and you ran `ollama pull llama3.2`.
- **First run is slow** — the embedding model downloads once (~90 MB), then caches.
- **Answers say "I don't know"** — the model is being faithful to your docs. Add more relevant files and re-index.
