# Environment Setup

Get a clean Python project ready for the RAG lab in about ten minutes: a virtual environment, all required packages, and a sensible folder layout.

## What you'll learn

- How to create and activate a virtual environment on Windows, macOS, and Linux
- Which Python packages the lab depends on and what each one does
- How to verify that every package installed correctly
- A recommended project directory layout

## Prerequisites

- Python 3.10 or later — check with `python --version` (Windows) or `python3 --version` (macOS/Linux).
- [Ollama installed and running](local-llm-ollama.md) (needed later; not required for this step).

!!! tip
    On Windows, install Python from [python.org](https://python.org) and tick **"Add Python to PATH"** during setup. Avoid the Microsoft Store version — it can cause `venv` issues.

## Step 1 — Create a virtual environment

### Windows (PowerShell)

```powershell
# Navigate to your project folder
cd C:\Users\ASUS\Documents\rag-lab

# Create the venv
python -m venv .venv

# Activate it
.\.venv\Scripts\Activate.ps1
```

!!! warning
    If PowerShell blocks the activation script, run this once: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

### macOS / Linux (bash)

```bash
cd ~/rag-lab

python3 -m venv .venv

source .venv/bin/activate
```

You will see `(.venv)` at the start of your prompt when the environment is active.

## Step 2 — Create requirements.txt

Create a file called `requirements.txt` in your project root with the following content:

```text
chromadb>=0.5.0
sentence-transformers>=3.0.0
ollama>=0.2.0
pypdf>=4.0.0
streamlit>=1.35.0
```

| Package | Purpose |
|---|---|
| `chromadb` | Local vector store for storing and querying embeddings |
| `sentence-transformers` | Loads embedding models (`all-MiniLM-L6-v2` and others) |
| `ollama` | Python client for the local Ollama server |
| `pypdf` | Parse PDF files for ingestion |
| `streamlit` | Build simple chat UIs for tutorial apps |

## Step 3 — Install packages

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

The first install may take several minutes. `sentence-transformers` downloads PyTorch and `chromadb` compiles some native extensions.

!!! note
    On Apple Silicon Macs, `sentence-transformers` will automatically use the MPS (Metal Performance Shaders) backend for faster embedding. No extra configuration needed.

## Step 4 — Verify the install

Run this quick check from a Python shell or save it as `check_env.py`:

```python
import chromadb
import sentence_transformers
import ollama
import pypdf
import streamlit

print("chromadb          ", chromadb.__version__)
print("sentence-transformers", sentence_transformers.__version__)
print("pypdf             ", pypdf.__version__)
print("streamlit         ", streamlit.__version__)
print("ollama client     OK")
print("\nAll packages imported successfully.")
```

```bash
python check_env.py
```

Expected output (versions may differ):

```text
chromadb           0.5.3
sentence-transformers 3.1.0
pypdf              4.2.0
streamlit          1.35.0
ollama client      OK

All packages imported successfully.
```

## Recommended project layout

```text
rag-lab/
├── .venv/                  # virtual environment (git-ignored)
├── requirements.txt
├── data/
│   └── docs/               # raw PDFs, text files to index
├── db/                     # ChromaDB persisted store
├── notebooks/              # exploration notebooks
├── src/
│   ├── ingest.py           # load + chunk + embed + store
│   ├── retrieve.py         # query the vector store
│   └── app.py              # Streamlit chat UI
└── check_env.py
```

Add `.venv/` and `db/` to your `.gitignore`.

## First-run model downloads

When you first use `sentence-transformers`, it will download `all-MiniLM-L6-v2` (~90 MB) from Hugging Face and cache it locally. Subsequent runs load from cache. Ollama models are downloaded separately with `ollama pull` — see [Setting up Ollama](local-llm-ollama.md).

## Next steps

- [Setting up Ollama](local-llm-ollama.md) — pull your first local LLM and verify the Python client works.
- [Tutorial 01 — Minimal RAG](../tutorials/01-minimal-rag.md) — build a working RAG pipeline with the environment you just set up.
