# Notebooks

Run the RAG Lab examples directly in your browser — no local setup required — or pull them down and run them on your own machine for full control.

## What you'll learn

- How to open and run any notebook in Google Colab with one click
- How to run notebooks locally with Jupyter or VS Code
- Which notebook matches which tutorial in the site

---

## Cloud vs local: which should I use?

| | Google Colab | Local runtime |
|---|---|---|
| **Setup** | Zero — just a Google account | Python + Jupyter or VS Code |
| **GPU** | Free tier available (~12 h cap, not guaranteed) | Your hardware |
| **Ollama** | Not supported on the default cloud runtime | Fully supported |
| **Persistence** | Resets on session end | Your disk |

!!! tip "Quick start recommendation"
    If you just want to read along and run cells without installing anything, use Colab.
    If you are working through the [Ollama + Chroma tutorial](../tutorials/02-rag-with-ollama-chroma.md) or need a persistent environment, run locally.

!!! warning "Ollama steps require a local runtime"
    Notebooks that use Ollama (e.g. loading `llama3` or `nomic-embed-text`) cannot connect to a locally running Ollama server from the Colab cloud runtime. Either skip those cells in Colab or switch to a **local runtime** (Runtime → Change runtime type → Local runtime in Colab). Notebooks that only use `sentence-transformers` or the OpenAI-compatible API work fine on the default Colab cloud.

---

## Available notebooks

### 01 — Minimal RAG

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sitharaj88/rag-lab/blob/main/notebooks/01-minimal-rag.ipynb)

Build a retrieval-augmented generation pipeline from scratch in under 50 lines of Python, using `sentence-transformers` for embeddings and a simple in-memory vector store.

**Colab compatible:** Yes — uses `sentence-transformers` (pip-installable), no Ollama required.

Matches tutorial: [Minimal RAG](../tutorials/01-minimal-rag.md)

---

### 02 — RAG Evaluation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sitharaj88/rag-lab/blob/main/notebooks/02-rag-evaluation.ipynb)

Measure the quality of your retrieval pipeline and generated answers with standard metrics including precision@k, recall@k, MRR, and faithfulness scoring.

**Colab compatible:** Yes — evaluation helpers are pure Python or lightweight pip packages.

Matches tutorial: [Evaluation](../advanced/evaluation.md)

---

## Running notebooks locally

### Option 1 — Jupyter Lab / Notebook

```bash
# Install Jupyter if you haven't already
pip install jupyterlab

# Clone the repo
git clone https://github.com/sitharaj88/rag-lab.git
cd rag-lab

# Install project dependencies
pip install -r requirements.txt

# Launch Jupyter Lab
jupyter lab notebooks/
```

Navigate to any `.ipynb` file in the file browser on the left and click to open it.

### Option 2 — VS Code

```bash
# Install the Jupyter extension from the Extensions panel (Ctrl+Shift+X)
# Search: "Jupyter" by Microsoft

# Open the repo folder
code /path/to/rag-lab

# Open any .ipynb file directly — VS Code renders it natively
```

!!! note "Kernel selection"
    When VS Code asks you to select a kernel, choose the Python environment where you installed `requirements.txt` (usually the virtual-env you created during [environment setup](../getting-started/environment-setup.md)).

### Keeping outputs clean for git

```bash
# Strip outputs before committing so diffs stay readable
pip install nbstripout
nbstripout --install          # installs a git pre-commit hook
```

---

## Where to go next

- Work through the hands-on challenges in [Exercises](exercises.md)
- Dive deeper into evaluation metrics in [Evaluation](../advanced/evaluation.md)
- Follow along step-by-step in [Minimal RAG tutorial](../tutorials/01-minimal-rag.md)
