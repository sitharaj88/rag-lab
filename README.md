# 🧪 RAG Lab

### Learn to build Retrieval-Augmented Generation systems — from your first 30-line script to production-grade pipelines — running **entirely on your own machine**.

[![Live Site](https://img.shields.io/badge/docs-live-2ea44f?logo=githubpages)](https://sitharaj88.github.io/rag-lab/)
[![Built with Material for MkDocs](https://img.shields.io/badge/built%20with-Material%20for%20MkDocs-526CFE?logo=materialformkdocs&logoColor=white)](https://squidfunk.github.io/mkdocs-material/)
[![Code License: MIT](https://img.shields.io/badge/code-MIT-blue.svg)](LICENSE)
[![Docs License: CC BY 4.0](https://img.shields.io/badge/docs-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

**RAG Lab** is a free, open, beginner→advanced learning portal. No API keys. No cloud bills. Every example is designed to run locally with open-source models via [Ollama](https://ollama.com), [ChromaDB](https://www.trychroma.com/), and [sentence-transformers](https://www.sbert.net/) — so you can experiment as much as you like, then carry the same architecture over to hosted APIs when you need to.

🔗 **Read it now: <https://sitharaj88.github.io/rag-lab/>**

---

## ✨ Why this exists

Most RAG tutorials either hand-wave the fundamentals or drown you in a single framework. RAG Lab does neither:

- **Local-first** — learn the *mechanics* of RAG without a paywall; nothing is hidden behind an API.
- **Plain-English** — the Python & ML pages lead with everyday analogies and keep the scary math optional.
- **Current, not stale** — the SDK pages are written against 2026 APIs, with deprecated patterns flagged.
- **Hands-on** — four runnable sample projects, two Colab notebooks, and 15 graded exercises.

## 📚 What's inside

| Section | You'll learn to… |
|---------|------------------|
| **Getting Started** | Explain RAG, set up Python + Ollama, and run a local model. |
| **Python Track** | Go from zero → AI-ready Python in three tiers: foundations → data/ML (NumPy, pandas, scikit-learn) → AI Python (typing/Pydantic v2, async, packaging). |
| **Foundations** | Reason about embeddings, vector databases, chunking, retrieval, and prompting. |
| **Building Blocks** | Assemble an ingestion → index → generation pipeline. |
| **Tutorials** | Build a minimal RAG, a ChromaDB-backed app, a Streamlit chat UI, and PDF Q&A. |
| **SDKs & Libraries** | Use LangChain, LlamaIndex, Haystack, Transformers, sentence-transformers, Ollama, FAISS/Qdrant/Weaviate/pgvector, FastAPI, Streamlit, Gradio, Docker — with current APIs. |
| **Advanced** | Hybrid search, reranking, query routing, late chunking, agentic & Graph RAG, multimodal, RAG-over-SQL, evaluation, observability, security, and production. |
| **Tools & Ecosystem** | Choose between local serving engines (Ollama, vLLM, LM Studio, llama.cpp) and embedding/rerank models. |
| **Projects** | Study and extend full, runnable sample applications. |
| **Resources** | Glossary, further reading, Colab notebooks, and exercises. |

> **87 pages · 4 sample projects · 2 notebooks · 15 exercises**, all building locally.

## 🛠️ Sample projects

All 100% local — no API keys. See [`projects/`](projects/).

| Project | What it does | Level |
|---------|--------------|-------|
| [**local-rag**](projects/local-rag/) | End-to-end RAG with a Streamlit chat UI — the canonical starter. | Beginner |
| [**multi-pdf-research-assistant**](projects/multi-pdf-research-assistant/) | Q&A across many PDFs with per-file & per-page citations. | Intermediate |
| [**codebase-qa**](projects/codebase-qa/) | Index a code repo and ask questions, with `path:line` citations. | Intermediate |
| [**hybrid-rerank-rag**](projects/hybrid-rerank-rag/) | BM25 + dense → RRF fusion → cross-encoder reranking, with a stage-by-stage `compare.py`. | Advanced |

## 🚀 Quickstart

### Read / edit the docs site

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
mkdocs serve                     # live preview at http://127.0.0.1:8000
```

### Run the starter RAG app

Needs [Ollama](https://ollama.com/download) running locally.

```bash
ollama pull llama3.2
cd projects/local-rag
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python ingest.py                 # load → chunk → embed → store
streamlit run app.py             # or:  python rag.py
```

Each project folder has its own README with full instructions.

## 🗂️ Repository layout

```
rag-lab/
├── docs/                 # all tutorial content (the MkDocs site)
├── projects/             # four runnable, local-first sample apps
├── notebooks/            # Colab-ready Jupyter notebooks
├── mkdocs.yml            # site config & navigation
├── requirements.txt      # docs-site build dependencies
└── .github/workflows/    # GitHub Pages deploy (manual)
```

## 🌐 Deploy to GitHub Pages

1. In **Settings → Pages → Build and deployment**, set **Source** to **GitHub Actions** (one-time).
2. **Deploy on demand**: **Actions** tab → **Deploy RAG Lab to GitHub Pages** → **Run workflow** — or run `gh workflow run deploy.yml`.

Deployment is intentionally **manual**: pushing to `main` does *not* auto-publish. The workflow ([`.github/workflows/deploy.yml`](.github/workflows/deploy.yml)) builds with `--strict` (failing on any broken link) and publishes to <https://sitharaj88.github.io/rag-lab/>.

## 🤝 Contributing

New tutorials, fixes, and tool coverage are all welcome — see [CONTRIBUTING.md](CONTRIBUTING.md). Run `mkdocs build --strict` before opening a PR; it fails on broken links and missing nav entries.

## ❤️ Support

Built by [Sitharaj](https://sitharaj.in). If RAG Lab helps you learn, a ⭐ on GitHub means a lot — and you can [**buy me a coffee ☕**](https://www.buymeacoffee.com/sitharaj88) to keep the project free and growing.

## 📄 License

- **Code** (sample projects, config): [MIT](LICENSE)
- **Docs / tutorial content**: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
