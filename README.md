# 🧪 RAG Lab — Learn Retrieval-Augmented Generation

A free, open, **beginner-to-advanced learning portal** for building Retrieval-Augmented Generation (RAG) systems — with a strong focus on **running everything locally** using open-source LLMs (no API keys, no cloud bills required).

> Built with [MkDocs Material](https://squidfunk.github.io/mkdocs-material/) and deployed to GitHub Pages.

## What's inside

- **Getting Started** — what RAG is, why local LLMs, environment setup, and installing Ollama.
- **Foundations** — embeddings, vector databases, chunking, retrieval, and prompting.
- **Building Blocks** — document loaders, text splitting, the indexing pipeline, and generation.
- **Tutorials** — hands-on, copy-pasteable builds from a 30-line minimal RAG to a Streamlit chat app.
- **Advanced** — hybrid search, reranking, query transformation, agentic RAG, GraphRAG, evaluation, and production concerns.
- **Tools & Ecosystem** — Ollama, LangChain, LlamaIndex, ChromaDB, vector stores, and embedding models.
- **Projects** — full, runnable sample apps (see [`projects/`](projects/)).

## Run the docs site locally

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
mkdocs serve
```

Open <http://127.0.0.1:8000>.

## Run the sample RAG project

A complete, 100%-local RAG app (Ollama + ChromaDB + sentence-transformers) lives in
[`projects/local-rag/`](projects/local-rag/). See its README for setup.

## Deploy to GitHub Pages

1. Push this repo to GitHub.
2. In **Settings → Pages → Build and deployment**, set **Source** to **GitHub Actions**.
3. Push to `main`. The workflow in [`.github/workflows/deploy.yml`](.github/workflows/deploy.yml) builds and publishes automatically.

Once enabled, the site publishes to **<https://sitharaj88.github.io/rag-lab/>**.

## Contributing

Contributions of new tutorials, fixes, and tools coverage are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).

## Support

Built by [Sitharaj](https://sitharaj.in). If RAG Lab helps you, a ⭐ on GitHub means a lot — and you can
[buy me a coffee ☕](https://www.buymeacoffee.com/sitharaj88) to support the project.

## License

- **Code** (sample projects, config): [MIT](LICENSE)
- **Docs / tutorial content**: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
