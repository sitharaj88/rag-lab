# Contributing to RAG Lab

Thanks for helping make RAG Lab better! This project is a community learning resource.

## Ways to contribute

- **Fix** typos, broken links, or outdated commands.
- **Improve** an existing tutorial with clearer explanations or diagrams.
- **Add** a new tutorial, tool page, or sample project.

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
mkdocs serve                     # live-reload preview at http://127.0.0.1:8000
```

## Style guide for docs pages

Every content page should:

1. Open with a short intro and a **"What you'll learn"** list.
2. State **Prerequisites** (link to earlier pages) where relevant.
3. Use fenced code blocks with an explicit language (` ```python `).
4. Use Material admonitions for asides: `!!! note`, `!!! tip`, `!!! warning`.
5. Prefer **local, open-source** tools (Ollama, ChromaDB, sentence-transformers) in examples.
6. End with a **"Next steps"** section linking onward.

Keep a friendly, practical tone. Show working code over abstract description.

## Adding a page to the navigation

Add your new file under `docs/` and register it in the `nav:` tree in [`mkdocs.yml`](mkdocs.yml).
Run `mkdocs build --strict` locally — it fails on broken links and missing nav entries.

## Pull requests

- One topic per PR keeps reviews fast.
- Run `mkdocs build --strict` before opening the PR.
- Be kind in reviews. We're all learning.
