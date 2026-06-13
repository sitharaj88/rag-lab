# Packaging and Project Tooling

Think of packaging like putting your project in a labeled box before shipping it. The box lists everything inside (your code), what tools are needed to open it (your dependencies), and the exact version of each so it works the same way on your machine, your colleague's machine, and a server. The technical word for this process is **packaging**, and the tools here make it much less painful than it used to be.

## What you'll learn

- Standard Python project layout for AI applications
- Declaring dependencies with `pyproject.toml`
- Managing environments with `uv` (modern) and `pip`/`venv` (classic)
- Locking and pinning versions
- Configuring `ruff` for linting and formatting

---

## Project layout

A consistent folder structure means everyone — including you, six months from now — can navigate the project without guessing. Here is the recommended layout for an AI application.

```
my-rag-app/
├── pyproject.toml       # project metadata + dependencies
├── uv.lock              # exact pinned versions (commit this)
├── .python-version      # optional: pin interpreter version
├── src/
│   └── my_rag_app/
│       ├── __init__.py
│       ├── pipeline.py
│       └── models.py
├── tests/
│   └── test_pipeline.py
└── docs/
```

Placing source code under `src/` prevents accidental imports of un-installed code and is the current community recommendation. It is a small habit that avoids confusing bugs later.

---

## pyproject.toml

This single file is the one place you declare your project's name, Python version requirement, and all its dependencies. Here is a complete example for a RAG application.

```toml
[project]
name = "my-rag-app"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "httpx>=0.27",
    "pydantic>=2.7",
    "chromadb>=0.5",
    "sentence-transformers>=3.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8",
    "mypy>=1.10",
    "ruff>=0.4",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.ruff]
line-length = 88
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "UP"]   # pycodestyle, pyflakes, isort, pyupgrade

[tool.mypy]
python_version = "3.11"
strict = true
```

Everything lives in one place: runtime packages, dev-only packages, linting rules, and type-checking settings. You don't need a separate `setup.py`, `requirements.txt`, or `setup.cfg`.

---

## uv — the modern default

[uv](https://github.com/astral-sh/uv) is a fast tool (written in Rust) that handles your virtual environment, installs packages, and pins versions — all in one command. It replaces `pip`, `pip-tools`, and `virtualenv`.

These commands take you from nothing to a running project.

```bash
# Install uv (once, globally)
pip install uv          # or: curl -LsSf https://astral.sh/uv/install.sh | sh

# Create a new project
uv init my-rag-app
cd my-rag-app

# Add runtime dependencies
uv add httpx pydantic chromadb

# Add dev-only dependencies
uv add --dev pytest ruff mypy

# Create / activate the virtual environment
uv venv                 # creates .venv/
source .venv/bin/activate   # Linux/macOS
.venv\Scripts\activate      # Windows

# Run a script inside the managed environment (no manual activate needed)
uv run python src/my_rag_app/pipeline.py

# Run a tool (e.g. ruff) without installing it globally
uv run ruff check src/

# Lock dependencies (generates uv.lock)
uv lock

# Reproduce an exact environment from the lockfile
uv sync
```

After `uv sync`, anyone who clones your repo gets exactly the same package versions you have. No surprises.

!!! tip "uv replaces several tools"
    `uv add` = `pip install` + version pinning, `uv sync` = `pip install -r requirements.txt`, `uv run` = activating the venv first. You rarely need to think about the virtual environment directly — `uv run` handles it for you.

!!! example "Try it yourself"
    Run `uv init hello-rag` in a temporary folder, then `uv add httpx` and `uv run python -c "import httpx; print(httpx.__version__)"`. You'll see uv create the environment, install the package, and run your one-liner — all without you manually activating anything.

---

## Classic pip / venv workflow

If you are working on an existing project that does not use `uv`, or your environment only has `pip`, the classic workflow still works fine.

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\activate           # Windows

pip install httpx pydantic chromadb
pip freeze > requirements.txt    # save exact versions

# Later, reproduce:
pip install -r requirements.txt
```

!!! note "When to use classic pip"
    Use `pip`/`venv` when contributing to an existing project that does not use `uv`, or when a CI environment only has `pip` available. Both approaches give you isolated environments — `uv` is just faster and more convenient for new projects.

---

## Pinning versions

Leaving dependencies unpinned (just `httpx` with no version) means a future install might pull in a newer version with breaking changes. Always pin versions in production so your environment stays predictable.

```text
# requirements.txt (generated by pip freeze or uv export)
httpx==0.27.2
pydantic==2.7.4
chromadb==0.5.3
```

With `uv`, the `uv.lock` file handles this automatically — it records cryptographically verified exact versions for your entire dependency tree. Commit it to source control so every team member and CI run uses identical packages.

---

## Linting and formatting with ruff

`ruff` is a fast linter and formatter. It catches common bugs, enforces consistent style, and can fix many issues automatically.

```bash
uv run ruff check src/           # lint
uv run ruff check --fix src/     # auto-fix safe issues
uv run ruff format src/          # format (replaces black)
```

You can add a pre-commit hook so `ruff` runs automatically before every `git commit`, keeping the codebase tidy without anyone having to remember to run it.

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.4.4
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
```

??? note "Going deeper (optional)"
    **Semantic versioning**: version numbers like `2.7.4` follow the pattern `major.minor.patch`. A `major` bump (2 → 3) usually means breaking changes. A `minor` bump (2.7 → 2.8) usually means new features that don't break existing code. A `patch` bump (2.7.4 → 2.7.5) is a bug fix. When you write `pydantic>=2.7` in `pyproject.toml` you are saying "any 2.x version from 2.7 onwards is fine" — which is usually a safe bet.

    **Editable installs**: running `uv pip install -e .` installs your own package in "editable" mode — changes to your source files take effect immediately without reinstalling. Useful during development.

---

## Next steps

- [Setup and tooling](../foundations/setup-and-tooling.md) — interpreter installation, PATH, and editor configuration
- [Docker SDK](../../sdks/docker.md) — containerise your packaged RAG application for reproducible deployment
