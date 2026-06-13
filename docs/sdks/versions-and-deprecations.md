# Versions & Deprecations

A fast-moving cheat-sheet of **what changed** in the RAG/LLM stack and which older patterns to avoid. The library world moves quickly, so the value here isn't the exact version numbers — it's knowing *which APIs were recently replaced* so your code (and the tutorials you follow elsewhere) don't quietly use a deprecated pattern.

!!! info "Snapshot date"
    Verified June 2026 against vendor primary docs and PyPI/GitHub release data. **Pin versions in your own projects and re-check the official docs before relying on a specific number.** Several of these libraries shipped breaking changes within weeks of this writing.

## The big shifts

| Library | Use this (current) | Avoid this (deprecated/removed) |
|---------|--------------------|---------------------------------|
| **LangChain v1** | `from langchain.agents import create_agent` | `create_react_agent`; legacy `LLMChain`/`ConversationChain` (moved to `langchain-classic`) |
| **LlamaIndex** (v0.10+) | global `Settings` object; modular `llama-index-*` packages; namespace imports | `ServiceContext` (removed in v0.11); flat `from llama_index.llms import OpenAI` |
| **Haystack 2.x** | `pip install haystack-ai`; `add_component()` + `connect()` | `farm-haystack` (1.x, EOL Mar 2025); auto-wired node chaining |
| **sentence-transformers v5** | `encode(inputs=...)`; `get_embedding_dimension()`; `SparseEncoder` | `encode(sentences=...)`; `get_sentence_embedding_dimension()` (warn) |
| **Transformers v5** | `AutoModelFor*` + `from_pretrained()`; `dtype=` | `torch_dtype=` (legacy, warns) |
| **FastAPI** (0.126+) | Pydantic **v2** only (`model_dump`, `Field`) | Pydantic v1 / `pydantic.v1` shim (removed in 0.128) |
| **qdrant-client** (v1.16+) | `upsert()`, `query_points()` | `search`, `recommend`, `discovery`, `*_batch` (removed) |
| **OpenAI embeddings** | `text-embedding-3-small` / `-3-large` | first-gen `text-similarity-ada-001` etc. (shut down 2024-01-04) |

## Details & why it matters

### LangChain v1 — agents replace chains

LangChain reorganized around **agents built with `create_agent`** (running on LangGraph with a middleware system), and moved the old chain abstractions out of the core package.

```python
# Current (LangChain v1)
from langchain.agents import create_agent

# Deprecated — legacy chains now require a separate install:
#   pip install langchain-classic
from langchain_classic.chains import LLMChain   # only if you must
```

For straightforward RAG, prefer **LCEL** (the `|` pipe composition) over legacy chains. Install the modular packages you need: `langchain`, `langchain-core`, `langchain-community`, and partner packages like `langchain-ollama` and `langchain-chroma`. See the [LangChain page](langchain.md).
[Source: LangChain v1 migration guide.](https://docs.langchain.com/oss/python/migrate/langchain-v1)

### LlamaIndex — `Settings` replaces `ServiceContext`

```python
# Current (v0.10+)
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

Settings.llm = Ollama(model="llama3.2")
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Removed: ServiceContext.from_defaults(...) / set_global_service_context(...)
```

Install the slim core plus per-integration packages: `pip install llama-index-core llama-index-llms-ollama llama-index-embeddings-huggingface`. Imports are **namespace-qualified** (`from llama_index.llms.ollama import Ollama`), not flat. See the [LlamaIndex page](llamaindex.md).
[Source: LlamaIndex v0.10 + ServiceContext migration.](https://docs.llamaindex.ai/en/stable/module_guides/supporting_modules/service_context_migration/)

### Haystack 2.x — explicit components

Install `haystack-ai` (the 1.x `farm-haystack` cannot coexist with it). Pipelines are assembled from named components and wired explicitly:

```python
from haystack import Pipeline
pipe = Pipeline()
pipe.add_component("retriever", retriever)
pipe.add_component("prompt", prompt_builder)
pipe.connect("retriever.documents", "prompt.documents")
```

[Source: Haystack 2.x migration.](https://docs.haystack.deepset.ai/docs/migration)

### sentence-transformers v5 — renamed params

```python
from sentence_transformers import SentenceTransformer, CrossEncoder, SparseEncoder
model = SentenceTransformer("all-MiniLM-L6-v2")
emb = model.encode(inputs=["hello world"], normalize_embeddings=True)  # not sentences=
dim = model.get_embedding_dimension()                                  # not get_sentence_embedding_dimension()
```

Old names still work but emit deprecation warnings. `SparseEncoder` (SPLADE-style sparse embeddings) is new in v5. See the [sentence-transformers page](sentence-transformers.md).
[Source: sentence-transformers migration guide.](https://sbert.net/docs/migration_guide.html)

### Transformers v5 — `dtype`, not `torch_dtype`

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B", dtype="auto")  # not torch_dtype=
```

[Source: Transformers models docs.](https://huggingface.co/docs/transformers/en/models)

### FastAPI — Pydantic v2 only

From 0.126.0 FastAPI dropped Pydantic v1; the `pydantic.v1` shim was removed in 0.128.0. Teach and use Pydantic v2 (`BaseModel`, `model_dump()`, `Field`, `field_validator`). See the [FastAPI page](fastapi.md).
[Source: FastAPI Pydantic v1→v2 migration.](https://fastapi.tiangolo.com/how-to/migrate-from-pydantic-v1-to-pydantic-v2/)

### qdrant-client — unified query API

The legacy `search`/`recommend`/`discovery` (and their `*_batch` variants) were **removed** in v1.16. Use `upsert()` to write and `query_points()` to read. See the [Qdrant page](qdrant.md).
[Source: qdrant-client releases.](https://github.com/qdrant/qdrant-client/releases)

### OpenAI embeddings — use v3 models

First-generation embedding models were shut down in January 2024. Use `text-embedding-3-small` (cheap) or `text-embedding-3-large` (higher quality).

!!! warning "Even v3 is moving"
    OpenAI has slated `text-embedding-3-small` for retirement as well — always check the [current model list](https://platform.openai.com/docs/models) rather than hard-coding a model name. See the [hosted SDKs page](hosted-sdks.md).

## Not independently re-verified

The research pass that produced this page did **not** adversarially verify versions/APIs for every library on the site — notably the data/ML stack (NumPy, pandas, matplotlib, scikit-learn), the Ollama Python SDK, ChromaDB, FAISS, Weaviate, pgvector, the data loaders (`unstructured`, `pypdf`), and Streamlit/Gradio/Docker. Those pages are written against stable, well-documented APIs, but absence of a verified claim is not a guarantee — check their official docs for your version.

## Next steps

- Apply these directly in **[LangChain](langchain.md)** and **[LlamaIndex](llamaindex.md)**.
- See the **[Tools](../tools/index.md)** section for the ecosystem map and comparisons.
