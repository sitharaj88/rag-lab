# Ollama

Ollama is a zero-configuration local LLM runtime that downloads, quantizes, and serves open-weight models through a single binary — no Docker, no GPU drivers to wrangle beyond the standard CUDA/Metal stack.

## What you'll learn

- How to install Ollama and pull your first model
- The core CLI verbs: `pull`, `run`, `serve`, `list`, `show`
- How the Modelfile lets you customize system prompts and parameters
- How to call the REST API and the official Python client for chat, generation, and embeddings
- GPU vs CPU quantization trade-offs

---

## Installation

=== "macOS / Linux"

    ```bash
    curl -fsSL https://ollama.com/install.sh | sh
    ```

=== "Windows"

    Download the installer from [https://ollama.com/download](https://ollama.com/download) and run it.

After installation, `ollama serve` starts the background daemon (it auto-starts on macOS/Windows).

---

## Core CLI

```bash
# Pull a model from the library
ollama pull llama3.2

# Interactive chat in the terminal
ollama run llama3.2

# List locally cached models
ollama list

# Show model metadata and parameters
ollama show llama3.2

# Pull an embedding model
ollama pull nomic-embed-text

# Remove a model
ollama rm llama3.2
```

!!! info "Model library"
    Browse all available models and their tags at [https://ollama.com/library](https://ollama.com/library). Tags encode quantization level, e.g. `llama3.2:3b-instruct-q4_K_M`. When no tag is given, Ollama pulls the recommended default.

---

## Modelfile basics

A `Modelfile` lets you bake in a system prompt or tweak sampling parameters without changing the base weights.

```dockerfile
FROM llama3.2

SYSTEM """
You are a concise technical assistant. Answer in plain text, no markdown.
"""

PARAMETER temperature 0.2
PARAMETER top_p 0.9
PARAMETER num_ctx 4096
```

```bash
ollama create my-assistant -f Modelfile
ollama run my-assistant
```

---

## REST API

Ollama exposes a REST API on `http://localhost:11434` by default.

```bash
# Chat completions
curl http://localhost:11434/api/chat \
  -d '{
    "model": "llama3.2",
    "messages": [{"role": "user", "content": "Explain RAG in one sentence."}],
    "stream": false
  }'

# Embeddings
curl http://localhost:11434/api/embed \
  -d '{"model": "nomic-embed-text", "input": "Hello, world!"}'
```

---

## Python client

```bash
pip install ollama
```

### Chat

```python
import ollama

response = ollama.chat(
    model="llama3.2",
    messages=[{"role": "user", "content": "What is retrieval-augmented generation?"}],
)
print(response["message"]["content"])
```

### Streaming chat

```python
import ollama

for chunk in ollama.chat(
    model="llama3.2",
    messages=[{"role": "user", "content": "Explain embeddings."}],
    stream=True,
):
    print(chunk["message"]["content"], end="", flush=True)
```

### Embeddings

```python
import ollama

result = ollama.embed(model="nomic-embed-text", input="Semantic search with vectors")
vector = result["embeddings"][0]   # list of floats, length 768
print(len(vector))
```

---

## GPU and quantization notes

!!! tip "Choosing a quantization level"
    | Suffix | Bits | Quality | VRAM (7B model) |
    |--------|------|---------|-----------------|
    | `q4_K_M` | 4-bit | Good — recommended default | ~4 GB |
    | `q5_K_M` | 5-bit | Better | ~5 GB |
    | `q8_0` | 8-bit | Near-lossless | ~7 GB |
    | `f16` | 16-bit | Full precision | ~14 GB |

    If your GPU VRAM is insufficient, Ollama automatically offloads layers to CPU RAM — slower but functional. Set `OLLAMA_NUM_GPU` to control the split.

!!! warning "Apple Silicon"
    On M-series Macs, Ollama uses the Metal backend automatically. No CUDA setup is needed.

---

## Next steps

- [../getting-started/local-llm-ollama.md](../getting-started/local-llm-ollama.md) — Getting started guide with Ollama in a RAG context
- [embedding-models.md](embedding-models.md) — Choosing the right embedding model to pair with Ollama
