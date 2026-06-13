# Hosted Model SDKs

Hosted model APIs — OpenAI, Anthropic, and others — let you call frontier-scale models over HTTPS without managing local hardware. This page covers the Python SDKs for OpenAI and Anthropic, shows how to read API keys safely from environment variables, and discusses when to choose hosted versus local models.

## What you'll learn

- When and why to use a hosted model instead of a local one
- Chat completions and embeddings with the OpenAI SDK
- Chat completions with the Anthropic SDK
- Why Anthropic has no embeddings endpoint and what to use instead
- Reading API keys from environment variables
- Swapping local and hosted models behind a single interface

## When to use hosted models

| Situation | Recommendation |
|---|---|
| Need the highest available reasoning quality | Hosted (GPT-4o, Claude Opus) |
| Data must stay on-premise / air-gapped | Local (Ollama, Transformers) |
| Prototyping fast with no GPU | Hosted |
| High-volume production with predictable cost | Local or fine-tuned local |
| Compliance requires data residency | Local |
| Experimenting with the latest frontier capabilities | Hosted |

See [why local LLMs matter](../getting-started/why-local-llm.md) for a fuller cost-and-privacy analysis.

## Install

```bash
pip install openai anthropic
```

## Reading API keys from environment variables

Never hard-code credentials in source files. Store them in your shell profile or a `.env` file and load them at runtime.

```bash
# .env (never commit this file)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

```python
import os
from dotenv import load_dotenv  # pip install python-dotenv

load_dotenv()  # reads .env into os.environ

openai_key = os.environ["OPENAI_API_KEY"]        # raises KeyError if missing
anthropic_key = os.environ["ANTHROPIC_API_KEY"]
```

The OpenAI and Anthropic SDKs automatically read their respective keys from the environment — you do not need to pass them explicitly if the variables are set.

## OpenAI SDK — chat completions

```python
from openai import OpenAI

client = OpenAI()  # reads OPENAI_API_KEY from environment

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful RAG assistant."},
        {"role": "user", "content": "What is retrieval-augmented generation?"},
    ],
    max_tokens=256,
    temperature=0.3,
)

print(response.choices[0].message.content)
```

## OpenAI SDK — embeddings

```python
from openai import OpenAI

client = OpenAI()

result = client.embeddings.create(
    model="text-embedding-3-small",
    input=["RAG reduces hallucinations.", "Vector databases store embeddings."],
)

for item in result.data:
    print(f"index={item.index}, dim={len(item.embedding)}")
```

!!! warning "First-generation OpenAI embedding models removed; check current model list"
    The `text-similarity-*`, `text-search-*`, and `code-search-*` embedding models
    (first-generation) were **shut down on 2024-01-04**. Any code using them will
    receive a 404 error.

    `text-embedding-3-small` (shown above) is the current recommended entry-level
    model, but **OpenAI may retire it in the future**. Always check the
    [OpenAI models page](https://platform.openai.com/docs/models) for the current
    list before choosing a model for a new project. See
    [versions and deprecations](versions-and-deprecations.md) for the migration
    history.

## Anthropic SDK — chat completions

```python
from anthropic import Anthropic

client = Anthropic()  # reads ANTHROPIC_API_KEY from environment

message = client.messages.create(
    model="claude-opus-4-5",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Explain how RAG pipelines reduce hallucinations."},
    ],
)

print(message.content[0].text)
```

### System prompts with Anthropic

```python
from anthropic import Anthropic

client = Anthropic()

message = client.messages.create(
    model="claude-opus-4-5",
    max_tokens=512,
    system="You are a concise technical assistant. Answer in three sentences or fewer.",
    messages=[
        {"role": "user", "content": "What is a vector database?"},
    ],
)

print(message.content[0].text)
```

!!! warning "Anthropic has no embeddings endpoint"
    The Anthropic API does **not** expose a text-embeddings endpoint. You cannot
    call `client.embeddings.create(...)` — that method does not exist.

    For embeddings when using Anthropic as your generation model, use one of:

    - **Local**: `sentence-transformers` or Ollama (`nomic-embed-text`)
    - **Hosted**: OpenAI `text-embedding-3-small`, or Voyage AI (Anthropic's
      recommended embedding partner)

    See the [sentence-transformers SDK](sentence-transformers.md) and
    [versions and deprecations](versions-and-deprecations.md).

## Swapping local and hosted behind one interface

A simple wrapper function lets you switch providers without touching the rest of your pipeline:

```python
import os
from typing import Literal

def chat(
    prompt: str,
    backend: Literal["openai", "anthropic", "ollama"] = "ollama",
    model: str | None = None,
) -> str:
    """Return a text completion from the chosen backend."""
    if backend == "openai":
        from openai import OpenAI
        _model = model or "gpt-4o-mini"
        client = OpenAI()
        resp = client.chat.completions.create(
            model=_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
        )
        return resp.choices[0].message.content

    elif backend == "anthropic":
        from anthropic import Anthropic
        _model = model or "claude-opus-4-5"
        client = Anthropic()
        msg = client.messages.create(
            model=_model,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text

    elif backend == "ollama":
        import ollama
        _model = model or "llama3.2"
        resp = ollama.chat(
            model=_model,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp["message"]["content"]

    else:
        raise ValueError(f"Unknown backend: {backend!r}")


# Swap freely:
print(chat("What is RAG?", backend="ollama"))
print(chat("What is RAG?", backend="openai"))
print(chat("What is RAG?", backend="anthropic"))
```

## Cost and privacy trade-offs

| Dimension | Local (Ollama / Transformers) | Hosted (OpenAI / Anthropic) |
|---|---|---|
| Cost per token | Hardware only (one-time) | Per-token billing |
| Data privacy | Data never leaves your machine | Data sent to third-party servers |
| Model size | Limited by local RAM/VRAM | Frontier-scale models available |
| Latency | Low (no network round-trip) | Higher (depends on provider load) |
| Maintenance | You manage model versions | Provider handles updates |

## Next steps

- Understand the privacy argument for local models in [why local LLMs](../getting-started/why-local-llm.md)
- See practical API patterns in [working with model APIs](../python/ai-python/working-with-model-apis.md)
- Review the full deprecation timeline in [versions and deprecations](versions-and-deprecations.md)
