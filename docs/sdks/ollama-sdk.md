# Ollama Python SDK

The Ollama Python SDK is a thin client over the local Ollama REST server. It gives you chat completions, text generation, and embeddings from quantised open-source models — entirely on your machine, with no API key and no data leaving your network.

## What you'll learn

- Sending chat and generate requests with `ollama.chat()` and `ollama.generate()`
- Producing embeddings with `ollama.embeddings()`
- Streaming responses token-by-token
- Passing model options such as `temperature`
- Using `ollama.Client` for custom hosts and `ollama.AsyncClient` for async code
- Pulling models programmatically
- A minimal local RAG generation snippet

## Install

```bash
pip install ollama
```

Ollama itself must be running locally. Download it from [ollama.com](https://ollama.com) and start the server:

```bash
ollama serve          # starts the REST server on http://localhost:11434
ollama pull llama3.2  # download a model before first use
```

See the [Ollama tool guide](../tools/ollama.md) and [local LLM quickstart](../getting-started/local-llm-ollama.md) for installation details.

## Chat completions

```python
import ollama

response = ollama.chat(
    model="llama3.2",
    messages=[
        {"role": "system", "content": "You are a helpful RAG assistant."},
        {"role": "user", "content": "What is retrieval-augmented generation?"},
    ],
)

print(response["message"]["content"])
```

## Text generation

`ollama.generate()` is a lower-level call that accepts a single prompt string — useful for completion-style tasks or when you are building your own message formatting.

```python
import ollama

result = ollama.generate(
    model="llama3.2",
    prompt="Complete this sentence: Retrieval-augmented generation works by",
)
print(result["response"])
```

## Embeddings

```python
import ollama

# Single prompt
resp = ollama.embeddings(model="nomic-embed-text", prompt="RAG reduces hallucinations.")
vector = resp["embedding"]
print(len(vector))  # dimension depends on the model

# ollama.embed is an alias accepted by newer server versions
resp2 = ollama.embed(model="nomic-embed-text", input="Another sentence.")
```

Pull an embedding model first if you have not already:

```bash
ollama pull nomic-embed-text
```

## Streaming

Pass `stream=True` to receive a generator of partial response chunks. This is essential for chat UIs where you want text to appear progressively.

```python
import ollama

stream = ollama.chat(
    model="llama3.2",
    messages=[{"role": "user", "content": "Explain vector databases in three sentences."}],
    stream=True,
)

for chunk in stream:
    print(chunk["message"]["content"], end="", flush=True)
print()  # newline after stream ends
```

Streaming also works with `ollama.generate()`:

```python
import ollama

for chunk in ollama.generate(model="llama3.2", prompt="List three RAG use cases.", stream=True):
    print(chunk["response"], end="", flush=True)
```

## Model options (temperature, etc.)

Pass an `options` dict to control sampling behaviour:

```python
import ollama

response = ollama.chat(
    model="llama3.2",
    messages=[{"role": "user", "content": "Give a creative title for a RAG tutorial."}],
    options={"temperature": 0.9, "top_p": 0.95, "num_predict": 60},
)
print(response["message"]["content"])
```

## Custom host with ollama.Client

Use `ollama.Client` when the server is not on `localhost:11434` — for example a remote machine or a Docker container.

```python
import ollama

client = ollama.Client(host="http://192.168.1.50:11434")

response = client.chat(
    model="llama3.2",
    messages=[{"role": "user", "content": "Hello from a remote Ollama server!"}],
)
print(response["message"]["content"])
```

## Async usage with AsyncClient

```python
import asyncio
import ollama

async def main():
    client = ollama.AsyncClient()
    response = await client.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": "What is a vector database?"}],
    )
    print(response["message"]["content"])

asyncio.run(main())
```

## Pulling models programmatically

```python
import ollama

# Pull a model — streams progress updates
for progress in ollama.pull("mistral", stream=True):
    status = progress.get("status", "")
    completed = progress.get("completed", 0)
    total = progress.get("total", 0)
    if total:
        print(f"{status}: {completed}/{total}", end="\r")
print("\nDone.")
```

## Minimal local RAG generation snippet

This snippet shows the generation step of a RAG pipeline: take retrieved context passages, inject them into a prompt, and call the local LLM.

```python
import ollama

def rag_generate(query: str, context_passages: list[str], model: str = "llama3.2") -> str:
    """Generate an answer grounded in retrieved context."""
    context = "\n\n".join(f"- {p}" for p in context_passages)
    prompt = (
        f"Answer the question using ONLY the context below. "
        f"If the answer is not in the context, say so.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}"
    )
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.2},  # low temperature for factual answers
    )
    return response["message"]["content"]


# Example usage
passages = [
    "RAG stands for Retrieval-Augmented Generation.",
    "It retrieves relevant documents from a knowledge base and passes them to an LLM.",
    "This grounds the model's answer in real data and reduces hallucinations.",
]

answer = rag_generate("What does RAG stand for and how does it work?", passages)
print(answer)
```

For a complete pipeline that also handles embedding and retrieval, see [generation building block](../building-blocks/generation.md).

## Next steps

- Set up Ollama locally with the [Ollama tool guide](../tools/ollama.md)
- Follow the end-to-end walkthrough in the [local LLM quickstart](../getting-started/local-llm-ollama.md)
- See how the generation step fits into a full pipeline in [generation](../building-blocks/generation.md)
