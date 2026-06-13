# Working with Model APIs

Calling an LLM from Python is a bit like sending a letter and waiting for a reply. You write your message (the prompt), put it in an envelope with your return address (the API key), and send it off. The model reads it, writes back, and your code reads the reply. The technical term for this exchange is an **API call**, and the patterns here make it safe, reliable, and easy to reuse.

## What you'll learn

- Loading API keys safely from environment variables and `.env` files
- Making synchronous and async HTTP calls with `httpx`
- Parsing and streaming JSON responses
- Applying timeouts, retries, and exponential backoff
- Calling a local Ollama model (no API key required) and a hosted model

---

## Secrets: environment variables and .env

Never write an API key directly in your code. If you do, it ends up in version control, and that is a real security risk. Instead, load keys from environment variables at runtime — your code reads the value from the environment, not from the file itself.

```python
import os

api_key = os.environ["MY_API_KEY"]   # raises KeyError if missing — fast fail
base_url = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
```

For local development, store keys in a `.env` file (which you add to `.gitignore`) and use `python-dotenv` to load it.

```bash
uv add python-dotenv
```

```python
from dotenv import load_dotenv
import os

load_dotenv()   # reads .env in the current directory
api_key = os.environ["MY_API_KEY"]
```

Your `.env` file looks like this — it is plain text, never committed to git.

```text
# .env  (never commit this file)
MY_API_KEY=sk-...
LLM_BASE_URL=https://api.openai.com/v1
```

!!! danger "Never commit secrets"
    Add `.env` to `.gitignore` immediately. For production deployments, use a secrets manager (Vault, AWS Secrets Manager, or environment injection in CI) rather than a file.

---

## Basic HTTP call with httpx

`httpx` is a modern HTTP client that works for both regular (synchronous) and async Python. Here is a function that sends a prompt to a hosted model and returns the text reply.

```python
import os
import httpx

def chat(prompt: str) -> str:
    api_key = os.environ["MY_API_KEY"]
    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
    }
    with httpx.Client(timeout=30.0) as client:
        response = client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload,
        )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]
```

`response.raise_for_status()` turns a failed HTTP status (like 401 Unauthorized or 429 Too Many Requests) into a Python exception, so you hear about it immediately rather than silently getting bad data.

---

## Local Ollama call — no API key

Ollama runs models on your own machine and exposes a REST API on `localhost`. No account, no key, no cost per token.

```python
import httpx

def ollama_chat(prompt: str, model: str = "llama3") -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }
    with httpx.Client(timeout=60.0) as client:
        response = client.post(
            "http://localhost:11434/api/chat",
            json=payload,
        )
    response.raise_for_status()
    return response.json()["message"]["content"]

print(ollama_chat("Explain RAG in one sentence."))
```

The response parsing (`response.json()["message"]["content"]`) differs slightly from the hosted API format — that is just a difference in how Ollama structures its JSON.

!!! note "Ollama must be running"
    Start it with `ollama serve` and pull a model with `ollama pull llama3` before calling this function. Once it is running, it stays running in the background.

---

## Streaming tokens

Instead of waiting for the full reply before displaying anything, you can stream the response — printing each token as the model generates it. This makes long answers feel much more responsive.

```python
import os
import httpx
import json

def stream_chat(prompt: str) -> None:
    api_key = os.environ["MY_API_KEY"]
    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
    }
    with httpx.Client(timeout=60.0) as client:
        with client.stream(
            "POST",
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    chunk = json.loads(line[6:])
                    delta = chunk["choices"][0]["delta"].get("content", "")
                    print(delta, end="", flush=True)
    print()
```

Each `line` is a small JSON object containing the next piece of the reply. The `end=""` and `flush=True` in `print` make the output appear character-by-character rather than all at once.

---

## Timeouts, retries, and backoff

LLM APIs are slower than most web services, and they occasionally return errors — especially `429 Too Many Requests` when you send too many calls at once. Building in retries with a short wait between each attempt means your code recovers automatically from these blips.

```python
import os
import time
import httpx

def chat_with_retry(prompt: str, max_retries: int = 3) -> str:
    api_key = os.environ["MY_API_KEY"]
    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
    }
    for attempt in range(max_retries):
        try:
            with httpx.Client(timeout=httpx.Timeout(connect=5.0, read=60.0, write=10.0, pool=5.0)) as client:
                response = client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json=payload,
                )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 429:
                wait = 2 ** attempt          # exponential backoff: 1s, 2s, 4s
                print(f"Rate limited. Waiting {wait}s...")
                time.sleep(wait)
            else:
                raise
        except httpx.TimeoutException:
            if attempt == max_retries - 1:
                raise
            time.sleep(1)
    raise RuntimeError("Max retries exceeded")
```

The wait time doubles on each attempt (`1s`, `2s`, `4s`). This is called **exponential backoff** and it gives the server time to recover without hammering it with immediate retries.

!!! tip "Use tenacity for production retry logic"
    The `tenacity` library (`uv add tenacity`) provides decorators for retries with jitter, circuit breaking, and logging — cleaner than hand-rolled loops once your codebase grows.

!!! example "Try it yourself"
    Copy the `ollama_chat` function (no API key needed) and call it with a short prompt. Then try setting `timeout=0.001` to force a timeout error and watch the exception. Raise it back to `60.0` and add a simple retry loop around it — you have just written your first resilient API call.

??? note "Going deeper (optional)"
    **Async version**: replace `httpx.Client` with `httpx.AsyncClient` and `def` with `async def` to get an async version of any of these functions. See the [Async Python](async.md) page for the full pattern.

    **Structured output**: many hosted APIs support a `response_format` parameter that forces the model to reply in JSON conforming to a schema you provide. Combine this with a Pydantic model for fully validated, typed replies — see [Typing and Pydantic](typing-and-pydantic.md) for how.

    **Token limits**: each model has a context window (maximum total tokens for prompt + reply). If your prompt is too long, the API returns a `400` error. Tools like `tiktoken` let you count tokens before sending so you can trim the prompt if needed.

---

## Next steps

- [Hosted SDK overview](../../sdks/hosted-sdks.md) — provider-specific SDKs (Anthropic, OpenAI) that handle auth and retries for you
- [Ollama SDK](../../sdks/ollama-sdk.md) — the official Python SDK for local Ollama models with streaming and async support
