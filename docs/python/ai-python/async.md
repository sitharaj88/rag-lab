# Async Python for LLM Applications

Imagine making breakfast. You could boil an egg, then make toast, then brew coffee — waiting for each to finish before starting the next. Or you could put on the kettle, drop the bread in the toaster, and crack the egg, all at the same time. The second approach isn't magic — it's just smarter use of your time while each thing is running on its own.

Python's `async` works the same way. When your code sends a request to an LLM API and waits for the reply, the CPU is doing nothing. The technical word for this is **IO-bound** work. `asyncio` lets Python do other things during that wait — like sending five more requests — so they all finish much sooner than if you did them one by one.

## What you'll learn

- How `async def` / `await` and the event loop work
- Running async code with `asyncio.run` and `asyncio.gather`
- Making concurrent HTTP calls with `httpx.AsyncClient`
- Fanning out many embedding calls in parallel
- Common pitfalls (awaiting inside a loop, blocking calls)

---

## The basics

An `async def` function is a **coroutine** — a function that can pause itself while waiting for something (like a network reply) and let other coroutines run in the meantime. You pause it with `await`.

This example shows the simplest possible async program — a function that pretends to do some IO, and a `main` that runs it.

```python
import asyncio

async def fetch_answer(question: str) -> str:
    await asyncio.sleep(0.1)   # simulate IO
    return f"Answer to: {question}"

async def main() -> None:
    result = await fetch_answer("What is RAG?")
    print(result)

asyncio.run(main())
```

`asyncio.run` creates an event loop, runs `main()` to completion, and closes the loop. Call it exactly once at the top level of your program.

---

## Concurrent calls with asyncio.gather

`asyncio.gather` starts several coroutines at the same time and returns all their results in order once they are all done. This is the main pattern for sending multiple embedding or LLM requests in parallel.

The code below embeds three text chunks at once instead of one at a time. All three requests are in flight simultaneously, so the total time is roughly the time for one request, not three.

```python
import asyncio
import httpx

OLLAMA_URL = "http://localhost:11434/api/embeddings"

async def embed_one(client: httpx.AsyncClient, text: str) -> list[float]:
    response = await client.post(
        OLLAMA_URL,
        json={"model": "nomic-embed-text", "prompt": text},
        timeout=30.0,
    )
    response.raise_for_status()
    return response.json()["embedding"]

async def embed_many(texts: list[str]) -> list[list[float]]:
    async with httpx.AsyncClient() as client:
        tasks = [embed_one(client, t) for t in texts]
        embeddings = await asyncio.gather(*tasks)
    return list(embeddings)

if __name__ == "__main__":
    chunks = [
        "Paris is the capital of France.",
        "The Eiffel Tower stands 330 metres tall.",
        "RAG combines retrieval with generation.",
    ]
    results = asyncio.run(embed_many(chunks))
    print(f"Embedded {len(results)} chunks, dim={len(results[0])}")
```

You get back a list of embedding vectors in the same order as your input texts.

!!! tip "Reuse a single client"
    Create `httpx.AsyncClient` once per batch and share it across all tasks. Each client manages a connection pool, so creating one per request is slow and wasteful. The `async with` block above does exactly this — one client, many requests.

---

## Pitfall: awaiting inside a loop

This is the most common mistake when learning async. If you `await` each call inside a `for` loop, you get zero concurrency — each call waits for the previous one to finish, which is the same as not using async at all.

```python
# SLOW — each call waits for the previous one to finish
async def embed_sequential(texts: list[str]) -> list[list[float]]:
    async with httpx.AsyncClient() as client:
        results = []
        for t in texts:
            emb = await embed_one(client, t)   # blocks here
            results.append(emb)
    return results
```

Replace the loop with `asyncio.gather` (or `asyncio.TaskGroup` on Python 3.11+) to fire them all off together.

```python
async def embed_parallel(texts: list[str]) -> list[list[float]]:
    async with httpx.AsyncClient() as client:
        async with asyncio.TaskGroup() as tg:
            tasks = [tg.create_task(embed_one(client, t)) for t in texts]
    return [t.result() for t in tasks]
```

`TaskGroup` gives you cleaner error handling: if any task fails, the others are cancelled and the exception surfaces immediately.

---

## Pitfall: blocking calls inside async functions

Synchronous (blocking) operations — like reading a big file, doing heavy CPU work, or calling `time.sleep` — freeze the entire event loop while they run. Every other coroutine has to wait.

When you need to do something blocking inside an async function, push it to a background thread with `run_in_executor`.

```python
import asyncio

async def load_file(path: str) -> str:
    # run_in_executor pushes the blocking call to a thread pool
    loop = asyncio.get_running_loop()
    content = await loop.run_in_executor(None, open(path).read)
    return content
```

This keeps the event loop free while the file is being read.

!!! warning "Never mix sync and async carelessly"
    Calling a blocking HTTP library like `requests` inside an `async def` will stall every other coroutine waiting on the event loop. Use `httpx.AsyncClient` for async HTTP instead.

---

## Handling errors in gather

By default `asyncio.gather` raises immediately if any task fails, and cancels the rest. Pass `return_exceptions=True` to collect all results — successes and errors — and handle them individually.

```python
results = await asyncio.gather(*tasks, return_exceptions=True)
for i, r in enumerate(results):
    if isinstance(r, Exception):
        print(f"Task {i} failed: {r}")
    else:
        print(f"Task {i} ok, dim={len(r)}")
```

This is useful when you are embedding a large batch and one chunk fails — you can log it and keep the others rather than losing the entire batch.

!!! example "Try it yourself"
    Take the `embed_many` function above and add a fake error: change one of the texts to an empty string and see what happens with and without `return_exceptions=True`. You'll see exactly how gather behaves differently in each mode.

??? note "Going deeper (optional)"
    **Rate limiting**: when you fan out hundreds of requests at once, you may hit the API's rate limit. A common pattern is to use `asyncio.Semaphore` to cap the number of in-flight requests. For example, `sem = asyncio.Semaphore(10)` and then `async with sem:` inside your task limits you to 10 concurrent requests at any time.

    **Async generators**: for streaming LLM responses token-by-token, you can write `async def stream(...): yield chunk` and consume it with `async for chunk in stream(...)`. This keeps memory low for long outputs.

---

## Next steps

- [Working with model APIs](working-with-model-apis.md) — apply async patterns to real LLM endpoint calls with timeouts and retries
- [Production RAG](../../advanced/production.md) — concurrency, batching, and rate-limit strategies at scale
