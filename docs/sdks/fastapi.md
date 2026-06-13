# FastAPI — Serving RAG as a JSON API

FastAPI is a modern Python web framework for building high-performance REST APIs. Paired with a RAG pipeline, it lets you expose retrieval-augmented generation as a production-grade endpoint that any client can call.

## What you'll learn

- How to define Pydantic v2 request and response models for a RAG query
- How to write an async `/query` POST endpoint that wraps a RAG pipeline
- How to stream token-by-token output with `StreamingResponse`
- How to run and reload the server with Uvicorn
- How to structure the app for later containerisation

## Install

```bash
pip install fastapi uvicorn pydantic
```

!!! warning "Pydantic v2 required"
    FastAPI 0.100+ dropped Pydantic v1 support. Use `model_dump()`, not `.dict()`, and import `Field` from `pydantic` directly. If you see `PydanticUserError` or a deprecation warning about `.dict()`, your environment has Pydantic v1 — upgrade with `pip install --upgrade pydantic`.

For a fuller data-validation overview see [Typing and Pydantic](../python/ai-python/typing-and-pydantic.md).

## Pydantic v2 models

Define the request and response shapes before writing any route logic.

```python
from pydantic import BaseModel, Field
from typing import List

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The question to answer")
    top_k: int = Field(default=4, ge=1, le=20, description="Number of chunks to retrieve")

class SourceDocument(BaseModel):
    content: str
    source: str
    score: float

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceDocument]
    model: str
```

!!! note "model_dump() not .dict()"
    Pydantic v2 replaced `.dict()` with `.model_dump()` and `.json()` with `.model_dump_json()`. Using the old methods raises a deprecation warning and will eventually error.

```python
# Pydantic v2 serialisation
response = QueryResponse(answer="Paris", sources=[], model="llama3")
data = response.model_dump()          # dict
json_str = response.model_dump_json() # JSON string
```

## The RAG pipeline wrapper

Keep pipeline initialisation outside request handlers so the model loads once at startup.

```python
from fastapi import FastAPI
from contextlib import asynccontextmanager

# Replace with your actual pipeline import
# from my_pipeline import RAGPipeline

_pipeline = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline
    # _pipeline = RAGPipeline.load("./index")  # load once
    yield
    # cleanup goes here if needed

app = FastAPI(title="RAG API", lifespan=lifespan)
```

!!! tip "Use @st.cache_resource in Streamlit"
    The pattern above is the FastAPI equivalent of `@st.cache_resource` in Streamlit — one load, many requests. See [Streamlit](streamlit.md) for the UI-facing counterpart.

## The /query endpoint

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List

app = FastAPI(title="RAG API")

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=4, ge=1, le=20)

class SourceDocument(BaseModel):
    content: str
    source: str
    score: float

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceDocument]
    model: str

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    """Run a RAG query and return the answer with source documents."""
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialised")

    try:
        # result = await _pipeline.aquery(request.question, top_k=request.top_k)
        # Placeholder until pipeline is wired up:
        result = {
            "answer": f"Answer to: {request.question}",
            "sources": [{"content": "chunk text", "source": "doc.pdf", "score": 0.91}],
            "model": "llama3",
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return QueryResponse(
        answer=result["answer"],
        sources=[SourceDocument(**s) for s in result["sources"]],
        model=result["model"],
    )
```

!!! info "Async endpoints"
    Declare endpoint functions with `async def` so FastAPI can handle other requests while your pipeline awaits I/O. If your retriever or LLM client exposes only synchronous APIs, wrap the call in `asyncio.to_thread()` to avoid blocking the event loop.

    ```python
    import asyncio

    @app.post("/query", response_model=QueryResponse)
    async def query(request: QueryRequest) -> QueryResponse:
        result = await asyncio.to_thread(sync_pipeline.query, request.question)
        ...
    ```

## Streaming with StreamingResponse

For long-form answers, stream tokens back as server-sent events so the client sees output incrementally.

```python
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator

async def token_stream(question: str) -> AsyncGenerator[str, None]:
    """Yield tokens from the pipeline one at a time."""
    # async for token in _pipeline.astream(question):
    #     yield f"data: {token}\n\n"
    for word in ["Paris ", "is ", "the ", "capital ", "of ", "France."]:
        yield f"data: {word}\n\n"

@app.post("/query/stream")
async def query_stream(request: QueryRequest) -> StreamingResponse:
    return StreamingResponse(
        token_stream(request.question),
        media_type="text/event-stream",
    )
```

Clients consume the stream with `EventSource` in JavaScript or `httpx`'s streaming interface in Python.

## Health check

```python
@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "pipeline_ready": _pipeline is not None}
```

## Running the server

```bash
uvicorn app:app --reload --port 8000
```

- `app:app` — module `app`, object `app`
- `--reload` — hot-reload on file changes (development only)
- `--port 8000` — default FastAPI port

For production, drop `--reload` and increase workers:

```bash
uvicorn app:app --workers 4 --port 8000
```

## Interactive docs

FastAPI auto-generates OpenAPI documentation. With the server running, open:

- `http://localhost:8000/docs` — Swagger UI (try the endpoint in-browser)
- `http://localhost:8000/redoc` — ReDoc

## Full minimal example

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List

app = FastAPI(title="RAG API")

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=4, ge=1, le=20)

class SourceDocument(BaseModel):
    content: str
    source: str
    score: float

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceDocument]
    model: str

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    # Wire your pipeline here
    raise HTTPException(status_code=501, detail="Pipeline not wired up yet")

@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
```

```bash
uvicorn app:app --reload
```

## Next steps

- [Docker](docker.md) — containerise this API alongside a vector database
- [Typing and Pydantic](../python/ai-python/typing-and-pydantic.md) — deeper Pydantic v2 patterns
- [Production](../advanced/production.md) — authentication, rate limiting, and deployment hardening
