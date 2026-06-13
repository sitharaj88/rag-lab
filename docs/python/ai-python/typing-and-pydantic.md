# Typing and Pydantic

Think of type hints like labels on jars in your kitchen. You could store anything in any jar, but labeling them "flour", "sugar", and "salt" means you — and anyone else — instantly know what's inside. The technical word for this is **type hints**, and they help both you and your tools catch mistakes before they turn into confusing errors deep in your code.

Pydantic takes that idea further: it checks that what's actually *inside* the jar matches the label, at the moment your program runs.

## What you'll learn

- Adding type hints to variables, functions, and collections
- Using `Optional`, union types (`|`), and `TypedDict` for structured data
- Running static analysis with mypy or pyright
- Defining Pydantic v2 `BaseModel` classes with `Field` validators
- Parsing and validating JSON responses from LLM APIs

---

## Type hints

Type hints are notes you add to your code to say what kind of value a variable or function should hold. Python itself doesn't enforce them at runtime, but your editor and static analysis tools do — they'll flag mismatches before you even run your code.

Here is how to annotate basic values and collections.

```python
# Basic variable annotations
name: str = "RAG Lab"
chunk_size: int = 512
temperature: float = 0.7
debug: bool = False

# Collections (Python 3.9+ lowercase generics)
tags: list[str] = ["rag", "llm"]
metadata: dict[str, str] = {"source": "wiki"}

# Optional — value may be None
from typing import Optional

def embed(text: str, model: Optional[str] = None) -> list[float]:
    model = model or "nomic-embed-text"
    ...
    return []
```

This tells you (and your editor) that `embed` takes a string and an optional string, and returns a list of floats.

From Python 3.10 onwards you can write `str | None` as a shorter form of `Optional[str]`.

```python
def chunk(text: str, size: int = 256) -> list[str] | None:
    if not text:
        return None
    return [text[i : i + size] for i in range(0, len(text), size)]
```

### TypedDict for lightweight structured dicts

When you want a regular dictionary but with known, named keys and no extra overhead, `TypedDict` is the right tool. It documents the shape of your data without adding validation logic.

```python
from typing import TypedDict

class Document(TypedDict):
    content: str
    source: str
    score: float
```

### Static analysis: mypy and pyright

These tools read your type hints and flag problems before you run the code. Add them as development dependencies and check your source folder.

```bash
uv add --dev mypy pyright
uv run mypy src/
uv run pyright src/
```

!!! tip "Why this matters for AI code"
    LLM APIs return raw `dict` or JSON. Without type hints you might pass the wrong field to an embedding call and get a confusing error ten steps later. Catching the mismatch at the type-checker level saves hours of debugging — these tools are on your side.

---

## Pydantic v2

Pydantic validates data at runtime against a schema you define as a Python class. It is the backbone of FastAPI's request/response models, and it is excellent for parsing and checking structured output from LLM APIs.

Here is a model for a RAG response that includes citations and a confidence score. Notice how `Field` lets you attach constraints right alongside the definition.

```python
from pydantic import BaseModel, Field, field_validator

class Citation(BaseModel):
    source: str
    page: int = Field(ge=1, description="1-indexed page number")
    snippet: str = Field(min_length=10)

class RAGResponse(BaseModel):
    answer: str
    citations: list[Citation]
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("answer")
    @classmethod
    def answer_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("answer must not be blank")
        return v
```

When you create a `RAGResponse`, Pydantic checks every field automatically — you get a clear error message instead of a silent bad value.

### Parsing LLM JSON output

When you ask a model to reply in JSON, use `model_validate_json` to parse and validate in one step. This line gives you either a typed `RAGResponse` object or a helpful error explaining exactly what was wrong.

```python
import os
import httpx
from pydantic import ValidationError

raw_json = '{"answer": "Paris", "citations": [{"source": "wiki", "page": 1, "snippet": "Paris is the capital of France."}], "confidence": 0.95}'

try:
    response = RAGResponse.model_validate_json(raw_json)
    print(response.answer)
    print(response.model_dump())   # returns a plain dict — not .dict()
except ValidationError as exc:
    print(exc.errors())
```

You get back a fully typed object — `response.answer` is a string, `response.citations` is a list of `Citation` objects, and so on.

!!! warning "Pydantic v2 API"
    Use `model_dump()` and `model_validate()` / `model_validate_json()`. The old `.dict()` and `.parse_raw()` methods are removed in v2.

### Round-tripping through a dict

You can go from a plain dict → typed object → plain dict (or JSON string) at any point. This is handy when you need to store, transmit, or log a response.

```python
data = {"answer": "Berlin", "citations": [], "confidence": 0.5}
obj = RAGResponse.model_validate(data)
back = obj.model_dump()          # {"answer": "Berlin", ...}
back_json = obj.model_dump_json() # compact JSON string
```

!!! example "Try it yourself"
    Copy the `RAGResponse` class above into a Python file. Then try passing it a dict that has a `confidence` of `1.5` or an empty `answer`. Read the `ValidationError` message — it tells you exactly which field failed and why. This is Pydantic doing its job.

??? note "Going deeper (optional)"
    `TypedDict` is lighter than Pydantic — no runtime validation, no extra dependency — so it is a good fit for internal functions where you trust the data shape. Pydantic shines at boundaries: when data comes in from outside (an API, a file, user input) and you need to be sure it is correct before your code touches it. You can also combine them: use `TypedDict` inside your pipeline and convert to a Pydantic model only at the entry and exit points.

---

## Next steps

- [Working with model APIs](working-with-model-apis.md) — make typed HTTP calls to LLM endpoints and validate the responses with these models
- [FastAPI SDK](../../sdks/fastapi.md) — build a typed REST layer around your RAG pipeline using these same Pydantic models
