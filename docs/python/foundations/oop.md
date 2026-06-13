# Object-Oriented Programming

Imagine a cookie cutter. You design the cutter once — its shape, its size — and then you can stamp out as many cookies as you want, each one a separate object. In Python, a **class** is the cookie cutter, and the cookies it produces are called **instances** (or **objects**). This page will show you how to make your own.

## What you'll learn

- How to define a class with `__init__`, attributes, and methods
- The difference between instance and class attributes
- Inheritance basics
- Why `@dataclass` is the practical default for data containers
- How a `Chunk` dataclass maps to real RAG pipeline objects

## Defining a Class

This example creates a class called `Document` to represent a source file loaded into a pipeline:

```python
class Document:
    """Represents a source document loaded into the pipeline."""

    def __init__(self, path: str, content: str):
        self.path = path            # saved onto this particular document
        self.content = content
        self.char_count = len(content)

    def preview(self, n: int = 100) -> str:
        """Return the first n characters of the document."""
        return self.content[:n]

    def __repr__(self) -> str:
        return f"Document(path={self.path!r}, chars={self.char_count})"


doc = Document("handbook.pdf", "This is the full content of the handbook...")
print(doc.preview(20))   # This is the full con
print(doc)               # Document(path='handbook.pdf', chars=44)
```

A few things to notice:

- `__init__` is called automatically when you create a new instance with `Document(...)`. Think of it as the "setup" step.
- `self` refers to the specific instance being created or used. Every method receives it as the first argument.
- `__repr__` controls what you see when you `print()` an object.

## Instance vs Class Attributes

An **instance attribute** (like `self.model_name` below) belongs to one specific object. A **class attribute** (like `default_model`) is shared by all objects created from that class:

```python
class EmbeddingModel:
    default_model = "all-MiniLM-L6-v2"   # shared by every instance

    def __init__(self, model_name: str = None):
        # unique to each instance — falls back to the class-level default
        self.model_name = model_name or EmbeddingModel.default_model
```

## Methods

A method is a function that lives inside a class and always has access to the instance via `self`. Here is a `TextChunker` class that uses its settings to split text:

```python
class TextChunker:
    def __init__(self, max_length: int = 200, overlap: int = 20):
        self.max_length = max_length
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        """Split text into overlapping chunks."""
        results = []
        start = 0
        while start < len(text):
            results.append(text[start : start + self.max_length])
            start += self.max_length - self.overlap
        return results

chunker = TextChunker(max_length=100, overlap=10)
chunks = chunker.chunk("A long document body here...")
```

Create the chunker once, then call `.chunk()` on it as many times as you need.

## Inheritance

Inheritance lets one class build on top of another. The child class gets everything the parent class has, and can add or replace whatever it needs:

```python
class BaseLoader:
    def load(self, path: str) -> str:
        raise NotImplementedError("Subclasses must implement load()")

class TextFileLoader(BaseLoader):
    def load(self, path: str) -> str:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

class MarkdownLoader(TextFileLoader):
    def load(self, path: str) -> str:
        raw = super().load(path)   # call the parent's method first
        # Strip markdown heading lines before embedding
        return "\n".join(
            line for line in raw.splitlines() if not line.startswith("#")
        )
```

`MarkdownLoader` reuses `TextFileLoader`'s file-reading code via `super()`, then adds its own step on top.

!!! tip "Keep inheritance shallow"
    More than two levels of inheritance usually makes code harder to follow. If you find yourself going three or four levels deep, it is worth stepping back and thinking of a simpler structure. Passing objects as arguments to other objects (the programming term is **composition**) is often cleaner.

## Dataclasses — The Practical Default

For classes that mainly hold data — like a chunk of text with some metadata — Python's `@dataclass` decorator does the boring boilerplate for you. It automatically generates `__init__`, `__repr__`, and `__eq__` based on the fields you declare:

```python
from dataclasses import dataclass, field


@dataclass
class Chunk:
    """A single chunk of text with its source metadata."""

    text: str
    source: str
    page: int = 0
    chunk_index: int = 0
    embedding: list[float] = field(default_factory=list)

    def char_count(self) -> int:
        return len(self.text)


c = Chunk(
    text="The capital of France is Paris.",
    source="geography.pdf",
    page=1,
    chunk_index=0,
)

print(c)
# Chunk(text='The capital of France is Paris.', source='geography.pdf', page=1, ...)
print(c.char_count())  # 31
```

!!! tip "Dataclasses are your friend in RAG projects"
    Objects like `Chunk`, `RetrievedResult`, and `GenerationInput` are perfect for dataclasses. You get a clean, readable definition with almost no extra code. You can also add methods to them just like a regular class.

!!! example "Try it yourself"
    A **frozen** dataclass cannot be changed after creation. That makes it safe to use as a dictionary key or in a set, because Python knows it will never accidentally change:

    ```python
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class ChunkID:
        source: str
        index: int

    # A frozen dataclass can be used inside a set
    seen = {ChunkID("doc.pdf", 0), ChunkID("doc.pdf", 1)}
    print(len(seen))  # 2
    ```

    Try creating a few `ChunkID` objects. Add duplicates to the set and watch the set automatically keep only unique ones.

## Next steps

- [Files and Errors](files-and-errors.md) — read documents from disk and handle failures gracefully
- [Typing and Pydantic](../ai-python/typing-and-pydantic.md) — add validation and strict schemas to your data models
