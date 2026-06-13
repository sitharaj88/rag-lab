# Functions and Modules

A function is like a recipe you can reuse. You write the steps once, give them a name, and then call that name whenever you need those steps done. A module is just a file full of recipes — so you can share them across your whole project.

## What you'll learn

- How to define functions with arguments, defaults, and return values
- The difference between positional and keyword arguments (`*args`, `**kwargs`)
- How to write docstrings
- How to import from the standard library and your own modules
- What `if __name__ == "__main__"` does
- How pip-installed packages fit into the picture

## Defining Functions

This defines a function called `greet` that takes a name and returns a greeting:

```python
def greet(name):
    """Return a greeting string for the given name."""
    return f"Hello, {name}!"

message = greet("Alice")
print(message)  # Hello, Alice!
```

- `def` is the keyword that starts a function definition.
- The body is indented underneath.
- `return` sends a value back to whoever called the function.
- If you leave out `return`, the function quietly returns `None`.

## Default Arguments

You can give arguments a default value. Then callers can leave those arguments out if the default is good enough for them:

```python
def chunk_text(text, max_length=200, overlap=20):
    """Split text into overlapping chunks of max_length characters."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_length
        chunks.append(text[start:end])
        start += max_length - overlap
    return chunks

# Uses both defaults
chunks = chunk_text("A very long document string...")

# Override just one default
chunks = chunk_text("A very long document string...", max_length=100)
```

!!! tip "One common trap: mutable default arguments"
    Never use a list or dictionary as a default value — Python creates that object once and reuses it forever, which causes surprising bugs. Use `None` as the default and create the object inside the function instead.

    ```python
    # This will cause bugs — avoid it
    def add_item(item, container=[]):
        container.append(item)
        return container

    # This is correct
    def add_item(item, container=None):
        if container is None:
            container = []
        container.append(item)
        return container
    ```

## `*args` and `**kwargs`

Sometimes you do not know in advance how many arguments someone will pass. `*args` collects any number of positional arguments into a tuple. `**kwargs` collects any number of keyword arguments into a dictionary:

```python
def log(*messages):
    """Accept any number of messages and print them all."""
    for msg in messages:
        print(f"[LOG] {msg}")

log("Started", "Loading document", "Done")
# [LOG] Started
# [LOG] Loading document
# [LOG] Done

def create_metadata(**fields):
    """Accept any keyword arguments and return them as a dict."""
    return dict(fields)

meta = create_metadata(source="handbook.pdf", page=3, author="Alice")
```

## Docstrings

A docstring is a short description of what a function does, written as a string right at the top of the function body. It is how Python knows what to show when someone types `help(your_function)`:

```python
def embed_text(text: str, model: str = "all-minilm") -> list[float]:
    """
    Return a vector embedding for the given text.

    Args:
        text: The input string to embed.
        model: Name of the embedding model to use.

    Returns:
        A list of floats representing the embedding vector.
    """
    ...  # implementation goes here
```

The `: str` and `-> list[float]` parts are optional **type hints** — they tell you and your editor what types are expected, but Python does not enforce them at runtime.

## Importing from the Standard Library

Python ships with a huge collection of ready-made modules called the **standard library**. You do not need to install anything — just `import` what you need:

```python
import os
import json
from pathlib import Path
from collections import defaultdict

# os — read environment variables and work with file paths
home = os.environ.get("HOME", "/tmp")

# json — turn Python objects into JSON text, and back again
data = json.loads('{"key": "value"}')
text = json.dumps(data, indent=2)

# pathlib — a modern way to work with file paths on any operating system
p = Path("docs") / "handbook.pdf"
print(p.suffix)   # .pdf
```

## Importing from Your Own Modules

Any `.py` file you create is a module that other files can import from. Here is an example with two files in the same folder:

```python
# utils.py
def clean_text(text):
    """Strip extra whitespace and normalise newlines."""
    return " ".join(text.split())
```

```python
# main.py
from utils import clean_text

raw = "  Some   messy   text\n\n"
print(clean_text(raw))  # Some messy text
```

Running `python main.py` will print `Some messy text`.

!!! tip "Watch out for naming conflicts"
    Python looks in the current folder first, then in installed packages, then in the standard library. If you name one of your files `json.py`, it will shadow the built-in `json` module and cause confusing errors. Avoid naming your files after standard library modules.

## `if __name__ == "__main__"`

When Python runs a file directly, it sets a special variable called `__name__` to `"__main__"`. When the same file is imported by another file, `__name__` is set to the module's name instead. This guard lets you put "run this file directly" code at the bottom without it running when the file is just being imported:

```python
# pipeline.py

def run_pipeline(query):
    print(f"Running pipeline for: {query}")

if __name__ == "__main__":
    run_pipeline("What is RAG?")
```

```bash
python pipeline.py          # prints: Running pipeline for: What is RAG?
python -c "import pipeline" # prints nothing — the guard blocks it
```

!!! example "Try it yourself"
    Create a file called `greetings.py` with a `greet()` function and an `if __name__ == "__main__"` block that calls it. Then create a second file that imports `greet` from `greetings`. Run each file and notice which one prints the greeting and which one does not.

## Pip-installed Packages

After activating your virtual environment, you can install third-party packages — tools the community has built and shared. Once installed, you import them the same way you import the standard library:

```bash
pip install sentence-transformers    # classic
uv pip install sentence-transformers # uv
```

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
embedding = model.encode("Hello, RAG!")
```

## Next steps

- [OOP](oop.md) — organise related functions and data into classes
- [Packaging](../ai-python/packaging.md) — turn your code into a distributable package
