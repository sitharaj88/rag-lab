# Files and Errors

Every RAG pipeline begins by reading documents from your computer. Think of this page as learning two things at once: how to open and read files (like opening a book), and how to handle it gracefully when something goes wrong (like Python politely telling you the book is missing, and what to do next).

## What you'll learn

- Reading and writing text files with `open()` and the `with` statement
- Modern path handling with `pathlib.Path`
- Loading and saving JSON data
- Handling exceptions with `try` / `except` / `finally`
- Raising your own exceptions
- The most common errors you will encounter

## Reading Files with `with`

The `with` statement is the safe way to open a file. It guarantees the file will be closed when you are done — even if something goes wrong in the middle:

```python
# Read an entire text file into a string
with open("handbook.txt", "r", encoding="utf-8") as f:
    content = f.read()

print(content[:200])  # Print the first 200 characters
```

After this runs, `content` holds all the text from the file as one big string.

**Read a large file line by line (uses less memory):**

```python
lines = []
with open("handbook.txt", "r", encoding="utf-8") as f:
    for line in f:
        lines.append(line.strip())
```

Each `line` comes with a newline character at the end — `.strip()` removes it.

**Write text to a file:**

```python
with open("output.txt", "w", encoding="utf-8") as f:
    f.write("Processed chunk 1\n")
    f.write("Processed chunk 2\n")
```

This creates `output.txt` (or overwrites it if it already exists) and writes two lines into it.

!!! tip "Always include `encoding=\"utf-8\"`"
    Without this, Python uses whatever encoding your operating system defaults to. On Windows that is often `cp1252`, which will silently mangle characters like accents and emoji. Writing `encoding="utf-8"` every time is a small habit that prevents very confusing bugs.

## Modern Paths with `pathlib`

String paths like `"docs\\handbook.pdf"` look different on Windows versus macOS and Linux. `pathlib.Path` handles all of that for you automatically — and it comes with handy extras:

```python
from pathlib import Path

docs_dir = Path("docs")
file_path = docs_dir / "handbook.pdf"   # the / operator joins path parts

print(file_path.name)    # handbook.pdf
print(file_path.stem)    # handbook
print(file_path.suffix)  # .pdf
print(file_path.parent)  # docs

# Check whether the file exists before opening it
if file_path.exists():
    print("File found!")

# Find every .txt file in a folder
for txt_file in docs_dir.glob("*.txt"):
    print(txt_file)

# Shortcut: read and write text without a with block
content = Path("readme.txt").read_text(encoding="utf-8")
Path("output.txt").write_text("Hello!", encoding="utf-8")
```

!!! tip "Prefer `pathlib` over string paths"
    `Path("a") / "b" / "c.txt"` works identically on Windows, macOS, and Linux. You never need to worry about backslashes versus forward slashes. It is also much easier to read.

## Reading and Writing JSON

JSON is a text format used everywhere for configuration files, API responses, and data storage. Python can convert between JSON text and Python dictionaries with just two functions:

```python
import json
from pathlib import Path

# Load JSON from a file — result is a Python dict or list
metadata_path = Path("chunk_metadata.json")
with metadata_path.open("r", encoding="utf-8") as f:
    metadata = json.load(f)

# Shorthand using pathlib
metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

# Save a Python object to a JSON file
chunks_data = [
    {"text": "Paris is the capital of France.", "page": 1},
    {"text": "The Eiffel Tower is 330 m tall.", "page": 2},
]
with open("chunks.json", "w", encoding="utf-8") as f:
    json.dump(chunks_data, f, indent=2, ensure_ascii=False)
```

`indent=2` makes the saved file nicely formatted and easy to read in a text editor.

## Exception Handling

When something goes wrong in Python, it raises an **exception** — Python's polite way of saying "I ran into a problem, here is what happened." You handle exceptions with `try` / `except`:

```python
def load_document(path: str) -> str:
    """Load a text document, returning an empty string on failure."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print(f"[WARN] File not found: {path}")
        return ""
    except PermissionError:
        print(f"[ERROR] No permission to read: {path}")
        return ""
    except Exception as e:
        print(f"[ERROR] Unexpected error reading {path}: {e}")
        raise   # re-raise anything unexpected — don't hide unknown problems
```

The `try` block runs your code. If an exception occurs, Python jumps to the matching `except` block. Code after the `except` block runs normally if nothing went wrong.

**`finally` runs no matter what — use it for cleanup:**

```python
import json

def parse_json_file(path: str):
    f = open(path, "r", encoding="utf-8")
    try:
        return json.load(f)
    except json.JSONDecodeError as e:
        print(f"Invalid JSON in {path}: {e}")
        return {}
    finally:
        f.close()   # this line always runs, even if an exception occurred
```

!!! tip "The `with` statement usually replaces `finally` for files"
    `with open(...)` handles closing automatically. You only need `finally` when you have custom cleanup beyond file closing — like disconnecting from a database or releasing a lock.

## Raising Your Own Exceptions

You can raise an exception yourself to signal that something is wrong with the inputs to your function. This makes bugs much easier to find:

```python
def chunk_document(text: str, max_length: int) -> list[str]:
    if not text:
        raise ValueError("text must not be empty")
    if max_length <= 0:
        raise ValueError(f"max_length must be positive, got {max_length}")
    # ... chunking logic continues ...
```

For larger projects, you can define your own exception types so callers can catch them specifically:

```python
class DocumentLoadError(Exception):
    """Raised when a document cannot be loaded into the pipeline."""
    pass

raise DocumentLoadError("Could not read handbook.pdf")
```

## Common Errors Quick Reference

You will run into these sooner or later. None of them mean you did something terrible — they are just Python telling you what happened:

| Exception              | When it happens                                |
|------------------------|------------------------------------------------|
| `FileNotFoundError`    | The path you gave does not exist               |
| `PermissionError`      | You do not have access to read or write that file |
| `IsADirectoryError`    | You passed a folder path where a file was expected |
| `json.JSONDecodeError` | The file exists but is not valid JSON          |
| `UnicodeDecodeError`   | The file has characters your encoding cannot handle |
| `ValueError`           | The type was right but the value was not acceptable |
| `TypeError`            | The wrong type was passed to a function        |
| `KeyError`             | You tried to look up a dictionary key that does not exist |
| `IndexError`           | You tried to access a list position that does not exist |

!!! example "Try it yourself"
    This function loads every `.txt` file from a folder and skips the ones it cannot read — which is exactly the kind of robust behaviour you need in a real pipeline:

    ```python
    from pathlib import Path

    def load_documents(folder: str) -> list[dict]:
        results = []
        for path in Path(folder).glob("*.txt"):
            try:
                text = path.read_text(encoding="utf-8")
                results.append({"source": str(path), "content": text})
            except Exception as e:
                print(f"Skipping {path}: {e}")
        return results
    ```

    Create a folder called `sample_docs`, put a couple of `.txt` files in it, then call `load_documents("sample_docs")`. Try also adding a file with a deliberately broken name to see the skip message appear.

## Next steps

- [Document Loaders](../../building-blocks/document-loaders.md) — higher-level abstractions for loading PDFs, HTML, and more into a RAG pipeline
- [Working with Model APIs](../ai-python/working-with-model-apis.md) — handle network errors and timeouts when calling LLM APIs
