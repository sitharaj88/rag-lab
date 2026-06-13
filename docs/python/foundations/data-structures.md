# Data Structures

When you go grocery shopping, you do not just carry everything in your hands — you use a bag, a list, maybe separate bags for fragile items. Python has its own ways of organising data. This page covers the four you will reach for most often.

## What you'll learn

- How to create and use `list`, `tuple`, `dict`, and `set`
- Indexing, slicing, and mutation
- Iteration patterns
- List and dict comprehensions
- When to reach for each structure

## Lists

A list is like a shopping list — an ordered collection of items that you can add to, remove from, and reorder. (The programming word for this is a **mutable sequence**.) Items stay in the order you put them in.

This example creates a list of text chunks from a document:

```python
chunks = [
    "The capital of France is Paris.",
    "Paris has a population of over 2 million.",
    "The Eiffel Tower is located in Paris.",
]

print(chunks[0])   # First chunk — counting starts at 0
print(chunks[-1])  # Last chunk — negative numbers count from the end
print(chunks[1:3]) # A slice: items at positions 1 and 2
```

You will see `The capital of France is Paris.` printed first, then the last chunk, then a new list containing items 1 and 2.

**Adding, inserting, and removing items:**

```python
chunks.append("Paris was founded in 250 BC.")  # Add to the end
chunks.insert(0, "Introduction sentence.")      # Insert at position 0
chunks.remove("Introduction sentence.")         # Remove a specific item
popped = chunks.pop()                           # Remove and return the last item
print(len(chunks))                              # How many items are in the list
```

**Sorting a list:**

```python
scores = [0.82, 0.95, 0.61, 0.88]
scores.sort(reverse=True)   # Sort in place, highest first
ranked = sorted(scores)     # Create a new sorted list (original unchanged)
```

## Tuples

A tuple is like a list, but you cannot change it once it is created. (The programming word for this is **immutable**.) Use a tuple when the data is fixed — like a pair of coordinates that should never be swapped around.

```python
chunk_ref = ("doc_42", 7)      # A (document_id, chunk_index) pair
doc_id, chunk_idx = chunk_ref  # Unpacking: assign both values at once

print(doc_id)    # doc_42
print(chunk_idx) # 7
```

!!! tip "Tuple or list — how do I choose?"
    If the collection will grow or change, use a list. If it represents a fixed record — like a coordinate, a colour value, or a function returning two things at once — use a tuple. The locked-down nature of a tuple communicates intent: "this data is not meant to be changed."

## Dictionaries

A dictionary works exactly like a real dictionary: you look up a word (the **key**) and get back its meaning (the **value**). Use a dictionary whenever you need to store named fields or look things up by a label.

This example stores metadata about a document chunk:

```python
chunk_meta = {
    "source": "docs/handbook.pdf",
    "page": 3,
    "chunk_index": 1,
    "score": 0.92,
}

print(chunk_meta["source"])                    # Access by key
print(chunk_meta.get("author", "Unknown"))     # Safe access: returns "Unknown" if key is missing

chunk_meta["token_count"] = 128  # Add a new key
del chunk_meta["score"]          # Remove a key

print(chunk_meta.keys())    # All the keys
print(chunk_meta.values())  # All the values
print(chunk_meta.items())   # All the (key, value) pairs
```

**Looping through a dictionary:**

```python
for key, value in chunk_meta.items():
    print(f"  {key}: {value}")
```

You will see each key and value printed on its own line.

## Sets

A set is an unordered collection where every item must be unique — duplicates are automatically dropped. Think of it as a bag where identical items merge into one. Sets are great for deduplication and for quickly checking whether something is in the collection.

```python
seen_sources = set()

sources = ["handbook.pdf", "faq.pdf", "handbook.pdf", "guide.pdf"]
for src in sources:
    seen_sources.add(src)

print(seen_sources)                # {'handbook.pdf', 'faq.pdf', 'guide.pdf'}
print("faq.pdf" in seen_sources)   # True — checking membership is very fast
```

Notice that `"handbook.pdf"` only appears once even though we added it twice.

## List Comprehensions

A list comprehension is a compact way to build a new list from an existing one — all in a single readable line. You will see this pattern everywhere in Python code.

Here is the same operation written two ways:

```python
# Classic loop — works perfectly, a bit longer
upper_chunks = []
for chunk in chunks:
    upper_chunks.append(chunk.upper())

# List comprehension — shorter and just as clear once you get used to it
upper_chunks = [chunk.upper() for chunk in chunks]

# You can add a filter with if at the end
long_chunks = [c for c in chunks if len(c) > 50]
```

!!! tip "Read it like English"
    A list comprehension reads almost like a sentence: "give me `chunk.upper()` for each `chunk` in `chunks`." Once you spot this pattern it becomes very natural.

## Dict Comprehensions

Dict comprehensions work the same way, but build a dictionary instead of a list:

```python
# Build a dictionary: each chunk mapped to its character length
chunk_lengths = {chunk: len(chunk) for chunk in chunks}

# Keep only the short ones
short = {k: v for k, v in chunk_lengths.items() if v < 60}
```

!!! example "Try it yourself"
    This snippet builds a lookup table — a dictionary that lets you find any chunk instantly by its position number:

    ```python
    chunks = [
        "The capital of France is Paris.",
        "Paris has a population of over 2 million.",
        "The Eiffel Tower is located in Paris.",
    ]

    # Build the lookup table
    chunk_index = {i: text for i, text in enumerate(chunks)}

    # Retrieve chunk number 0
    print(chunk_index[0])
    # The capital of France is Paris.
    ```

    Try changing the number inside `chunk_index[...]` to `1` or `2` and see what you get.

## Quick Reference: When to Use Which

| Need                           | Use       |
|--------------------------------|-----------|
| Ordered, growable collection   | `list`    |
| Fixed, positional record       | `tuple`   |
| Key → value lookup / metadata  | `dict`    |
| Unique items / membership test | `set`     |

## Next steps

- [Functions and Modules](functions-and-modules.md) — encapsulate logic that operates on these structures
- [NumPy](../data-ml/numpy.md) — array operations on numerical data (embeddings live here)
