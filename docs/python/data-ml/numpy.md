# NumPy for RAG

Imagine you have a long shopping list. Now imagine 10,000 shopping lists, and you need to add up the totals for all of them — fast. You *could* loop through every item one by one in plain Python, but that would take ages. NumPy is like hiring a team that handles all 10,000 lists at the same moment. It's the reason machine learning code runs in seconds instead of hours.

You don't need to be a math person to use NumPy. The library does the heavy lifting. Your job is just to know which tool to reach for.

## What you'll learn

- What a NumPy array is (hint: it's just a tidy row or grid of numbers)
- Why NumPy is so much faster than a normal Python list
- How to do simple math on whole lists of numbers at once — no loops needed
- The one idea that powers RAG similarity search: "how alike are two lists of numbers?"
- How to write the cosine-similarity calculation that vector databases use under the hood

---

## What is an array, really?

Think of a NumPy **array** like a row of labeled boxes — each box holds one number, and all the boxes are the same type. If a Python list is a junk drawer where anything goes, an array is a neat pill organizer where every slot is exactly the same size.

In plain words, an array is an ordered list of numbers stored so the computer can do math on all of them at once. (The fancy name you'll see in textbooks is **ndarray**, short for "n-dimensional array" — but "list of numbers" works just fine for now.)

Here's what this does, in human terms: we create a tiny array of four numbers, then peek at its shape and type.

```python
import numpy as np

# A single row of four numbers — like one entry in a spreadsheet column
v = np.array([0.1, 0.4, -0.2, 0.9], dtype=np.float64)

print(v.shape)   # (4,)  — four items in a single row
print(v.dtype)   # float64  — each number is a decimal
print(v.ndim)    # 1  — it's a flat, one-dimensional list
```

The output `(4,)` just means "four numbers in a line." That's it.

!!! note "NumPy 2.x heads-up"
    Older tutorials sometimes write `np.float_`. That was removed in NumPy 2.0 — you'll get an error if you use it. Just write `np.float64` (or plain Python `float`) instead. If you copy-paste old code and it breaks, this is the first thing to check.

---

## Rows, grids, and shapes

A 1-D array is a single row of numbers. A 2-D array is a grid — like a spreadsheet with rows and columns. In RAG, a grid of rows is a natural way to hold many document "fingerprints" (embeddings) at the same time.

Here's what this does, in human terms: we create a pretend batch of three documents, each described by four numbers, then grab the first one.

```python
# Three documents, each with four numbers — like a tiny spreadsheet
embeddings = np.random.randn(3, 4).astype(np.float64)

print(embeddings.shape)   # (3, 4) — 3 rows, 4 columns
print(embeddings[0])      # the first document's four numbers

# Sometimes you need to add an extra dimension for other libraries
single = embeddings[0].reshape(1, -1)   # turns (4,) into (1, 4)
```

Don't worry about `reshape` for now — it's just NumPy's way of re-packaging the same numbers into a different container shape.

---

## Doing math on the whole list at once

Here's the superpower: instead of looping through every number and multiplying one at a time, NumPy does it all in one line. This is called **vectorization** — which just means "apply this operation to every item at once."

Here's what this does, in human terms: we multiply each column of our three-document grid by a different weight — think of it as "some features matter more than others."

```python
weights = np.array([1.0, 0.5, 2.0, 0.8])

scaled = embeddings * weights   # applies weights to all 3 rows at once
```

No loop. No fuss. NumPy figures out the right way to pair up the numbers automatically. (The technical term for this magic is **broadcasting**, but you don't need to memorize that — just know it works.)

!!! tip
    If you ever find yourself writing a `for` loop to do math on arrays, stop and ask: "can NumPy do this in one line?" Usually the answer is yes, and it'll be 10–100× faster.

---

## The one idea that matters most for RAG: how similar are two lists?

Here's the core question in retrieval: *"Out of all my stored documents, which one is most similar to the user's question?"*

To answer that, we need a way to measure similarity between two lists of numbers. Here's a friendly mental model:

Imagine two people standing in a field, each pointing their arm in a direction. If both arms point roughly the same way — toward the same star on the horizon — those two people are "similar." If one points north and the other points south, they're very different.

An embedding (a document's numeric fingerprint) is like that pointing arm. The measure of "how much do two arms point the same way" is called **cosine similarity**. A score close to **1.0** means "nearly identical direction." A score close to **0** means "totally unrelated."

!!! tip
    You don't need to understand the geometry to use this. The library calculates it for you. Just remember: **higher score = more similar**.

??? note "For the curious (optional math)"
    You can skip this — nothing breaks.

    Cosine similarity between two lists **a** and **b** is:

    ```
    cosine_similarity(a, b) = dot(a, b) / (norm(a) * norm(b))
    ```

    - `dot(a, b)` — multiply matching pairs of numbers, then add everything up
    - `norm(a)` — the "length" of the list, calculated as the square root of the sum of squares
    - Dividing by both lengths gives a score between -1 and 1, regardless of how big the numbers are

Here's what this does, in human terms: we pretend a user's question and one document are each described by eight numbers, then measure how similar those two sets of numbers are.

```python
import numpy as np

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Return a similarity score between -1 and 1. Closer to 1 = more alike."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# Pretend fingerprints for a question and one document chunk
query_emb = np.array([0.2, -0.1,  0.8, 0.0,  0.5, -0.3, 0.1,  0.7], dtype=np.float64)
chunk_emb = np.array([0.1,  0.0,  0.9, 0.1,  0.4, -0.2, 0.2,  0.6], dtype=np.float64)

score = cosine_similarity(query_emb, chunk_emb)
print(f"Similarity: {score:.4f}")   # e.g. 0.9921
```

A score of `0.9921` means these two are extremely similar — the document chunk is a great match for the query.

!!! tip "Scoring many documents at once"
    In real RAG you compare one query against hundreds or thousands of chunks. Instead of calling `cosine_similarity` in a loop, stack all the chunks into a grid and let NumPy compare them all at once:

    ```python
    chunks = np.stack([chunk_emb, query_emb])               # (2, 8) demo grid
    norms  = np.linalg.norm(chunks, axis=1, keepdims=True)
    normed = chunks / norms                                  # shrink to unit length
    scores = normed @ query_emb / np.linalg.norm(query_emb)
    print(scores)   # one score per chunk
    ```

    This is hundreds of times faster than a Python loop.

---

## Handy NumPy tools you'll use again and again

| What you want to do | How to do it |
|---|---|
| Make a grid of zeros | `np.zeros((n, d))` |
| Make a row of ones | `np.ones(d)` |
| Make random numbers | `np.random.randn(n, d)` |
| Stack a list of rows into a grid | `np.stack(list_of_arrays)` |
| Join two grids end-to-end | `np.concatenate([a, b], axis=0)` |
| Find which item scored highest | `np.argmax(scores)` |
| Sort from highest to lowest | `np.argsort(scores)[::-1]` |

---

!!! example "Try it yourself"
    Copy the cosine similarity function above. Create two arrays of your own — try making them very similar (similar numbers) and very different (opposite-sign numbers). Run the function and see how the score changes. Can you make a score above 0.99? Can you make it negative?

---

## Next steps

- [Embeddings fundamentals](../../foundations/embeddings.md) — understand what those lists of numbers actually represent and why they capture meaning.
- [Scikit-learn](scikit-learn.md) — use TF-IDF and classifiers built on top of NumPy arrays.
