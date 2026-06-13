# Pandas for RAG

Imagine a spreadsheet — rows of data, columns with names, and the ability to filter or sort with a click. Now imagine controlling that spreadsheet with a few lines of code instead of a mouse. That's pandas. It's Excel, but scriptable.

You won't need to memorise formulas or worry about cell references. You just describe what you want in plain Python, and pandas figures it out.

## What you'll learn

- What a DataFrame and a Series are (don't worry, they're just tables and columns)
- How to open a CSV file — like opening a spreadsheet
- How to ask your table a question — filtering rows you care about
- How to add a new column with your own logic
- How to "make piles and count each pile" — groupby aggregations
- How to audit a chunk corpus before you index it

---

## A table and a column

There are two main building blocks in pandas. A **Series** is one column of data — a list with a label. A **DataFrame** is a full table: multiple columns sitting side by side.

You'll mostly work with DataFrames. Think of them as the spreadsheet, and Series as individual columns you pull out of it.

Here's how to make both from scratch:

```python
import pandas as pd

# A Series is just a labelled list — one column of data
lengths = pd.Series([120, 340, 88, 512], name="length")
print(lengths.mean())   # 265.0  ← the average of those four numbers

# A DataFrame is a full table built from a dictionary
data = {
    "source": ["report.pdf", "report.pdf", "faq.txt"],
    "chunk":  ["Intro paragraph", "Methods section", "Q: What is RAG?"],
    "length": [120, 340, 88],
}
df = pd.DataFrame(data)
print(df.dtypes)   # tells you what type each column is (text, number, etc.)
print(df.head())   # shows the first 5 rows — a quick peek at your data
```

After running this, `df.head()` prints a neat little table in your terminal. That's your DataFrame. Congrats — you already know how to make one.

---

## Opening a CSV file

Reading a CSV is the first thing you'll do in almost every data task. In plain words: it opens the file and loads it into a table you can work with. (The fancy name for this is **reading into a DataFrame**.)

```python
# Load a pre-chunked corpus from a file on disk
df = pd.read_csv("chunks.csv")          # pandas guesses the column types for you

# Three ways to take a quick look at what you've got
print(df.shape)      # (rows, cols) — how big is this table?
print(df.info())     # column names, types, and whether any values are missing
print(df.describe()) # quick stats for numeric columns: min, max, average, etc.
```

`df.shape` might print `(250, 3)` — meaning 250 rows and 3 columns. That's it. Now you know how big your dataset is.

```python
# When you're done, save the table back to a CSV file
df.to_csv("chunks_annotated.csv", index=False)
```

The `index=False` just means "don't add a column of row numbers to the file." You almost always want that.

---

## Asking the table a question

Filtering is how you say "show me only the rows where X is true." You're not changing the data — you're just asking a question and getting a smaller table back.

```python
# Pull out just one column (returns a Series)
sources = df["source"]

# Pull out two columns at once (returns a smaller DataFrame)
subset = df[["source", "length"]]

# Keep only the rows where length is greater than 200
long_chunks = df[df["length"] > 200]

# Same thing, written in a more readable way
long_chunks = df.query("length > 200")

# You can combine conditions too — PDFs that are also longer than 100 characters
pdf_long = df.query("source.str.endswith('.pdf') and length > 100")
```

The result of each filter is a brand new table — smaller, but same shape. You haven't touched the original `df`.

!!! tip "Filtering feels weird at first"
    `df[df["length"] > 200]` looks like you're passing a table into itself, and that's... kind of what you're doing. The inner part `df["length"] > 200` creates a column of True/False values, and the outer part uses that to pick rows. Once it clicks, you'll use it constantly.

---

## Adding a column with your own logic

Sometimes you want to create a new column based on values that already exist. You can use `apply` to run a function on every row. Think of it as "do this thing to each value in this column."

```python
# Add a word-count column by splitting each chunk on spaces
df["word_count"] = df["chunk"].apply(lambda text: len(text.split()))

# Or use a proper function for something more complex
def size_label(n: int) -> str:
    if n < 100:
        return "small"
    elif n < 300:
        return "medium"
    return "large"

# Apply that function to every value in the "length" column
df["size_label"] = df["length"].apply(size_label)
```

After this runs, your DataFrame has two new columns: `word_count` and `size_label`. The original data is untouched — you just added to it.

---

## Make piles and count each pile

`groupby` is how you answer questions like "how many chunks came from each document?" You pick a column to group by (like `source`), and pandas makes a separate pile for each unique value. Then you count, average, or total up each pile.

```python
# How many chunks per source file, and what's the average chunk length?
summary = (
    df.groupby("source")
    .agg(
        num_chunks=("chunk", "count"),    # count the rows in each pile
        avg_length=("length", "mean"),    # average the "length" column
        max_length=("length", "max"),     # find the longest chunk in each pile
    )
    .reset_index()   # turns the group labels back into a regular column
)
print(summary)
```

The result is a tidy table — one row per source file, with the stats you asked for next to it. Perfect for a quick sanity check on your corpus.

---

## Auditing a chunk corpus

Before you index your chunks into a vector store, it's worth checking them. Are any too short to be useful? Are pages missing? This is a common RAG workflow step.

```python
import pandas as pd

# Pretend this came from your text splitter
raw_chunks = [
    {"source": "paper.pdf",  "chunk": "Retrieval-Augmented Generation combines...", "page": 1},
    {"source": "paper.pdf",  "chunk": "The retriever scores candidates by cosine...", "page": 2},
    {"source": "readme.txt", "chunk": "Install with pip install -r requirements.txt", "page": None},
]

df = pd.DataFrame(raw_chunks)

# Add some useful columns
df["length"]     = df["chunk"].str.len()              # how many characters?
df["word_count"] = df["chunk"].str.split().str.len()  # how many words?
df["has_page"]   = df["page"].notna()                 # does it have a page number?

# Flag anything suspiciously short — tiny chunks often hurt retrieval quality
df["too_short"] = df["length"] < 50

# Print only the problematic ones so you can review them
print(df[df["too_short"]][["source", "chunk", "length"]])

# Save the full audit to a file
df.to_csv("corpus_audit.csv", index=False)
```

Running this gives you a filtered table of short chunks — just the ones that might cause problems. You can open `corpus_audit.csv` in a spreadsheet app to review everything at your own pace.

!!! tip "Pretty-print as a table"
    `df.to_markdown()` (needs the `tabulate` package) prints your DataFrame as a GitHub-flavoured Markdown table. Handy for pasting corpus stats into experiment notes.

!!! example "Try it yourself"
    Load the corpus audit DataFrame above and add one more column: `"is_question"` that is `True` if the chunk text contains a `?` character. Hint: `df["chunk"].str.contains("?", regex=False)` does the trick.

---

## Next steps

- [Matplotlib](matplotlib.md) — visualise the `length` and `word_count` distributions from your corpus DataFrame.
- [RAG Evaluation](../../advanced/evaluation.md) — load retrieval metrics into a DataFrame and aggregate results across query sets.
