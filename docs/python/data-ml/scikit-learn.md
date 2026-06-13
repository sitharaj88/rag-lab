# Scikit-learn for RAG

Think about how you learned to recognize a cat as a kid. Nobody handed you a rulebook that said "four legs + whiskers + pointy ears = cat." Instead, you just saw *lots* of cats, and eventually your brain picked up the pattern on its own.

That's machine learning in one sentence: **show the computer many examples, and it figures out the pattern itself**. Scikit-learn is the Python library that makes this almost embarrassingly easy to do.

You don't need to understand the math behind any of this. The library handles every calculation. Your job is to feed it examples and tell it what to do.

## What you'll learn

- The three-step rhythm every scikit-learn tool follows: split your data, study the examples, then make predictions
- How to turn text into numbers so a computer can work with it (this is called **TF-IDF** — a fancy name for "count which words make a document special")
- How to build a tiny classifier that automatically sorts incoming questions into categories
- Where scikit-learn fits in RAG versus fancier deep-learning models

---

## The three-step rhythm: split, study, guess

Every scikit-learn workflow follows the same pattern, and it maps perfectly to how humans learn:

| Step | What it does | Like a student who… |
|---|---|---|
| **split** | Set aside some examples to test with later | Saves a few practice problems for self-quizzing |
| **fit** | Learn the pattern from your training examples | Studies the material |
| **predict** | Make a guess on new, unseen examples | Takes the actual test |

That's really all there is to it. `fit` = study. `predict` = guess.

!!! tip
    You'll also see `transform` — that just means "convert this data into a different format" (like turning words into numbers). And `fit_transform` is a shortcut that does both in one go.

---

## Step 1 — Set some examples aside for testing

Before you teach the computer anything, you set aside a slice of your examples to test with *after* training. This prevents cheating — the computer won't have already seen the test answers.

Here's what this does, in human terms: we shuffle a small list of questions and their labels, then keep 25% of them as a quiz for later.

```python
from sklearn.model_selection import train_test_split

texts  = ["What is RAG?", "Explain chunking", "List embeddings", "How to index?",
          "Retrieval methods", "Generation step", "Prompt template", "Vector store"]
labels = ["retrieval", "chunking", "embedding", "indexing",
          "retrieval", "generation", "prompting", "indexing"]

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.25, random_state=42
)
print(len(X_train), len(X_test))   # 6 training examples, 2 test examples
```

`X_train` is the study material. `X_test` is the quiz the computer hasn't peeked at yet.

---

## Step 2 — Turn words into numbers with TF-IDF

Computers can't understand words directly — they work with numbers. So we need a way to convert text into a list of numbers that captures *what the text is about*.

**TF-IDF** does this by asking: "which words appear a lot in *this* document but rarely in *all* documents?" Those rare-but-frequent words are what make the document special. Common words like "the" and "is" get ignored; distinctive words like "embeddings" or "chunking" get high scores.

In plain words, TF-IDF is a way of saying "here's a fingerprint of this document based on its unusual words." (The fancy name stands for Term Frequency–Inverse Document Frequency, but you never need to spell that out.)

!!! tip
    Don't memorize the formula — the library calculates it for you. Just remember: **rare words that appear in this document get a high score**.

Here's what this does, in human terms: we take four short documents, turn them into number-fingerprints, then score a query against all four to find the best match.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

corpus = [
    "Retrieval-Augmented Generation combines retrieval with language models.",
    "Vector databases store dense embeddings for semantic search.",
    "TF-IDF uses term frequency and inverse document frequency.",
    "Chunking splits documents into smaller passages for indexing.",
]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(corpus)   # turns 4 documents into 4 fingerprints

query = "How does retrieval work in RAG?"
query_vec = vectorizer.transform([query])          # same fingerprint style for the query

scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

for idx in scores.argsort()[::-1]:
    print(f"{scores[idx]:.4f}  {corpus[idx][:60]}")
```

The output ranks documents from most to least relevant to your query. The one mentioning "retrieval" should come out on top.

!!! info "Keyword search vs. meaning search"
    TF-IDF is great at finding documents that share the **exact same words** as your query. It's fast, needs no training data, and works well for precise terms like product codes or technical jargon. Deep embedding models (dense retrieval) go further — they understand that "automobile" and "car" mean the same thing. Hybrid search combines both approaches — see [Hybrid Search](../../advanced/hybrid-search.md).

---

## Step 3 — Auto-sort questions into buckets with a classifier

Imagine you run a help desk and every question that comes in needs to go to the right team: coding questions to the devs, billing questions to finance, and so on. You *could* read every question yourself, or you could teach a program to sort them automatically.

In RAG, a **query router** does the same thing: it reads an incoming question and decides which tool, retriever, or prompt template to use. Below, we build a tiny one.

Here's what this does, in human terms: we give the computer several example questions with labels ("this is a concept question", "this is a code question"), let it study those examples, then ask it to label two new questions it's never seen.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# Training examples: question text + the correct category label
train_queries = [
    "What is an embedding?",          # concept
    "How do vector databases work?",  # concept
    "Show me Python code for RAG",    # code
    "Write a function to chunk text", # code
    "What happened in the meeting?",  # factual
    "Summarise the Q3 report",        # factual
    "Explain attention mechanism",    # concept
    "Give me a FastAPI example",      # code
]
train_labels = ["concept", "concept", "code", "code",
                "factual", "factual", "concept", "code"]

# Two new questions to test on
test_queries = ["How does chunking affect retrieval?", "Write a loader for PDFs"]
test_labels  = ["concept", "code"]

# A Pipeline chains the steps: first turn text into numbers, then classify
router = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),   # text → numbers
    ("clf",   LogisticRegression(max_iter=500)),        # numbers → category
])

router.fit(train_queries, train_labels)       # study the examples
predictions = router.predict(test_queries)    # guess the categories

print(classification_report(test_labels, predictions, zero_division=0))

# Show the routing decision for each question
for query, route in zip(test_queries, predictions):
    print(f"[{route}]  {query}")
```

The `Pipeline` object is a convenience shortcut — instead of calling `transform` manually every time, you just call `fit` and `predict` on the whole chain at once.

!!! tip
    With only 8 training examples, accuracy will be imperfect — that's expected. In a real project you'd have dozens or hundreds of labeled examples. The same code works at any scale; you just swap in more data.

---

## Scikit-learn vs. deep embedding models — a quick comparison

| | Scikit-learn (TF-IDF + cosine) | Deep embedding model |
|---|---|---|
| Speed | Very fast, runs on a laptop | Works best with a GPU for large collections |
| What it understands | Exact keywords | Meaning and context ("car" ≈ "automobile") |
| Training data needed | None — just give it your documents | Pre-trained; optional fine-tuning |
| Best for | Hybrid retrieval, routing, quick baselines | Primary semantic search |

Neither approach is "better" — they're tools for different jobs. Many production RAG systems use both.

---

!!! example "Try it yourself"
    Add two more training examples to `train_queries` and `train_labels` with a brand-new category called `"greeting"` (e.g., "Hello, how are you?" and "Good morning!"). Then add `"Hi there"` to `test_queries` and see if the router correctly identifies it as a greeting. What happens to the `classification_report`?

---

## Next steps

- [Hybrid Search](../../advanced/hybrid-search.md) — combine TF-IDF keyword scores with dense embedding scores for the best of both worlds.
- [Working with Model APIs](../ai-python/working-with-model-apis.md) — call embedding and generation endpoints to complement scikit-learn's classical ML.
