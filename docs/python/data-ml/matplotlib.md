# Matplotlib for RAG

Making a chart in Python is just you telling Python: "here are my numbers — draw them." That's really it. You hand over a list of values, say what kind of chart you want, add a title so a human can read it, and you're done.

Matplotlib is the library that handles the drawing. You don't need to be a designer. You don't need to know any math. You just describe the picture you want, step by step.

## What you'll learn

- The two-line setup every chart starts with
- When to use a line chart, a bar chart, a histogram, or a scatter plot — in plain everyday terms
- How to add labels and a legend so the chart makes sense to anyone
- How to save the chart as an image file
- A note on seaborn, which makes everything look nicer with one extra line

---

## The two-line setup

Every matplotlib chart starts the same way. You create a **figure** (the blank canvas) and an **axes** (the area where the chart actually lives). Most of the time you do both in one line:

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()   # one canvas, one chart area

ax.set_title("My Chart")   # title at the top
ax.set_xlabel("X axis")    # label along the bottom
ax.set_ylabel("Y axis")    # label along the left side

plt.tight_layout()   # fixes spacing so nothing overlaps
plt.show()           # opens a window to display the chart
```

Running this shows a blank chart with labels. Not exciting yet — but that's the skeleton every chart builds on.

!!! tip "Two styles you'll see online"
    You might see `plt.plot(...)` (quick and short) or `ax.plot(...)` (slightly more typing). Both work. The `ax.` style is safer when you have multiple charts, and it's what this guide uses. When you copy code from the internet, just know that both are the same library.

---

## Line chart — "how does this change over time?"

Use a line chart when your values go in order and you want to see a trend. In RAG terms: "how does the similarity score drop as we look at lower-ranked results?"

```python
import matplotlib.pyplot as plt

# Similarity scores for the top 10 retrieved chunks, from best to worst
ranks  = list(range(1, 11))
scores = [0.91, 0.87, 0.83, 0.79, 0.74, 0.68, 0.61, 0.55, 0.49, 0.42]

fig, ax = plt.subplots(figsize=(7, 4))

# Draw the line — the dots (marker="o") help you see each individual score
ax.plot(ranks, scores, marker="o", linewidth=2, color="steelblue", label="cosine similarity")

# A dashed red line shows the cutoff threshold — chunks below this might not be useful
ax.axhline(0.70, color="tomato", linestyle="--", label="threshold = 0.70")

ax.set_title("Retrieval similarity scores — query: 'What is chunking?'")
ax.set_xlabel("Rank")
ax.set_ylabel("Cosine similarity")
ax.set_xticks(ranks)   # make sure every rank number shows on the x-axis
ax.legend()            # shows the labels ("cosine similarity", "threshold = 0.70")
ax.grid(True, alpha=0.3)  # light grid lines to make values easier to read

plt.tight_layout()
plt.savefig("retrieval_scores.png", dpi=150)   # save it as an image file
plt.show()
```

The chart will show a line going steadily downward — which is exactly what you'd expect. Chunks ranked #1 are closest to your query; chunks ranked #10 are the weakest match. The red dashed line shows where you'd cut off results.

---

## Bar chart — "how do these categories compare?"

Use a bar chart when you have named groups and a number for each one. In RAG terms: "which source document tends to come up in retrieval, and how relevant is it?"

```python
import matplotlib.pyplot as plt

sources = ["paper.pdf", "readme.txt", "faq.md", "blog_post.txt"]
avg_sim = [0.82, 0.65, 0.74, 0.59]

fig, ax = plt.subplots(figsize=(6, 4))

# One bar per source — height shows average similarity
bars = ax.bar(sources, avg_sim, color="cornflowerblue", edgecolor="white")

# Print the exact number on top of each bar so you don't have to squint
ax.bar_label(bars, fmt="%.2f", padding=3)

ax.set_ylim(0, 1.0)   # similarity scores run from 0 to 1
ax.set_title("Average retrieval similarity by source")
ax.set_ylabel("Mean cosine similarity")
ax.set_xlabel("Source document")

plt.tight_layout()
plt.savefig("similarity_by_source.png", dpi=150)
plt.show()
```

Each bar's height tells you how relevant that document was on average. A quick glance reveals which sources are actually being useful in your pipeline — and which ones might be dead weight.

---

## Histogram — "how are my values spread out?"

A histogram is the right tool when you have a big pile of numbers and want to see where most of them land. In RAG terms: "are my chunk lengths mostly in a healthy range, or do I have a lot of tiny or enormous chunks?"

Short and overly long chunks can both hurt retrieval quality, so this is a genuinely useful check.

```python
import matplotlib.pyplot as plt
import numpy as np

# In a real project you'd load these from your corpus DataFrame
# (see the Pandas page for how to build that)
rng = np.random.default_rng(42)
lengths = rng.integers(50, 800, size=300)   # 300 synthetic chunk lengths

fig, ax = plt.subplots(figsize=(7, 4))

# Each bar covers a range of lengths; taller bars mean more chunks landed there
ax.hist(lengths, bins=30, color="mediumseagreen", edgecolor="white")

# Mark the average length with a vertical dashed line
ax.axvline(lengths.mean(), color="tomato", linestyle="--", label=f"mean = {lengths.mean():.0f}")

ax.set_title("Chunk-length distribution")
ax.set_xlabel("Characters")
ax.set_ylabel("Count")
ax.legend()

plt.tight_layout()
plt.savefig("chunk_length_hist.png", dpi=150)
plt.show()
```

A healthy histogram has most of its bars clustered together in the middle — not spread wildly from 10 to 5000. If you see a big spike at very low values, you probably have a lot of near-empty chunks worth investigating.

---

## Scatter plot — "do these two things relate to each other?"

Use a scatter plot when you want to see whether two numbers seem to move together. Each dot is one data point; you put one measurement on each axis and look for patterns.

In RAG terms: "do longer chunks get worse retrieval scores?"

```python
import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(0)
lengths = rng.integers(80, 600, size=50)
scores  = 0.9 - (lengths / 600) * 0.4 + rng.normal(0, 0.05, 50)

fig, ax = plt.subplots(figsize=(6, 4))

# Each dot is one chunk — its x position is its length, y position is its score
ax.scatter(lengths, scores, alpha=0.6, edgecolors="k", linewidths=0.4)

ax.set_title("Chunk length vs. retrieval similarity")
ax.set_xlabel("Chunk length (characters)")
ax.set_ylabel("Cosine similarity")

plt.tight_layout()
plt.savefig("length_vs_similarity.png", dpi=150)
plt.show()
```

If the dots trend downward from left to right, longer chunks tend to score worse. If the dots look like random confetti, there's no obvious relationship. Either way, the chart shows you instantly — no spreadsheet gymnastics needed.

---

## Saving the chart to a file

Once you're happy with a chart, save it. You can save as a PNG (a regular image) or SVG (a vector format that stays sharp at any size, great for reports).

```python
plt.savefig("output.png", dpi=150, bbox_inches="tight")   # sharp raster image
plt.savefig("output.svg")                                  # vector — scales to any size
```

Call `savefig` before `plt.show()` — after `show()` the figure gets cleared.

!!! tip "Want prettier charts with almost no extra work?"
    [Seaborn](https://seaborn.pydata.org/) is built on top of matplotlib and adds nicer default colours, fonts, and some extra chart types (`sns.histplot`, `sns.heatmap`). Adding two lines to any script upgrades how everything looks:
    ```python
    import seaborn as sns
    sns.set_theme()
    ```
    All the `ax.*` calls you already know still work exactly the same.

!!! example "Try it yourself"
    Take the bar chart example above and change the bar colours so any source with an average similarity above 0.75 appears in green and the rest in grey. Hint: build the colour list before calling `ax.bar()`, like `colors = ["green" if s > 0.75 else "grey" for s in avg_sim]`.

---

## Next steps

- [Pandas](pandas.md) — build the DataFrame of chunk metadata you plotted above.
- [RAG Evaluation](../../advanced/evaluation.md) — plot precision@k and MRR curves across evaluation sets.
