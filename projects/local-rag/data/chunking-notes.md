# Chunking notes (sample document)

Chunking is the process of splitting documents into smaller pieces before
embedding them. The chunk size affects retrieval quality in two competing ways.

## Trade-offs

- **Small chunks** (e.g. 200–400 characters) give precise matches but may lack
  enough surrounding context for the model to answer well.
- **Large chunks** (e.g. 1000–2000 characters) carry more context but dilute the
  embedding, making retrieval less precise and wasting prompt tokens.

A common starting point is roughly 500–1000 characters with a 10–20% overlap.

## Why overlap matters

Overlap means adjacent chunks share some text. Without overlap, a sentence that
straddles a chunk boundary can be split so that neither chunk contains the full
idea, making it hard to retrieve. A typical overlap is 50–150 characters.

## Structure-aware splitting

Splitting on natural boundaries — paragraphs, headings, or sentences — usually
beats naive fixed-size splitting because it keeps related ideas together.
Recursive character splitters try a list of separators in order (paragraph,
line, sentence, word) to avoid cutting mid-thought.
