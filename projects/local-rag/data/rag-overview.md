# Retrieval-Augmented Generation (sample document)

Retrieval-Augmented Generation (RAG) is a technique that combines a retrieval
system with a generative language model. Instead of relying solely on the
parameters learned during training, a RAG system fetches relevant documents
from an external knowledge base and provides them to the model as context.

## Why teams use RAG

- **Freshness**: The knowledge base can be updated at any time without retraining the model.
- **Grounding**: Answers are based on retrieved evidence, which reduces hallucination.
- **Attribution**: Because the source passages are known, answers can cite them.
- **Cost**: Updating an index is far cheaper than fine-tuning or retraining a model.

## The two phases of RAG

1. **Indexing (offline)**: Documents are loaded, split into chunks, embedded into
   vectors, and stored in a vector database.
2. **Querying (online)**: A user question is embedded, the most similar chunks are
   retrieved, and those chunks are inserted into the prompt sent to the LLM.

## Key components

- **Embedding model**: Turns text into vectors that capture meaning. A common
  open-source choice is `all-MiniLM-L6-v2`, which produces 384-dimensional vectors.
- **Vector store**: Holds embeddings and supports fast nearest-neighbor search.
  ChromaDB is a popular local option.
- **LLM**: Generates the final answer. With Ollama you can run models like
  Llama 3.2 entirely on your own machine.
