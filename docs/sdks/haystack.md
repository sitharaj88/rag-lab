# Haystack SDK

Haystack 2.x is a pipeline-centric framework for building production-ready NLP and RAG systems. You assemble typed components into a directed graph, wire their inputs and outputs explicitly, and run the whole graph with a single `.run()` call.

## What you'll learn

- How Haystack's component-and-connect model works
- How to build a two-stage pipeline: an indexing pipeline and a query pipeline
- How to use `InMemoryDocumentStore`, SentenceTransformers embedders, a retriever, `ChatPromptBuilder`, and a local Ollama generator
- When Haystack is the right framework to choose
- Which legacy package to avoid

## Install

```bash
pip install haystack-ai
# For local Ollama generation:
pip install ollama-haystack
# For SentenceTransformers embedders (included in haystack-ai extras or install separately):
pip install sentence-transformers
```

!!! warning "farm-haystack is EOL — do not install it"
    `farm-haystack` is the 1.x package and is end-of-life. It has a completely different API (no `Pipeline.add_component`, no `connect`). All new projects must use `haystack-ai` (Haystack 2.x). Searching for Haystack tutorials online will surface many 1.x examples — they are incompatible. See [versions-and-deprecations.md](versions-and-deprecations.md) for the full 1.x → 2.x migration notes.

## The component-and-connect model

Haystack 2.x pipelines are explicit dataflow graphs:

1. Create a `Pipeline()` instance.
2. Add each component with `pipeline.add_component("name", ComponentInstance())`.
3. Wire outputs to inputs with `pipeline.connect("sender.output_name", "receiver.input_name")`.
4. Run with `pipeline.run({"component_name": {"param": value}})`.

Components are typed Python classes. Mismatched types are caught at `connect()` time, not at runtime.

## Full local RAG example

### Indexing pipeline

The indexing pipeline converts raw text into embedded documents stored in a document store.

```python
from haystack import Pipeline, Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.writers import DocumentWriter

# Shared document store — pass the same instance to both pipelines
document_store = InMemoryDocumentStore()

# Raw documents to index
docs = [
    Document(content="Retrieval-Augmented Generation (RAG) grounds LLM answers in external knowledge."),
    Document(content="Vector databases store embeddings and support approximate nearest-neighbour search."),
    Document(content="Chunking splits long documents into smaller pieces before embedding."),
    Document(content="Ollama lets you run open-weight LLMs locally without a cloud API key."),
    Document(content="Haystack 2.x uses a component-and-connect model for explicit pipeline composition."),
]

# Build the indexing pipeline
index_pipeline = Pipeline()
index_pipeline.add_component(
    "embedder",
    SentenceTransformersDocumentEmbedder(model="BAAI/bge-small-en-v1.5"),
)
index_pipeline.add_component("writer", DocumentWriter(document_store=document_store))

index_pipeline.connect("embedder.documents", "writer.documents")

# Warm up the embedder model, then run
index_pipeline.get_component("embedder").warm_up()
index_pipeline.run({"embedder": {"documents": docs}})

print(f"Indexed {document_store.count_documents()} documents")
```

### Query pipeline

The query pipeline embeds the user question, retrieves relevant documents, fills a prompt template, and generates an answer with a local Ollama model.

```python
from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.builders import ChatPromptBuilder
from haystack.dataclasses import ChatMessage
from ollama_haystack import OllamaChatGenerator  # pip install ollama-haystack

# Prompt template — Jinja2 syntax
template = [
    ChatMessage.from_system("You are a helpful assistant. Answer only from the context provided."),
    ChatMessage.from_user(
        """Context:
{% for doc in documents %}
- {{ doc.content }}
{% endfor %}

Question: {{ question }}
Answer:"""
    ),
]

query_pipeline = Pipeline()
query_pipeline.add_component(
    "text_embedder",
    SentenceTransformersTextEmbedder(model="BAAI/bge-small-en-v1.5"),
)
query_pipeline.add_component(
    "retriever",
    InMemoryEmbeddingRetriever(document_store=document_store, top_k=3),
)
query_pipeline.add_component(
    "prompt_builder",
    ChatPromptBuilder(template=template),
)
query_pipeline.add_component(
    "generator",
    OllamaChatGenerator(model="llama3.2", url="http://localhost:11434"),
)

# Wire the components
query_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
query_pipeline.connect("retriever.documents", "prompt_builder.documents")
query_pipeline.connect("prompt_builder.prompt", "generator.messages")

# Warm up the embedder
query_pipeline.get_component("text_embedder").warm_up()

# Run a query
question = "What is RAG and why does it help?"
result = query_pipeline.run(
    {
        "text_embedder": {"text": question},
        "prompt_builder": {"question": question},
    }
)

answer = result["generator"]["replies"][0].content
print(answer)
```

## Pipeline visualisation

Haystack can draw the pipeline graph for debugging:

```python
query_pipeline.show()          # opens an interactive view in Jupyter
query_pipeline.draw("rag.png") # saves a PNG (requires pygraphviz)
```

## Pros and cons

| Pros | Cons |
|---|---|
| Explicit typed dataflow makes pipelines auditable and debuggable | More verbose than LCEL for simple chains |
| Type-checked `connect()` catches wiring errors early | Smaller integration ecosystem than LangChain |
| Strong production features: serialisation, tracing, evaluation | In-memory store is not suitable for large corpora in production |
| Pipeline serialisation to YAML enables config-driven deployments | Learning curve for understanding component I/O contracts |

**When to choose Haystack:** Haystack excels when you need an auditable, type-safe pipeline that can be serialised to YAML for config-driven deployments, or when you are building a production system and want early detection of component wiring mistakes. For a broader integration ecosystem, see [LangChain](langchain.md). For a document-centric query engine with rich metadata, see [LlamaIndex](llamaindex.md).

## Next steps

- Deepen your retrieval understanding: [../foundations/retrieval.md](../foundations/retrieval.md)
- Compare with [LangChain](langchain.md)
- Review framework version history: [versions-and-deprecations.md](versions-and-deprecations.md)
- See vector store options: [../sdks/faiss.md](faiss.md), [../sdks/qdrant.md](qdrant.md)
- Follow a complete tutorial: [../tutorials/02-rag-with-ollama-chroma.md](../tutorials/02-rag-with-ollama-chroma.md)
