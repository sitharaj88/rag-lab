# LangChain

LangChain is the most widely-used Python framework for building LLM-powered applications, offering a rich library of integrations and a composable chain interface (LCEL) that makes wiring RAG pipelines straightforward.

## What you'll learn

- LangChain's core abstractions and where they fit in a RAG pipeline
- How to build a minimal local RAG chain using Ollama, Chroma, and LCEL
- LangChain Expression Language (LCEL) pipe syntax
- Honest pros and cons to help you choose between LangChain and LlamaIndex

---

## Installation

```bash
pip install langchain langchain-core langchain-community langchain-ollama langchain-chroma langchain-text-splitters
```

!!! note "Package split"
    As of LangChain 0.2+, integrations live in `langchain-community` or dedicated partner packages (`langchain-ollama`, `langchain-chroma`, `langchain-text-splitters`). The core `langchain` and `langchain-core` packages contain only the abstractions.

!!! warning "Legacy chains moved to `langchain-classic`"
    `LLMChain`, `RetrievalQA`, `ConversationChain`, and other legacy chain classes are **no longer part of the main `langchain` package**. They now require `pip install langchain-classic` and are imported from `langchain_classic.chains`. Prefer LCEL composition (`prompt | llm | parser`) for all new code. See [Versions & Deprecations](../sdks/versions-and-deprecations.md) for the full migration guide, and the [LangChain deep-dive tutorial](../sdks/langchain.md) for LCEL patterns.

---

## Core abstractions

| Abstraction | Role in RAG | Example class |
|-------------|-------------|---------------|
| **Document Loaders** | Load raw text from files, URLs, DBs | `PyPDFLoader`, `WebBaseLoader` |
| **Text Splitters** | Chunk documents into passages | `RecursiveCharacterTextSplitter` |
| **Embeddings** | Produce vectors from text | `OllamaEmbeddings`, `HuggingFaceEmbeddings` |
| **Vector Stores** | Store and retrieve vectors | `Chroma`, `FAISS`, `Qdrant` |
| **Retrievers** | Wrap vector stores with search logic | `vectorstore.as_retriever()` |
| **LLMs / ChatModels** | Generate answers | `OllamaLLM`, `ChatOllama` |
| **Prompts** | Structure context + question | `ChatPromptTemplate` |
| **Output Parsers** | Parse LLM output | `StrOutputParser` |

---

## Minimal local RAG chain

This example loads a plain-text file, indexes it into Chroma with Ollama embeddings, and answers questions — all locally.

```python
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 1. Load and split
loader = TextLoader("data/my_document.txt", encoding="utf-8")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# 2. Embed and store
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma.from_documents(chunks, embedding=embeddings, persist_directory="./chroma_db")

# 3. Retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# 4. Prompt
prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer the question using only the context below.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question: {question}
""")

# 5. LLM
llm = OllamaLLM(model="llama3.2")

# 6. LCEL chain  (pipe operator composes left-to-right)
def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 7. Invoke
answer = chain.invoke("What does the document say about embeddings?")
print(answer)
```

### Streaming the answer

```python
for token in chain.stream("Summarize the key points."):
    print(token, end="", flush=True)
```

---

## LCEL at a glance

LCEL (LangChain Expression Language) chains any `Runnable` with the `|` operator:

```python
chain = prompt | llm | parser
# equivalent to: parser(llm(prompt(input)))
```

Every `Runnable` supports `.invoke()`, `.stream()`, `.batch()`, and `.ainvoke()` (async) with the same interface.

---

## Pros and cons

!!! success "Strengths"
    - Largest ecosystem of integrations (100+ loaders, 50+ vector stores)
    - LCEL makes complex pipelines readable
    - Excellent documentation and community
    - LangSmith tracing for debugging chains in production

!!! warning "Weaknesses"
    - Abstraction can obscure what is actually happening
    - Breaking changes across versions; pin your dependencies
    - For document-heavy use cases, LlamaIndex's node system is more ergonomic

---

## Next steps

- [llamaindex.md](llamaindex.md) — Compare LangChain with LlamaIndex for document-heavy workloads
- [../building-blocks/indexing-pipeline.md](../building-blocks/indexing-pipeline.md) — Deep dive on the indexing pipeline
