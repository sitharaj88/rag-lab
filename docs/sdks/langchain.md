# LangChain SDK

LangChain is a composable framework for building LLM-powered applications, offering modular packages for every layer of the stack. This page covers the current v1 API with LCEL-based chains and local RAG using Ollama, Chroma, and HuggingFace embeddings.

## What you'll learn

- How LangChain's modular package layout works and which packages to install
- How to compose pipelines with LCEL (LangChain Expression Language)
- How to build a fully local RAG chain: load → split → embed → store → retrieve → answer
- How to wire an agent with `create_agent` for agentic flows
- Which legacy patterns to avoid and where to find them if needed

## Install

Install only the packages you need. The core split keeps your environment lean:

```bash
pip install langchain langchain-core langchain-community \
            langchain-ollama langchain-chroma langchain-text-splitters
```

| Package | Purpose |
|---|---|
| `langchain` | High-level chains, agents, and utilities |
| `langchain-core` | LCEL primitives, `Runnable`, base classes |
| `langchain-community` | Community document loaders, tools, integrations |
| `langchain-ollama` | `ChatOllama` and `OllamaEmbeddings` |
| `langchain-chroma` | `Chroma` vector store integration |
| `langchain-text-splitters` | `RecursiveCharacterTextSplitter` and friends |

!!! warning "Legacy chains are no longer in core"
    `LLMChain`, `ConversationChain`, and other legacy chain classes were removed from `langchain` core in v1. They are still available via `pip install langchain-classic` and `from langchain_classic.chains import LLMChain`, but **new projects should use LCEL** (`prompt | llm | parser`) instead. See [versions-and-deprecations.md](versions-and-deprecations.md) for the full migration timeline.

## LCEL: LangChain Expression Language

LCEL lets you compose runnables with the pipe operator `|`. Every component — prompts, models, parsers, retrievers — is a `Runnable` with a uniform `.invoke()` / `.stream()` / `.batch()` interface.

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

prompt = ChatPromptTemplate.from_template("Answer briefly: {question}")
llm = ChatOllama(model="llama3.2")
parser = StrOutputParser()

chain = prompt | llm | parser
print(chain.invoke({"question": "What is a vector database?"}))
```

## Full local RAG chain

The example below is self-contained and runs entirely on your machine. It loads text files, splits them, embeds them with Ollama, stores them in Chroma, then answers questions using an LCEL retrieval chain.

### Step 1 — load and split documents

```python
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load every .txt file in ./docs_data/
loader = DirectoryLoader("./docs_data", glob="**/*.txt", loader_cls=TextLoader)
raw_docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)
splits = splitter.split_documents(raw_docs)
print(f"Split into {len(splits)} chunks")
```

### Step 2 — embed and store in Chroma

```python
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

embed_model = OllamaEmbeddings(model="nomic-embed-text")

vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embed_model,
    persist_directory="./chroma_db",  # persists to disk automatically
    collection_name="rag_lab",
)
```

### Step 3 — build the retrieval chain with LCEL

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

prompt = ChatPromptTemplate.from_template(
    """You are a helpful assistant. Answer the question using only the context below.
If you cannot answer from the context, say "I don't know."

Context:
{context}

Question: {question}
"""
)

llm = ChatOllama(model="llama3.2")
parser = StrOutputParser()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | parser
)

answer = rag_chain.invoke("What topics are covered in the documentation?")
print(answer)
```

### Step 4 — reload from disk (no re-embedding needed)

```python
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

embed_model = OllamaEmbeddings(model="nomic-embed-text")

vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embed_model,
    collection_name="rag_lab",
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
```

## Agentic flows with `create_agent`

For tool-using agents, import `create_agent` from `langchain.agents`. Pass a list of tools and the LLM:

```python
from langchain.agents import create_agent, AgentExecutor
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools import DuckDuckGoSearchRun

llm = ChatOllama(model="llama3.2")
tools = [DuckDuckGoSearchRun()]

# create_agent returns a Runnable agent; wrap in AgentExecutor to run
agent = create_agent(llm=llm, tools=tools, prompt=ChatPromptTemplate.from_template("{input}"))
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
result = executor.invoke({"input": "What is RAG in AI?"})
print(result["output"])
```

## Pros and cons

| Pros | Cons |
|---|---|
| Largest ecosystem of integrations | Package surface area can feel overwhelming |
| LCEL makes pipeline composition explicit and debuggable | Some abstractions add indirection that complicates debugging |
| First-class streaming and async support | Rapid API evolution — pin versions in production |
| Strong community and documentation | Legacy patterns still visible in many tutorials online |

**When to use LangChain:** Choose it when you need a broad integration library (many loaders, tools, vector stores) or want LCEL's composable, inspectable pipeline model. For a more opinionated document-centric query engine, consider [LlamaIndex](llamaindex.md).

## Next steps

- Explore the local tooling guide: [../tools/langchain.md](../tools/langchain.md)
- Compare frameworks: [LlamaIndex](llamaindex.md) and [Haystack](haystack.md)
- Review breaking changes and migration notes: [versions-and-deprecations.md](versions-and-deprecations.md)
- Deepen your retrieval understanding: [../foundations/retrieval.md](../foundations/retrieval.md)
- See a full worked project: [../tutorials/02-rag-with-ollama-chroma.md](../tutorials/02-rag-with-ollama-chroma.md)
