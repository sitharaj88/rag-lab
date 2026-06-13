# Further Reading

A curated list of papers, official documentation, and learning resources for going deeper on RAG and its components.

## Foundational papers

[**Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks** — Lewis et al., 2020](https://arxiv.org/abs/2005.11401)
:   The paper that named and popularised the RAG pattern. Introduces the retriever-generator architecture and demonstrates strong results on open-domain QA benchmarks.

[**Dense Passage Retrieval for Open-Domain Question Answering** — Karpukhin et al., 2020](https://arxiv.org/abs/2004.04906)
:   Establishes dense bi-encoder retrieval as a practical alternative to BM25 for open-domain QA, and provides the training recipe that most modern embedding models descend from.

[**Precise Zero-Shot Dense Retrieval without Relevance Labels (HyDE)** — Gao et al., 2022](https://arxiv.org/abs/2212.10496)
:   Proposes Hypothetical Document Embeddings — generating a fake answer and embedding it instead of the raw query — as a zero-shot way to close the query-document embedding gap.

[**ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT** — Khattab & Zaharia, 2020](https://arxiv.org/abs/2004.12832)
:   Introduces late interaction: store per-token document embeddings at index time and compute fine-grained relevance scores at query time, achieving near cross-encoder accuracy at bi-encoder speed.

[**REALM: Retrieval-Augmented Language Model Pre-Training** — Guu et al., 2020](https://arxiv.org/abs/2002.08909)
:   An early end-to-end approach that jointly trains a retriever and language model, motivating later work on learned retrieval in generative pipelines.

## Official documentation

[**Ollama**](https://ollama.com/docs)
:   The official guide for installing, pulling models, and using the REST API and Python library that power local generation in this portal.

[**ChromaDB**](https://docs.trychroma.com)
:   Full reference for ChromaDB's `PersistentClient`, collections, embedding functions, metadata filtering, and query API.

[**LangChain**](https://python.langchain.com/docs/introduction)
:   Documentation for the most widely-used RAG orchestration framework, covering document loaders, text splitters, retrievers, chains, and agents.

[**LlamaIndex**](https://docs.llamaindex.ai)
:   Documentation for LlamaIndex (formerly GPT Index), which provides high-level abstractions for ingestion pipelines, query engines, and multi-document reasoning.

[**sentence-transformers**](https://sbert.net)
:   The library documentation for the embedding models used in this portal, including model cards, training guides, and the `SentenceTransformer` API.

[**Hugging Face Model Hub**](https://huggingface.co/models?pipeline_tag=sentence-similarity)
:   Browse and compare sentence-similarity and embedding models, including `all-MiniLM-L6-v2` and its larger siblings.

## Evaluation

[**Ragas — Evaluation Framework for RAG Pipelines**](https://docs.ragas.io)
:   An open-source library that computes RAG-specific metrics — faithfulness, answer relevancy, context precision, and context recall — with or without a reference answer.

[**BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models**](https://arxiv.org/abs/2104.08663)
:   A standard benchmark suite for retrieval models across 18 datasets; useful for comparing embedding models before committing to one for production.

## Learning resources

[**Pinecone Learning Center — What is RAG?**](https://www.pinecone.io/learn/retrieval-augmented-generation/)
:   A thorough visual explainer of the RAG pattern, covering chunking, embedding, retrieval, and generation with diagrams that complement the foundations section of this portal.

[**Langchain Blog — Evaluating RAG**](https://blog.langchain.dev/evaluating-rag-pipelines-with-ragas-langsmith/)
:   A practical walkthrough of wiring Ragas into a LangChain pipeline, with concrete examples of metric interpretation.

[**fast.ai Practical Deep Learning — NLP chapters**](https://course.fast.ai)
:   Free course that builds intuition for language models and embeddings from first principles — good background reading before diving into retrieval architectures.

!!! tip "Keep it grounded"
    When reading papers, cross-reference their claims against the evaluation benchmarks they use. A model that tops one leaderboard may underperform on your specific domain or document type.

## Next steps

- [Tools overview](../tools/index.md) — find the right library for your use-case
- [Evaluation deep-dive](../advanced/evaluation.md) — measure and improve your RAG pipeline
