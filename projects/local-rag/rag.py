"""The query-time half of RAG: retrieve relevant chunks, then generate a
grounded answer with a local Ollama model.

Run interactively:
    python rag.py

Or import ``RAGPipeline`` from the Streamlit app / your own code.
"""
from __future__ import annotations

from dataclasses import dataclass

import chromadb
import ollama
from sentence_transformers import SentenceTransformer

import config

SYSTEM_PROMPT = """You are a precise assistant. Answer the user's question using \
ONLY the context provided. If the context does not contain the answer, say you \
don't know based on the provided documents — do not invent facts. When you use a \
fact, cite its source in square brackets like [source.md]."""


@dataclass
class Retrieved:
    text: str
    source: str
    score: float  # similarity in [0, 1]; higher is closer


class RAGPipeline:
    """Loads the embedder + vector store once and answers questions."""

    def __init__(self) -> None:
        self.embedder = SentenceTransformer(config.EMBEDDING_MODEL)
        client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        self.collection = client.get_or_create_collection(
            name=config.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def retrieve(self, question: str, top_k: int | None = None) -> list[Retrieved]:
        """Embed the question and fetch the most similar chunks."""
        top_k = top_k or config.TOP_K
        query_emb = self.embedder.encode([question], normalize_embeddings=True).tolist()
        res = self.collection.query(
            query_embeddings=query_emb,
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]
        # cosine distance -> similarity
        return [
            Retrieved(text=d, source=m.get("source", "?"), score=1.0 - float(dist))
            for d, m, dist in zip(docs, metas, dists)
        ]

    def build_prompt(self, question: str, chunks: list[Retrieved]) -> str:
        context = "\n\n".join(f"[{c.source}]\n{c.text}" for c in chunks)
        return f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"

    def answer(self, question: str, top_k: int | None = None) -> tuple[str, list[Retrieved]]:
        """Retrieve, then generate. Returns (answer_text, sources_used)."""
        chunks = self.retrieve(question, top_k=top_k)
        if not chunks:
            return ("The knowledge base is empty. Run `python ingest.py` first.", [])

        prompt = self.build_prompt(question, chunks)
        response = ollama.chat(
            model=config.OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            options={"temperature": config.OLLAMA_TEMPERATURE},
        )
        return response["message"]["content"], chunks


def _repl() -> None:
    print("Local RAG ready. Ask a question (Ctrl+C to quit).\n")
    pipeline = RAGPipeline()
    if pipeline.collection.count() == 0:
        print("⚠️  No chunks indexed yet. Run `python ingest.py` first.\n")
    try:
        while True:
            q = input("You: ").strip()
            if not q:
                continue
            ans, sources = pipeline.answer(q)
            print(f"\nAssistant: {ans}")
            if sources:
                names = ", ".join(sorted({s.source for s in sources}))
                print(f"  (sources: {names})\n")
    except (KeyboardInterrupt, EOFError):
        print("\nBye!")


if __name__ == "__main__":
    _repl()
