# rag.py — retrieval-augmented generation pipeline for codebase Q&A.
#
# Usage (interactive REPL):
#   python rag.py
#   python rag.py --top-k 8    # override number of retrieved chunks
#
# Programmatic usage:
#   from rag import RAGPipeline
#   pipe = RAGPipeline()
#   answer_text, sources = pipe.answer("Where is the database connection set up?")

from __future__ import annotations

import argparse
import textwrap
from typing import List, Tuple, TypedDict

import chromadb
from chromadb.config import Settings
import ollama
from sentence_transformers import SentenceTransformer

import config


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class RetrievedChunk(TypedDict):
    text: str
    path: str
    start_line: int
    end_line: int
    score: float  # cosine similarity (0-1, higher is better)


# ---------------------------------------------------------------------------
# RAG Pipeline
# ---------------------------------------------------------------------------

class RAGPipeline:
    """Embed a query, retrieve top-k chunks, and answer with Ollama."""

    # System prompt tuned for grounded code Q&A.
    SYSTEM_PROMPT = textwrap.dedent("""\
        You are a senior software engineer assistant helping a developer
        understand an unfamiliar codebase.

        Rules:
        1. Answer ONLY from the code context provided below.
        2. Cite every claim with the exact file reference shown in the context
           header, formatted as  path/to/file.py:start_line-end_line .
        3. If the context does not contain enough information to answer,
           say "I don't see that in the indexed code." — do not guess.
        4. Keep answers concise but include relevant code snippets when helpful.
        5. Never invent function names, class names, or file paths.
    """)

    def __init__(self, top_k: int = config.TOP_K) -> None:
        self.top_k = top_k

        # Load the same embedding model used during ingest
        print(f"[rag] Loading embedding model '{config.EMBEDDING_MODEL}' …")
        self.model = SentenceTransformer(config.EMBEDDING_MODEL)

        # Connect to ChromaDB
        self.client = chromadb.PersistentClient(
            path=config.CHROMA_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_collection(config.COLLECTION_NAME)
        print(
            f"[rag] Connected to collection '{config.COLLECTION_NAME}' "
            f"({self.collection.count()} vectors)."
        )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str) -> List[RetrievedChunk]:
        """Return the top-k most relevant chunks for *query*."""
        query_vector = self.model.encode(
            query,
            normalize_embeddings=True,
        ).tolist()

        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=self.top_k,
            include=["documents", "metadatas", "distances"],
        )

        chunks: List[RetrievedChunk] = []
        # ChromaDB returns nested lists (one per query); index 0 = our query
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]  # cosine distance (0=identical)

        for doc, meta, dist in zip(documents, metadatas, distances):
            chunks.append(
                RetrievedChunk(
                    text=doc,
                    path=meta["path"],
                    start_line=int(meta["start_line"]),
                    end_line=int(meta["end_line"]),
                    score=round(1.0 - dist, 4),  # distance → similarity
                )
            )

        return chunks

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def build_prompt(self, query: str, chunks: List[RetrievedChunk]) -> str:
        """Assemble a user-turn message that includes the retrieved context."""
        context_blocks: List[str] = []
        for chunk in chunks:
            header = f"# {chunk['path']}:{chunk['start_line']}-{chunk['end_line']}"
            context_blocks.append(f"{header}\n{chunk['text']}")

        context_str = "\n\n---\n\n".join(context_blocks)

        return (
            f"CODE CONTEXT:\n\n{context_str}\n\n"
            f"---\n\n"
            f"QUESTION: {query}"
        )

    # ------------------------------------------------------------------
    # Answer generation
    # ------------------------------------------------------------------

    def answer(self, query: str) -> Tuple[str, List[RetrievedChunk]]:
        """Retrieve relevant chunks and generate an answer with Ollama.

        Returns:
            answer_text: the LLM's response string
            sources: the list of RetrievedChunk dicts used as context
        """
        chunks = self.retrieve(query)
        if not chunks:
            return "No relevant code found in the index.", []

        prompt = self.build_prompt(query, chunks)

        response = ollama.chat(
            model=config.OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            options={"temperature": config.OLLAMA_TEMPERATURE},
        )

        answer_text: str = response["message"]["content"]
        return answer_text, chunks


# ---------------------------------------------------------------------------
# CLI REPL
# ---------------------------------------------------------------------------

def _print_sources(sources: List[RetrievedChunk]) -> None:
    """Pretty-print source citations."""
    print("\n--- Sources ---")
    for i, src in enumerate(sources, 1):
        print(f"  [{i}] {src['path']}:{src['start_line']}-{src['end_line']}  (score={src['score']})")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask questions about an indexed codebase.")
    parser.add_argument(
        "--top-k",
        type=int,
        default=config.TOP_K,
        help=f"Number of chunks to retrieve (default: {config.TOP_K}).",
    )
    args = parser.parse_args()

    pipeline = RAGPipeline(top_k=args.top_k)

    print("\nCodebase Q&A — type your question, or 'quit' to exit.\n")
    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not query:
            continue
        if query.lower() in {"quit", "exit", "q"}:
            print("Bye!")
            break

        answer_text, sources = pipeline.answer(query)
        print(f"\nAssistant: {answer_text}\n")
        _print_sources(sources)


if __name__ == "__main__":
    main()
