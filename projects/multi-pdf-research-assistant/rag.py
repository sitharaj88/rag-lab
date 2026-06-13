"""
rag.py — Retrieval-Augmented Generation pipeline.

Public API
----------
RAGPipeline
    .retrieve(query, top_k)   -> list[RetrievedChunk]
    .build_prompt(query, chunks) -> str
    .answer(query)            -> tuple[str, list[RetrievedChunk]]

Run as a script for an interactive REPL:
    python rag.py
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass

import chromadb
import ollama
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model for retrieval results
# ---------------------------------------------------------------------------


@dataclass
class RetrievedChunk:
    """A single chunk returned from the vector store, with its provenance.

    Attributes
    ----------
    text:         The raw chunk text.
    source:       Filename of the originating PDF.
    page:         1-based page number within that PDF.
    score:        Similarity score in [0, 1] — computed as 1 - cosine_distance.
    chunk_index:  0-based chunk position on its page (for deduplication).
    """

    text: str
    source: str
    page: int
    score: float
    chunk_index: int


# ---------------------------------------------------------------------------
# RAGPipeline
# ---------------------------------------------------------------------------


class RAGPipeline:
    """End-to-end RAG pipeline: embed query → retrieve → LLM → answer.

    Parameters
    ----------
    embedding_model:
        Name of the sentence-transformers model used to embed queries.
        Must match the model used during ingest.
    ollama_model:
        Ollama model tag for answer generation (e.g. "llama3.2").
    collection_name:
        ChromaDB collection to query.
    chroma_dir:
        Path to the ChromaDB persistence directory.
    top_k:
        Default number of chunks to retrieve per query.
    temperature:
        Sampling temperature for the Ollama LLM.
    """

    def __init__(
        self,
        embedding_model: str = config.EMBEDDING_MODEL,
        ollama_model: str = config.OLLAMA_MODEL,
        collection_name: str = config.COLLECTION_NAME,
        chroma_dir: str = str(config.CHROMA_DIR),
        top_k: int = config.TOP_K,
        temperature: float = config.OLLAMA_TEMPERATURE,
    ) -> None:
        self.ollama_model = ollama_model
        self.top_k = top_k
        self.temperature = temperature

        logger.info("Loading embedding model '%s' …", embedding_model)
        self._embedder = SentenceTransformer(embedding_model)

        logger.info("Connecting to ChromaDB at '%s' …", chroma_dir)
        client = chromadb.PersistentClient(
            path=chroma_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = client.get_collection(collection_name)
        logger.info(
            "Connected.  Collection '%s' holds %d document(s).",
            collection_name,
            self._collection.count(),
        )

    # ------------------------------------------------------------------
    # Retrieve
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """Embed *query* and return the top-k most similar chunks.

        Parameters
        ----------
        query:
            The user's natural-language question.
        top_k:
            How many chunks to return.  Defaults to self.top_k.

        Returns
        -------
        list[RetrievedChunk]
            Chunks ordered by descending similarity score.
        """
        if top_k is None:
            top_k = self.top_k

        # Embed the query with the same settings used at ingest time.
        query_vector = (
            self._embedder.encode([query], normalize_embeddings=True)[0].tolist()
        )

        results = self._collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        chunks: list[RetrievedChunk] = []
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for doc, meta, dist in zip(documents, metadatas, distances):
            # ChromaDB returns cosine *distance* in [0, 2].  Convert to a
            # similarity score in [0, 1]: score = 1 - (distance / 2).
            score = max(0.0, 1.0 - dist / 2.0)
            chunks.append(
                RetrievedChunk(
                    text=doc,
                    source=meta["source"],
                    page=int(meta["page"]),
                    score=score,
                    chunk_index=int(meta["chunk_index"]),
                )
            )

        return chunks

    # ------------------------------------------------------------------
    # Prompt builder
    # ------------------------------------------------------------------

    @staticmethod
    def build_prompt(query: str, chunks: list[RetrievedChunk]) -> str:
        """Assemble the user-turn prompt that is sent to the LLM.

        Each chunk is prefixed with a citation tag so the model can refer to
        specific sources in its answer.

        Parameters
        ----------
        query:
            The user's question.
        chunks:
            Ranked list of retrieved context chunks.

        Returns
        -------
        str
            The full prompt string.
        """
        context_parts: list[str] = []
        for i, chunk in enumerate(chunks, start=1):
            tag = f"[{chunk.source} p.{chunk.page}]"
            context_parts.append(f"--- Context {i} {tag} ---\n{chunk.text}")

        context_block = "\n\n".join(context_parts)

        prompt = (
            f"The following excerpts are retrieved from the research documents.\n\n"
            f"{context_block}\n\n"
            f"Question: {query}"
        )
        return prompt

    # ------------------------------------------------------------------
    # Answer
    # ------------------------------------------------------------------

    def answer(self, query: str, top_k: int | None = None) -> tuple[str, list[RetrievedChunk]]:
        """Run the full RAG pipeline and return an answer with sources.

        Parameters
        ----------
        query:
            The user's natural-language question.
        top_k:
            Number of chunks to retrieve.  Defaults to self.top_k.

        Returns
        -------
        tuple[str, list[RetrievedChunk]]
            A 2-tuple of (answer_text, retrieved_chunks).
            The caller can use the chunks to render citation UI.
        """
        chunks = self.retrieve(query, top_k=top_k)

        if not chunks:
            return (
                "I could not find any relevant information in the loaded documents.",
                [],
            )

        user_prompt = self.build_prompt(query, chunks)

        system_prompt = (
            "You are a precise research assistant that answers questions ONLY "
            "using the context excerpts provided.  "
            "If the answer is not present in the context, say: "
            "'I don't have enough information in the provided documents to answer that.'  "
            "Never fabricate facts.  "
            "Cite the source of every claim using the format [filename.pdf p.N] "
            "exactly as it appears in the context tags."
        )

        response = ollama.chat(
            model=self.ollama_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            options={"temperature": self.temperature},
        )

        answer_text: str = response["message"]["content"]
        return answer_text, chunks


# ---------------------------------------------------------------------------
# Interactive REPL
# ---------------------------------------------------------------------------


def _repl() -> None:
    """Simple read-eval-print loop for quick command-line testing."""
    print("\n=== Multi-PDF Research Assistant — CLI REPL ===")
    print("Type your question and press Enter.  Type 'quit' or 'exit' to stop.\n")

    try:
        pipeline = RAGPipeline()
    except Exception as exc:
        print(
            f"ERROR: Could not initialise the pipeline.\n"
            f"  Make sure you have run `python ingest.py` first.\n"
            f"  Details: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    while True:
        try:
            query = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not query:
            continue
        if query.lower() in {"quit", "exit", "q"}:
            print("Goodbye.")
            break

        print("\nThinking …\n")
        answer_text, sources = pipeline.answer(query)

        print("Answer:")
        print(answer_text)

        if sources:
            print("\nSources used:")
            seen: set[str] = set()
            for chunk in sources:
                key = f"{chunk.source} (p.{chunk.page})"
                if key not in seen:
                    seen.add(key)
                    print(f"  • {key}  — similarity {chunk.score:.2f}")
        print()


if __name__ == "__main__":
    _repl()
