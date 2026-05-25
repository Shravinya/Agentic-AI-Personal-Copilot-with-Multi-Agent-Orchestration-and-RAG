"""RAG retrieval tool for the executor (search_docs)."""
from __future__ import annotations

from rag.retriever import get_retriever
from utils.logger import log_step


def search_docs(query: str, k: int = 4) -> str:
    log_step("tool:search_docs", "invoke", {"query": query, "k": k}, "running", tool="search_docs")
    retriever = get_retriever()
    chunks = retriever.search(query, k=k)
    if not chunks:
        out = "No document chunks retrieved. Upload a PDF and wait for indexing."
    else:
        parts = []
        for i, c in enumerate(chunks, 1):
            parts.append(f"[{i}] (score={c.score:.4f})\n{c.text}")
        out = "\n\n".join(parts)
    log_step("tool:search_docs", "result", {"query": query}, out[:800], tool="search_docs")
    return out
