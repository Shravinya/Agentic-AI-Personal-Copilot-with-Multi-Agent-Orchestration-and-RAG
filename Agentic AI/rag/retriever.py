"""FAISS retrieval over embedded chunks."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import faiss
import google.generativeai as genai
import numpy as np

from config import GEMINI_EMBEDDING_MODEL, require_api_key
from utils.logger import log_step

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_INDEX_PATH = _DATA_DIR / "faiss.index"
_META_PATH = _DATA_DIR / "faiss_meta.json"


@dataclass
class RetrievedChunk:
    text: str
    score: float


class FaissRetriever:
    def __init__(self) -> None:
        self._index: faiss.Index | None = None
        self._chunks: list[str] = []

    def load(self) -> bool:
        if not _INDEX_PATH.exists() or not _META_PATH.exists():
            log_step("rag:retriever", "load", {}, "missing index or meta", tool=None)
            return False
        self._index = faiss.read_index(str(_INDEX_PATH))
        with _META_PATH.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        self._chunks = list(meta.get("chunks", []))
        log_step(
            "rag:retriever",
            "load_ok",
            {},
            {"vectors": int(self._index.ntotal), "chunks": len(self._chunks)},
            tool=None,
        )
        return True

    def search(self, query: str, k: int = 4) -> list[RetrievedChunk]:
        if self._index is None and not self.load():
            return []
        genai.configure(api_key=require_api_key())
        qemb = genai.embed_content(model=GEMINI_EMBEDDING_MODEL, content=query)
        vec = np.array(qemb["embedding"], dtype="float32").reshape(1, -1)
        dists, idxs = self._index.search(vec, min(k, len(self._chunks)))
        out: list[RetrievedChunk] = []
        for dist, i in zip(dists[0], idxs[0]):
            if i < 0 or i >= len(self._chunks):
                continue
            out.append(RetrievedChunk(text=self._chunks[i], score=float(dist)))
        log_step("rag:retriever", "search", {"query": query, "k": k}, f"hits={len(out)}", tool=None)
        return out


_singleton = FaissRetriever()


def get_retriever() -> FaissRetriever:
    return _singleton
