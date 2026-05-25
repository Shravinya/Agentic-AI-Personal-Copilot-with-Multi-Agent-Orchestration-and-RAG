"""PDF text extraction, chunking, Gemini embeddings, FAISS index."""
from __future__ import annotations

import json
import re
from pathlib import Path

import faiss
import fitz  # pymupdf
import google.generativeai as genai
import numpy as np

from config import GEMINI_EMBEDDING_MODEL, require_api_key
from utils.logger import log_step

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_INDEX_PATH = _DATA_DIR / "faiss.index"
_META_PATH = _DATA_DIR / "faiss_meta.json"


def extract_pdf_text(path: Path) -> str:
    log_step("rag:ingest", "extract_pdf", {"path": str(path)}, "opening", tool=None)
    doc = fitz.open(path)
    parts: list[str] = []
    for page in doc:
        parts.append(page.get_text())
    text = "\n".join(parts).strip()
    log_step("rag:ingest", "extract_pdf_done", {"pages": len(doc)}, f"chars={len(text)}", tool=None)
    doc.close()
    return text


def chunk_text(text: str, max_chars: int = 1200, overlap: int = 150) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        chunk = text[start:end]
        chunks.append(chunk.strip())
        if end >= len(text):
            break
        start = max(0, end - overlap)
    log_step("rag:ingest", "chunk", {"n": len(chunks)}, f"first_len={len(chunks[0]) if chunks else 0}", tool=None)
    return chunks


def embed_batch(texts: list[str]) -> np.ndarray:
    key = require_api_key()
    genai.configure(api_key=key)
    log_step("rag:ingest", "embed_start", {"batch": len(texts)}, "calling Gemini embeddings", tool=None)
    vectors: list[list[float]] = []
    for t in texts:
        result = genai.embed_content(model=GEMINI_EMBEDDING_MODEL, content=t)
        emb = result.get("embedding")
        if emb is None:
            raise RuntimeError("No embedding in response")
        vectors.append(emb)
    arr = np.array(vectors, dtype="float32")
    log_step("rag:ingest", "embed_done", {"shape": list(arr.shape)}, "ok", tool=None)
    return arr


def build_index_for_pdf(pdf_path: Path) -> tuple[int, int]:
    """Returns (num_chunks, dim)."""
    text = extract_pdf_text(pdf_path)
    chunks = chunk_text(text)
    if not chunks:
        log_step("rag:ingest", "empty_pdf", {}, "no text extracted", tool=None)
        return 0, 0
    vectors = embed_batch(chunks)
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(_INDEX_PATH))

    meta = {"chunks": chunks, "source": str(pdf_path.name)}
    with _META_PATH.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)
    log_step("rag:ingest", "index_saved", {"chunks": len(chunks), "dim": dim}, str(_INDEX_PATH), tool=None)
    return len(chunks), dim
