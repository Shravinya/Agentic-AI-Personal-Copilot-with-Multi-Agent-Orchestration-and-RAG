"""Retrieve chunks and answer with Gemini grounded on context."""
from __future__ import annotations

import google.generativeai as genai

from config import GEMINI_MODEL, require_api_key
from rag.retriever import get_retriever
from utils.logger import log_step


def answer_with_rag(user_query: str) -> tuple[str, str]:
    retriever = get_retriever()
    chunks = retriever.search(user_query, k=5)
    if not chunks:
        log_step("rag_agent", "no_index", user_query, "empty retrieval", tool=None)
        return "", "I could not find any indexed document content yet. Upload a PDF in the sidebar first."

    context = "\n\n".join(f"### Excerpt {i+1}\n{c.text}" for i, c in enumerate(chunks))
    genai.configure(api_key=require_api_key())
    model = genai.GenerativeModel(GEMINI_MODEL)
    prompt = f"""Use ONLY the provided excerpts to answer. If the excerpts are insufficient, say what is missing.

EXCERPTS:
{context}

QUESTION:
{user_query}
"""
    log_step("rag_agent", "generate_input", {"query": user_query}, f"context_chars={len(context)}", tool=None)
    resp = model.generate_content(prompt)
    answer = (resp.text or "").strip()
    log_step("rag_agent", "generate_output", user_query, answer[:800], tool=None)
    return context, answer
