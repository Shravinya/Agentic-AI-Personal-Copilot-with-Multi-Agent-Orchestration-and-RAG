"""Route user message to task / rag / chat."""
from __future__ import annotations

import json
import re

import google.generativeai as genai

from config import GEMINI_MODEL, require_api_key
from utils.logger import log_step


def _extract_json(text: str) -> dict:
    text = text.strip()
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError("no json object in model output")
    return json.loads(m.group(0))


def classify_query(user_text: str) -> str:
    genai.configure(api_key=require_api_key())
    model = genai.GenerativeModel(
        GEMINI_MODEL,
        generation_config={
            "response_mime_type": "application/json",
            "temperature": 0.2,
        },
    )
    prompt = f"""Classify the user message into exactly one category:
- "task" — adding/listing/managing tasks, reminders, to-dos, planning execution steps for the user's workload
- "rag" — questions that should be answered using the user's uploaded documents (summaries, "according to the PDF", "this document", etc.)
- "chat" — general knowledge or conversation not requiring the private document corpus

User message:
{user_text}

Return JSON: {{"type":"task"|"rag"|"chat","reason":"short"}}"""
    log_step("router", "classify_input", user_text, "calling model", tool=None)
    resp = model.generate_content(prompt)
    raw = (resp.text or "").strip()
    data = _extract_json(raw)
    route = str(data.get("type", "chat")).lower().strip()
    if route not in ("task", "rag", "chat"):
        route = "chat"
    log_step("router", "classify_output", user_text, data, tool=None, state_delta={"route": route})
    return route
