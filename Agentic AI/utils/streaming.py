"""Stream tokens from Gemini for the final user-visible answer."""
from __future__ import annotations

from typing import Iterator

import google.generativeai as genai

from config import GEMINI_MODEL, require_api_key
from utils.logger import log_step


def stream_final_answer(synthesis_brief: str) -> Iterator[str]:
    key = require_api_key()
    genai.configure(api_key=key)
    model = genai.GenerativeModel(GEMINI_MODEL)
    log_step("streaming", "start", synthesis_brief[:800], "stream=True", tool=None)
    responses = model.generate_content(synthesis_brief, stream=True)
    for chunk in responses:
        t = getattr(chunk, "text", None) or ""
        if t:
            yield t
    log_step("streaming", "end", "", "stream_complete", tool=None)
