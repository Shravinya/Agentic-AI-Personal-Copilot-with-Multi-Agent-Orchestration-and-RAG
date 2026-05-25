"""Break a user goal into ordered steps (task-focused)."""
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


def plan_steps(user_text: str) -> list[str]:
    genai.configure(api_key=require_api_key())
    model = genai.GenerativeModel(
        GEMINI_MODEL,
        generation_config={
            "response_mime_type": "application/json",
            "temperature": 0.3,
        },
    )
    prompt = f"""You are a planner for a personal task copilot with tools: add_task, get_tasks, search_docs.

Decompose the user's request into 3-8 concrete steps. Prefer steps that map to tool use when needed.

User request:
{user_text}

Return JSON: {{"steps":["..."]}}"""
    log_step("planner", "plan_input", user_text, "calling model", tool=None)
    resp = model.generate_content(prompt)
    data = _extract_json(resp.text or "")
    steps = data.get("steps") or []
    if not isinstance(steps, list):
        steps = []
    steps = [str(s).strip() for s in steps if str(s).strip()]
    log_step("planner", "plan_output", user_text, {"steps": steps}, tool=None, state_delta={"steps": steps})
    return steps
