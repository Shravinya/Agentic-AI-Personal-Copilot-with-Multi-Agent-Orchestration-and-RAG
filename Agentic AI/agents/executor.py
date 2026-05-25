"""ReAct-style executor: Thought → Action (tool) → Observation loop."""
from __future__ import annotations

import json
import re

import google.generativeai as genai

from config import GEMINI_MODEL, require_api_key
from tools.rag_tools import search_docs
from tools.task_tools import add_task, get_tasks
from utils.logger import log_step

_TOOLS = {
    "add_task": add_task,
    "get_tasks": get_tasks,
    "search_docs": search_docs,
}

_SYSTEM = """You are the Executor in a ReAct agent. You MUST respond with ONE JSON object per turn (no markdown fences).

Schema:
{{"thought":"why you are choosing this action","action":"TOOL"|"FINAL","tool_name":"add_task|get_tasks|search_docs|null","tool_input":"string argument for the tool (single string)","final":"if action is FINAL, your user-facing answer"}}

Rules:
- Use add_task with a short task title string when the user wants to add a task.
- Use get_tasks when the user wants to see tasks (tool_input can be "").
- Use search_docs with the user question or keywords when you need information from uploaded documents.
- If you can answer without tools, use action FINAL.
- Keep tool_input a single plain string (combine details into one line if needed).
"""


def _extract_json(text: str) -> dict:
    text = text.strip()
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError("no json object in model output")
    return json.loads(m.group(0))


def run_executor(user_query: str, steps: list[str] | None = None, max_turns: int = 8) -> str:
    genai.configure(api_key=require_api_key())
    model = genai.GenerativeModel(GEMINI_MODEL, generation_config={"temperature": 0.2})
    steps_hint = "\n".join(f"- {s}" for s in (steps or [])) or "(none)"
    transcript: list[str] = []
    messages = f"""User request:
{user_query}

Planner steps (hints):
{steps_hint}
"""
    log_step("executor", "start", user_query, {"steps": steps}, tool=None)

    for turn in range(max_turns):
        prompt = _SYSTEM + "\n\n" + messages + "\n\nRespond with the next JSON object only."
        log_step("executor", f"turn_{turn}_ask", messages[-1500:], "model call", tool=None)
        resp = model.generate_content(prompt)
        raw = (resp.text or "").strip()
        try:
            data = _extract_json(raw)
        except Exception as e:
            log_step("executor", f"turn_{turn}_parse_error", raw, str(e), tool=None)
            transcript.append(f"Parse error: {e}; raw={raw[:500]}")
            break

        thought = str(data.get("thought", ""))
        action = str(data.get("action", "")).upper()
        transcript.append(f"Thought: {thought}")

        if action == "FINAL":
            final = str(data.get("final", "")).strip()
            transcript.append(f"Final: {final}")
            log_step("executor", "final", user_query, final[:1200], tool=None)
            return "\n".join(transcript)

        if action != "TOOL":
            transcript.append(f"Unknown action: {action}; stopping.")
            break

        tool_name = str(data.get("tool_name", "") or "").strip()
        tool_input = str(data.get("tool_input", "") or "")
        fn = _TOOLS.get(tool_name)
        if fn is None:
            obs = f"ERROR: unknown tool {tool_name!r}"
            transcript.append(f"Action: {tool_name}({tool_input!r})\nObservation: {obs}")
            log_step("executor", f"turn_{turn}_bad_tool", tool_name, obs, tool=tool_name)
        else:
            try:
                obs = fn(tool_input)
            except Exception as e:
                obs = f"ERROR: {e}"
            transcript.append(f"Action: {tool_name}({tool_input!r})\nObservation: {obs}")
            log_step("executor", f"turn_{turn}_tool", tool_name, obs[:1200], tool=tool_name)

        messages += (
            "\n\n"
            + transcript[-1]
            + "\n\nContinue until you can respond with action FINAL and a helpful final message for the user."
        )

    log_step("executor", "max_turns", user_query, "\n".join(transcript)[-1200:], tool=None)
    return "\n".join(transcript)
