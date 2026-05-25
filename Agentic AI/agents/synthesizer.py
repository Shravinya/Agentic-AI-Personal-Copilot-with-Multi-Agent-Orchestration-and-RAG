"""Build the final prompt brief for streaming (merges agent outputs)."""
from __future__ import annotations

from utils.logger import log_step


def build_synthesis_brief(
    query: str,
    route: str,
    steps: list[str] | None,
    executor_trace: str | None,
    rag_context: str | None,
    rag_answer: str | None,
) -> str:
    parts = [
        "You are the user's personal task copilot.",
        "Write a clear, helpful final message.",
        f"Original user message: {query}",
        f"Router label: {route}",
    ]
    if steps:
        parts.append("Planner steps:\n- " + "\n- ".join(steps))
    if executor_trace:
        parts.append("Executor trace (Thought/Action/Observation):\n" + executor_trace)
    if rag_answer:
        parts.append("Document-grounded draft answer:\n" + rag_answer)
    if rag_context and route == "rag":
        parts.append(
            "If needed, you may lightly polish the draft; do not invent facts beyond excerpts + draft."
        )
    brief = "\n\n".join(parts)
    log_step("synthesizer", "brief_built", query, brief[:1500], tool=None)
    return brief
