"""Task management tools used by the executor agent."""
from __future__ import annotations

from memory.store import TaskStore
from utils.logger import log_step

_store = TaskStore()


def add_task(task: str) -> str:
    log_step("tool:add_task", "invoke", {"task": task}, "running", tool="add_task")
    item = _store.add_task(task)
    out = f"Added task #{item['id']}: {item['title']}"
    log_step("tool:add_task", "result", {"task": task}, out, tool="add_task")
    return out


def get_tasks() -> str:
    log_step("tool:get_tasks", "invoke", {}, "running", tool="get_tasks")
    tasks = _store.get_tasks()
    if not tasks:
        out = "No tasks yet."
    else:
        lines = [f"{t['id']}. {t['title']}" for t in tasks]
        out = "Tasks:\n" + "\n".join(lines)
    log_step("tool:get_tasks", "result", {}, out, tool="get_tasks")
    return out
