"""Task list + lightweight chat history (JSON-backed)."""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

_LOCK = threading.Lock()
_DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "tasks.json"


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"tasks": [], "chat": []}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    _ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


class TaskStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or _DEFAULT_PATH

    def add_task(self, title: str) -> dict[str, Any]:
        title = (title or "").strip()
        if not title:
            raise ValueError("empty task")
        with _LOCK:
            data = _read_json(self.path)
            tid = len(data.get("tasks", [])) + 1
            item = {"id": tid, "title": title, "done": False}
            data.setdefault("tasks", []).append(item)
            _write_json(self.path, data)
        return item

    def get_tasks(self) -> list[dict[str, Any]]:
        with _LOCK:
            data = _read_json(self.path)
            return list(data.get("tasks", []))

    def append_chat(self, role: str, content: str) -> None:
        with _LOCK:
            data = _read_json(self.path)
            data.setdefault("chat", []).append({"role": role, "content": content})
            # cap history for file size
            data["chat"] = data["chat"][-200:]
            _write_json(self.path, data)

    def recent_chat(self, n: int = 20) -> list[dict[str, str]]:
        with _LOCK:
            data = _read_json(self.path)
            return data.get("chat", [])[-n:]
