"""Centralized debug logging for agent steps (UI + stdout)."""
from __future__ import annotations

import json
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class DebugLogEntry:
    ts: float
    agent: str
    step: str
    input_summary: str
    output_summary: str
    tool: str | None
    state_delta: dict[str, Any] | None


class SessionDebugLogger:
    """Thread-safe in-memory log for the current Streamlit session."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.entries: list[DebugLogEntry] = []
        self._listeners: list[Callable[[DebugLogEntry], None]] = []

    def subscribe(self, fn: Callable[[DebugLogEntry], None]) -> None:
        with self._lock:
            self._listeners.append(fn)

    def log_step(
        self,
        agent: str,
        step: str,
        input_data: Any,
        output_data: Any,
        tool: str | None = None,
        state_delta: dict[str, Any] | None = None,
    ) -> None:
        def _short(obj: Any, limit: int = 1200) -> str:
            try:
                s = obj if isinstance(obj, str) else json.dumps(obj, default=str, ensure_ascii=False)
            except Exception:
                s = repr(obj)
            s = s.replace("\n", " ")
            return s if len(s) <= limit else s[: limit - 3] + "..."

        entry = DebugLogEntry(
            ts=time.time(),
            agent=agent,
            step=step,
            input_summary=_short(input_data),
            output_summary=_short(output_data),
            tool=tool,
            state_delta=state_delta,
        )
        line = (
            f"[{agent}] {step} | tool={tool!r} | in={entry.input_summary[:200]}... "
            f"| out={entry.output_summary[:200]}..."
        )
        print(line, file=sys.stderr)
        with self._lock:
            self.entries.append(entry)
            for fn in self._listeners:
                try:
                    fn(entry)
                except Exception:
                    pass


# Streamlit re-runs scripts; a module-level default logger is recreated each run.
_default_logger = SessionDebugLogger()


def use_logger(instance: SessionDebugLogger) -> None:
    global _default_logger
    _default_logger = instance


def get_logger() -> SessionDebugLogger:
    return _default_logger


def log_step(
    agent: str,
    step: str,
    input_data: Any,
    output_data: Any,
    tool: str | None = None,
    state_delta: dict[str, Any] | None = None,
) -> None:
    _default_logger.log_step(agent, step, input_data, output_data, tool, state_delta)
