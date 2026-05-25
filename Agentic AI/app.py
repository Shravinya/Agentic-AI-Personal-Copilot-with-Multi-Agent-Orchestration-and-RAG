"""Streamlit UI: chat, PDF upload, task tools via LangGraph, debug sidebar."""
from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from config import require_api_key
from graph.langgraph_flow import run_turn
from memory.store import TaskStore
from rag.ingest import build_index_for_pdf
from utils.logger import SessionDebugLogger, use_logger
from utils.streaming import stream_final_answer

st.set_page_config(page_title="AI Task Copilot", page_icon="🧭", layout="wide")
if prompt:
if "debug_logger" not in st.session_state:
    st.session_state.debug_logger = SessionDebugLogger()
use_logger(st.session_state.debug_logger)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "task_store" not in st.session_state:
    st.session_state.task_store = TaskStore()


def _render_debug_sidebar() -> None:
    st.sidebar.header("Debug trace")
    logger = st.session_state.debug_logger
    if not logger.entries:
        st.sidebar.caption("No steps yet — send a message to populate logs.")
        return
    for i, e in enumerate(reversed(logger.entries[-80:]), 1):
        with st.sidebar.expander(f"{i}. [{e.agent}] {e.step}", expanded=False):
            st.text(f"time: {e.ts:.3f}")
            st.markdown("**Input**")
            st.code(e.input_summary or "—", language="text")
            st.markdown("**Output**")
            st.code(e.output_summary or "—", language="text")
            if e.tool:
                st.markdown("**Tool**")
                st.code(e.tool, language="text")
            if e.state_delta:
                st.markdown("**State Δ**")
                st.json(e.state_delta)


def _clear_debug() -> None:
    st.session_state.debug_logger = SessionDebugLogger()
    use_logger(st.session_state.debug_logger)


st.title("AI Personal Task Copilot")
st.caption("Gemini · LangGraph · RAG (FAISS) · ReAct tools · Streaming")

with st.sidebar:
    st.subheader("Document (RAG)")
    up = st.file_uploader("Upload PDF", type=["pdf"])
    if up and st.button("Index PDF", type="primary"):
        try:
            require_api_key()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(up.getvalue())
                tmp_path = Path(tmp.name)
            n, dim = build_index_for_pdf(tmp_path)
            tmp_path.unlink(missing_ok=True)
            st.success(f"Indexed {n} chunks (dim={dim}).")
        except Exception as ex:
            st.error(f"Indexing failed: {ex}")

    st.divider()
    if st.button("Clear debug log"):
        _clear_debug()
    _render_debug_sidebar()

try:
    require_api_key()
except Exception as e:
    st.error(str(e))
    st.stop()

prompt = st.chat_input("Ask, add tasks, or query your PDF…")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.task_store.append_chat("user", prompt)

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if prompt:
    with st.chat_message("assistant"):
        status = st.status("Running agents…", expanded=True)
        try:
            state = run_turn(prompt)
            route = state.get("route", "?")
            status.write(f"Router: **{route}**")
            if state.get("steps"):
                status.write("Planner steps: " + "; ".join(state["steps"][:6]))
            if state.get("executor_trace"):
                status.write("Executor trace recorded.")
            if state.get("rag_answer"):
                status.write("RAG draft recorded.")
            brief = state.get("synthesis_brief") or ""
            if not brief.strip():
                brief = f"Answer the user helpfully:\n{prompt}"
            text_box = st.empty()
            acc: list[str] = []

            def _stream():
                for piece in stream_final_answer(brief):
                    acc.append(piece)
                    text_box.markdown("".join(acc))

            _stream()
            final = "".join(acc).strip()
        except Exception as ex:
            status.update(label="Failed", state="error")
            st.error(str(ex))
            final = f"(error) {ex}"
        else:
            status.update(label="Done", state="complete")

    st.session_state.messages.append({"role": "assistant", "content": final})
    st.session_state.task_store.append_chat("assistant", final)
