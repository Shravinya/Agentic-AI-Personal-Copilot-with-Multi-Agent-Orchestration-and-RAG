# AI Personal Task Copilot

An end-to-end **agentic AI assistant** built with **Google Gemini**, **LangGraph**, and **Streamlit**. The system routes user messages through specialized agents, uses tools for task management, supports **RAG** over uploaded PDFs, and exposes a **debug trace** so you can see how each step runs.

## Overview

This project is a learning-oriented **multi-agent copilot** that can:

- **Plan and execute** task-related requests (add/list tasks via tools)
- **Answer questions** from uploaded documents (chunk → embed → FAISS retrieval)
- **Handle general chat** without documents or tools
- **Stream** the final response in the UI
- **Log every step** (agent, input/output, tools, state) in a sidebar debug panel

The design follows a clear pipeline: **Router → (Planner → Executor | RAG Agent | Chat) → Synthesizer → Streamed reply**.

## Features

| Feature | Description |
|--------|-------------|
| **Multi-agent routing** | Router classifies input as `task`, `rag`, or `chat` |
| **Planner** | Breaks complex task requests into ordered steps |
| **ReAct executor** | Thought → Action → Observation loop with `add_task`, `get_tasks`, `search_docs` |
| **RAG pipeline** | PDF upload, text extraction, chunking, Gemini embeddings, FAISS search |
| **LangGraph orchestration** | Shared state and conditional edges between agents |
| **Streaming UI** | Token-by-token final answer via Gemini streaming API |
| **Debug visibility** | Sidebar shows agent steps, tool calls, and state updates |
| **Persistent tasks** | Tasks and chat history stored in `data/tasks.json` |

## Architecture
