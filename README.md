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
User (Streamlit) │ ▼ LangGraph: router ──┬── task → planner → executor (tools) ──┐ ├── rag → rag_agent (retrieve + LLM) ──┼──► synthesizer → stream final answer └── chat ───────────────────────────────┘

**Offline (sidebar):** PDF → ingest → FAISS index (`data/faiss.index`, `data/faiss_meta.json`)
## Tech stack
- **LLM & embeddings:** Google Gemini API (`google-generativeai`)
- **Orchestration:** LangGraph
- **UI:** Streamlit
- **Vector search:** FAISS (`faiss-cpu`)
- **PDF parsing:** PyMuPDF
- **Config:** `python-dotenv`
## Project structure
├── app.py # Streamlit UI ├── config.py # Environment / API settings ├── test_gemini_api.py # API connectivity check ├── agents/ │ ├── router.py # Intent classification │ ├── planner.py # Step decomposition │ ├── rag_agent.py # Document Q&A │ ├── executor.py # ReAct tool loop │ └── synthesizer.py # Merge outputs for final prompt ├── tools/ │ ├── task_tools.py # add_task, get_tasks │ └── rag_tools.py # search_docs ├── rag/ │ ├── ingest.py # PDF → chunks → embeddings → FAISS │ └── retriever.py # Semantic search ├── graph/ │ └── langgraph_flow.py # Agent graph definition ├── memory/ │ └── store.py # JSON task + chat persistence └── utils/ ├── logger.py # Debug logging └── streaming.py # Gemini streaming helper

## Prerequisites
- Python 3.10+
- [Gemini API key](https://aistudio.google.com/apikey)
## Setup
1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd "Agentic AI"
Create a virtual environment

python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
Install dependencies

pip install -r requirements.txt
Configure environment

Create a .env file in the project root:

GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-2.5-flash
GEMINI_EMBEDDING_MODEL=models/gemini-embedding-001
Never commit .env or API keys. Use .env.example with placeholder values only.

Verify the API

python test_gemini_api.py
Run the app

streamlit run app.py
Usage
Open the app in your browser (Streamlit default: http://localhost:8501).
Upload a PDF in the sidebar and click Index PDF for document Q&A.
Use the chat box for tasks, document questions, or general chat.
Open the Debug trace sidebar to inspect routing, planner steps, tool calls, and RAG steps.
Example prompts
Prompt	Expected route
Add task to study graphs
task → planner → executor
Show my tasks
task → executor
Summarize the uploaded PDF
rag
Plan my DSA prep using this document
task (may use search_docs)
Explain binary search
chat
How agents work (short)
Router — Gemini returns JSON: task | rag | chat.
Planner (task only) — Returns a list of steps as hints for the executor.
Executor (task only) — ReAct loop: model chooses tools or a final answer; Python runs tools and feeds observations back.
RAG agent (rag only) — Retrieves top chunks from FAISS, then answers with Gemini using only those excerpts.
Synthesizer — Builds one prompt from all prior outputs (no extra LLM call).
Streaming — Final user-facing reply is generated and streamed in the UI.
Data & storage
Path	Purpose
data/tasks.json
Tasks and chat history
data/faiss.index
Vector index
data/faiss_meta.json
Chunk text metadata
Security notes
Keep GEMINI_API_KEY in .env only; add .env to .gitignore.
Rotate keys if they were ever shared publicly.
Uploaded PDFs are processed locally; index files stay under data/.
Limitations
Final answer streaming runs after the LangGraph pass (agent steps are not streamed).
One FAISS index per project (re-indexing replaces the previous PDF index).
Uses the legacy google-generativeai SDK (Google recommends migrating to google.genai long term).
Future ideas
Calendar integration
Voice input
Multi-user / per-user indexes
Migrate to google.genai SDK
License
Add your license here (e.g. MIT).

Author
Your name / team — add links as needed.
