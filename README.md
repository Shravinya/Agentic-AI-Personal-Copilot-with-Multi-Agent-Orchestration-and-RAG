# AI Personal Task Copilot – Multi-Agent Workflow & RAG System

## Overview

AI Personal Task Copilot is an end-to-end Agentic AI system built using Google Gemini, LangGraph, FAISS, and Streamlit. The application demonstrates a complete multi-agent workflow where user requests are intelligently routed through specialized agents for planning, task execution, document retrieval, and response synthesis.

The system combines conversational AI, task automation, Retrieval-Augmented Generation (RAG), tool-calling agents, and workflow orchestration into a single interactive application. It also provides full execution transparency through a debug panel that visualizes agent decisions, tool invocations, and workflow transitions.

---

## Key Features

### Multi-Agent Orchestration
- Routes user requests through specialized agents using LangGraph.
- Supports dynamic workflow execution based on query intent.
- Enables planning, reasoning, retrieval, and synthesis across agents.

### Intelligent Task Management
- Create and manage personal tasks using tool-calling agents.
- ReAct-based execution loop for autonomous decision-making.
- Persistent task storage and retrieval.

### Retrieval-Augmented Generation (RAG)
- Upload PDF documents and create searchable knowledge bases.
- Uses Gemini Embeddings and FAISS vector search.
- Provides context-aware answers grounded in uploaded documents.

### Conversational AI
- Handles general-purpose chat and reasoning tasks.
- Context-aware interactions powered by Google Gemini.

### Workflow Observability
- Real-time debugging sidebar.
- Displays routing decisions, agent execution steps, tool calls, and outputs.
- Improves explainability and transparency of AI workflows.

### Streaming Responses
- Token-by-token response generation.
- Provides a smooth real-time conversational experience.

---

## System Architecture

```text
User (Streamlit UI)
        │
        ▼
   Router Agent
        │
 ┌──────┼─────────┐
 │      │         │
 ▼      ▼         ▼
Task   RAG      Chat
Agent Agent    Agent
 │      │
 ▼      ▼
Planner Retrieval
 │
 ▼
Executor (Tools)
 │
 └───────────────┐
                 ▼
          Synthesizer
                 │
                 ▼
       Streamed Final Response
```

---

## Agent Workflow

### 1. Router Agent
- Classifies user intent into:
  - Task
  - RAG
  - General Chat

### 2. Planner Agent
- Breaks complex task requests into executable steps.
- Generates structured plans for downstream execution.

### 3. Executor Agent
- Uses a ReAct (Reason + Act) workflow.
- Invokes tools such as:
  - Add Task
  - List Tasks
  - Search Documents

### 4. RAG Agent
- Retrieves relevant document chunks from FAISS.
- Generates responses grounded in uploaded PDFs.

### 5. Synthesizer Agent
- Combines outputs from all agents.
- Produces a coherent final response.

### 6. Streaming Layer
- Streams the final response token-by-token to the UI.

---

## Tech Stack

### AI & Agent Frameworks
- LangGraph
- LangChain
- ReAct Agents

### LLM & Embeddings
- Google Gemini 2.5 Flash
- Gemini Embeddings

### Retrieval & Vector Search
- FAISS
- Retrieval-Augmented Generation (RAG)

### Frontend
- Streamlit

### Backend
- Python

### Document Processing
- PyMuPDF

### Configuration & Utilities
- Python Dotenv
- JSON Storage

---

## Project Structure

```text
AI-Personal-Task-Copilot/
│
├── app.py
├── config.py
├── test_gemini_api.py
│
├── agents/
│   ├── router.py
│   ├── planner.py
│   ├── executor.py
│   ├── rag_agent.py
│   └── synthesizer.py
│
├── graph/
│   └── langgraph_flow.py
│
├── rag/
│   ├── ingest.py
│   └── retriever.py
│
├── tools/
│   ├── task_tools.py
│   └── rag_tools.py
│
├── memory/
│   └── store.py
│
├── utils/
│   ├── logger.py
│   └── streaming.py
│
└── data/
    ├── tasks.json
    ├── faiss.index
    └── faiss_meta.json
```

---

## Installation

### Clone Repository

```bash
git clone <repository-url>
cd AI-Personal-Task-Copilot
```

### Create Virtual Environment

```bash
python -m venv .venv
```

### Activate Environment

#### Windows

```bash
.venv\Scripts\activate
```

#### Linux / macOS

```bash
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Configuration

Create a `.env` file:

```env
GEMINI_API_KEY=YOUR_API_KEY
GEMINI_MODEL=gemini-2.5-flash
GEMINI_EMBEDDING_MODEL=models/gemini-embedding-001
```

---

## Run Application

```bash
streamlit run app.py
```

---

## Example Prompts

### Task Management

```text
Add a task to complete DSA practice
```

```text
Show all my pending tasks
```

### RAG Queries

```text
Summarize the uploaded document
```

```text
What are the key findings mentioned in the PDF?
```

### General Chat

```text
Explain Binary Search
```

```text
How does LangGraph work?
```

---

## Resume Description

### Project Title
**Agentic AI Personal Copilot with Multi-Agent Orchestration and RAG**

### Resume Points

- Built an end-to-end multi-agent AI system using LangGraph for dynamic routing, planning, tool execution, and response synthesis across conversational, task automation, and document intelligence workflows.
- Developed a Retrieval-Augmented Generation (RAG) pipeline using Gemini Embeddings and FAISS vector search, enabling accurate PDF-based question answering with contextual retrieval and grounded reasoning.
- Integrated ReAct-based tool-calling agents, real-time response streaming, and workflow observability dashboards to improve transparency, explainability, and user experience.

### Tech Stack

```text
Python, LangGraph, LangChain, Google Gemini, Gemini Embeddings,
FAISS, Streamlit, Retrieval-Augmented Generation (RAG),
ReAct Agents, Multi-Agent Systems, Vector Search,
PyMuPDF, JSON Storage
```

---

## Future Enhancements

- Calendar Integration
- Voice-Based Interaction
- Multi-User Support
- Persistent Memory Layer
- MCP Integration
- Cloud Deployment
- Agent Evaluation Framework
- Long-Term Memory Management
- Migration to Google GenAI SDK

---

## License

```text
MIT License
```

---

## Author

**Gadeela Shravinya**

AI Agentic Engineer | LangGraph | LangChain | Multi-Agent Systems | RAG | Python
