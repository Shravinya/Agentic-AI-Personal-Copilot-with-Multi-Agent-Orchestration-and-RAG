"""LangGraph orchestration: router → (planner→executor | rag) → synthesizer."""
from __future__ import annotations

from typing import Literal, TypedDict

from langgraph.graph import END, StateGraph

from agents.executor import run_executor
from agents.planner import plan_steps
from agents.rag_agent import answer_with_rag
from agents.router import classify_query
from agents.synthesizer import build_synthesis_brief
from utils.logger import log_step


class AgentState(TypedDict, total=False):
    query: str
    route: str
    steps: list[str]
    executor_trace: str
    rag_context: str
    rag_answer: str
    synthesis_brief: str


def _node_router(state: AgentState) -> AgentState:
    q = state.get("query", "")
    log_step("graph", "node_router_in", q, "enter", tool=None)
    route = classify_query(q)
    out = {**state, "route": route}
    log_step("graph", "node_router_out", q, route, tool=None, state_delta={"route": route})
    return out


def _node_planner(state: AgentState) -> AgentState:
    q = state.get("query", "")
    log_step("graph", "node_planner_in", q, "enter", tool=None)
    steps = plan_steps(q)
    out = {**state, "steps": steps}
    log_step("graph", "node_planner_out", q, {"steps": steps}, tool=None)
    return out


def _node_executor(state: AgentState) -> AgentState:
    q = state.get("query", "")
    steps = state.get("steps") or []
    log_step("graph", "node_executor_in", q, {"steps": steps}, "enter", tool=None)
    trace = run_executor(q, steps)
    out = {**state, "executor_trace": trace}
    log_step("graph", "node_executor_out", q, trace[:800], tool=None)
    return out


def _node_rag(state: AgentState) -> AgentState:
    q = state.get("query", "")
    log_step("graph", "node_rag_in", q, "enter", tool=None)
    ctx, ans = answer_with_rag(q)
    out = {**state, "rag_context": ctx, "rag_answer": ans}
    log_step("graph", "node_rag_out", q, ans[:800], tool=None)
    return out


def _node_synthesizer(state: AgentState) -> AgentState:
    q = state.get("query", "")
    route = state.get("route") or "chat"
    brief = build_synthesis_brief(
        query=q,
        route=route,
        steps=state.get("steps"),
        executor_trace=state.get("executor_trace"),
        rag_context=state.get("rag_context"),
        rag_answer=state.get("rag_answer"),
    )
    out = {**state, "synthesis_brief": brief}
    log_step("graph", "node_synthesizer_out", q, brief[:800], tool=None)
    return out


def _after_router(state: AgentState) -> Literal["planner", "rag_agent", "synthesizer"]:
    r = (state.get("route") or "chat").lower()
    if r == "task":
        return "planner"
    if r == "rag":
        return "rag_agent"
    return "synthesizer"


def build_app_graph():
    g = StateGraph(AgentState)
    g.add_node("router", _node_router)
    g.add_node("planner", _node_planner)
    g.add_node("executor", _node_executor)
    g.add_node("rag_agent", _node_rag)
    g.add_node("synthesizer", _node_synthesizer)

    g.set_entry_point("router")
    g.add_conditional_edges(
        "router",
        _after_router,
        {
            "planner": "planner",
            "rag_agent": "rag_agent",
            "synthesizer": "synthesizer",
        },
    )
    g.add_edge("planner", "executor")
    g.add_edge("executor", "synthesizer")
    g.add_edge("rag_agent", "synthesizer")
    g.add_edge("synthesizer", END)
    return g.compile()


_compiled = None


def get_compiled_graph():
    global _compiled
    if _compiled is None:
        _compiled = build_app_graph()
    return _compiled


def run_turn(query: str) -> AgentState:
    log_step("graph", "invoke_start", query, "compile+invoke", tool=None)
    graph = get_compiled_graph()
    result = graph.invoke({"query": query})
    log_step("graph", "invoke_end", query, {"route": result.get("route")}, tool=None)
    return result
