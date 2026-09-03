"""LangGraph pipeline.

    detector1_node -> detector2_node -> decision (conditional edge)
        |- flagged -> blocked_response_node -> END
        |- clean   -> llm_node -> END
"""
from __future__ import annotations

import json
import os
import time
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from . import config, llm as llm_mod
from .detectors import detector1, detector2


class GraphState(TypedDict, total=False):
    query: str
    chat_history: list          # list[BaseMessage] (Human/AI), persists across turns
    detector1_result: dict      # {label, score}
    detector2_result: dict      # {label, reason}
    is_blocked: bool
    final_response: str
    policy: str                 # "or" | "and"
    latencies: dict             # per-node seconds (observability)


# ----------------------------------------------------------------- nodes
def detector1_node(state: GraphState) -> dict:
    t0 = time.perf_counter()
    res = detector1(state["query"])
    lat = {**state.get("latencies", {}), "detector1": time.perf_counter() - t0}
    return {"detector1_result": res, "latencies": lat}


def detector2_node(state: GraphState) -> dict:
    t0 = time.perf_counter()
    res = detector2(state["query"])
    lat = {**state.get("latencies", {}), "detector2": time.perf_counter() - t0}
    return {"detector2_result": res, "latencies": lat}


def decision(state: GraphState) -> str:
    d1 = state["detector1_result"]["label"] == "injection"
    d2 = state["detector2_result"]["label"] == "injection"
    policy = state.get("policy", config.DEFAULT_POLICY)
    flagged = (d1 or d2) if policy == "or" else (d1 and d2)
    return "blocked" if flagged else "clean"


def blocked_response_node(state: GraphState) -> dict:
    _log_blocked_event(state)
    return {"is_blocked": True, "final_response": config.BLOCKED_MESSAGE}


def llm_node(state: GraphState) -> dict:
    t0 = time.perf_counter()
    model = llm_mod.get_chat_llm()
    if model is None:
        text = llm_mod.mock_reply(state["query"])
    else:
        messages = [SystemMessage(content=config.SYSTEM_PROMPT)]
        messages += list(state.get("chat_history", []))
        messages.append(HumanMessage(content=state["query"]))
        text = model.invoke(messages).content
    lat = {**state.get("latencies", {}), "llm": time.perf_counter() - t0}
    return {"is_blocked": False, "final_response": text, "latencies": lat}


def _log_blocked_event(state: GraphState) -> None:
    os.makedirs(os.path.dirname(config.LOG_PATH), exist_ok=True)
    event = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "query": state["query"],
        "detector1": state.get("detector1_result"),
        "detector2": state.get("detector2_result"),
        "policy": state.get("policy", config.DEFAULT_POLICY),
    }
    with open(config.LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


# ----------------------------------------------------------------- graph
def build_graph():
    g = StateGraph(GraphState)
    g.add_node("detector1_node", detector1_node)
    g.add_node("detector2_node", detector2_node)
    g.add_node("blocked_response_node", blocked_response_node)
    g.add_node("llm_node", llm_node)

    g.set_entry_point("detector1_node")
    g.add_edge("detector1_node", "detector2_node")
    g.add_conditional_edges("detector2_node", decision,
                            {"blocked": "blocked_response_node", "clean": "llm_node"})
    g.add_edge("blocked_response_node", END)
    g.add_edge("llm_node", END)
    return g.compile()


def run_turn(graph, query: str, chat_history: list, policy: str) -> GraphState:
    """One conversational turn. Returns final state; caller appends to history."""
    state: GraphState = {
        "query": query,
        "chat_history": chat_history,
        "policy": policy,
        "latencies": {},
    }
    result = graph.invoke(state)
    # multi-turn: persist the exchange in chat_history (blocked turns are NOT
    # added as Human/AI content so the attack text never reaches the LLM later)
    if not result["is_blocked"]:
        chat_history.append(HumanMessage(content=query))
        chat_history.append(AIMessage(content=result["final_response"]))
    return result
