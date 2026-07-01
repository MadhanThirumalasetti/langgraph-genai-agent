"""
Automation Workflow — LangGraph stateful workflow combining
RAG retrieval and network monitoring into a unified
AI-powered automation pipeline.
"""

from typing import TypedDict, List, Dict, Literal
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
import logging

logger = logging.getLogger(__name__)


class AutomationState(TypedDict):
    query: str
    network_metrics: Dict[str, str]
    context: List[str]
    intent: Literal["network", "knowledge", "unknown"]
    response: str
    action_taken: str


def classify_intent(state: AutomationState) -> AutomationState:
    """Classify user query intent using LLM."""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    prompt = f"""Classify this query into one of: network, knowledge, unknown.

Query: {state["query"]}

Respond with only one word: network, knowledge, or unknown."""
    response = llm.invoke(prompt)
    intent = response.content.strip().lower()
    state["intent"] = intent if intent in ["network", "knowledge"] else "unknown"
    logger.info(f"Intent classified as: {state['intent']}")
    return state


def handle_network_query(state: AutomationState) -> AutomationState:
    """Handle network-related queries using SNMP metrics."""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    metrics_text = "\n".join([
        f"{k}: {v}" for k, v in state["network_metrics"].items()
    ])
    prompt = f"""Based on these network metrics, answer the query:

Metrics:
{metrics_text}

Query: {state["query"]}"""
    response = llm.invoke(prompt)
    state["response"] = response.content
    state["action_taken"] = "network_analysis"
    return state


def handle_knowledge_query(state: AutomationState) -> AutomationState:
    """Handle knowledge queries using RAG retrieval."""
    context_text = "\n\n".join(state["context"]) if state["context"] else "No context available."
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    prompt = f"""Answer using this context:

{context_text}

Query: {state["query"]}"""
    response = llm.invoke(prompt)
    state["response"] = response.content
    state["action_taken"] = "rag_retrieval"
    return state


def handle_unknown_query(state: AutomationState) -> AutomationState:
    """Handle unknown intent queries with general LLM response."""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    response = llm.invoke(state["query"])
    state["response"] = response.content
    state["action_taken"] = "general_llm"
    return state


def route_by_intent(state: AutomationState) -> str:
    """Route workflow based on classified intent."""
    return state["intent"]


def build_automation_workflow() -> StateGraph:
    """Build complete LangGraph automation workflow."""
    workflow = StateGraph(AutomationState)
    workflow.add_node("classify_intent", classify_intent)
    workflow.add_node("handle_network", handle_network_query)
    workflow.add_node("handle_knowledge", handle_knowledge_query)
    workflow.add_node("handle_unknown", handle_unknown_query)
    workflow.set_entry_point("classify_intent")
    workflow.add_conditional_edges(
        "classify_intent",
        route_by_intent,
        {
            "network": "handle_network",
            "knowledge": "handle_knowledge",
            "unknown": "handle_unknown"
        }
    )
    workflow.add_edge("handle_network", END)
    workflow.add_edge("handle_knowledge", END)
    workflow.add_edge("handle_unknown", END)
    return workflow.compile()


if __name__ == "__main__":
    app = build_automation_workflow()
    result = app.invoke({
        "query": "Is there any BGP routing issue on the network?",
        "network_metrics": {
            "bgp_peers": "3 active",
            "ospf_neighbors": "5 active",
            "interface_errors": "2 errors detected"
        },
        "context": [],
        "intent": "",
        "response": "",
        "action_taken": ""
    })
    print(f"Response: {result['response']}")
    print(f"Action: {result['action_taken']}")
