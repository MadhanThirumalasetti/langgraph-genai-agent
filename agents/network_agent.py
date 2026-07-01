"""
Network Agent — LangGraph-based agent for network-aware AI automation.
Collects SNMP telemetry and uses LLM to detect anomalies
and trigger automated responses to network events.
"""

from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from pysnmp.hlapi import (
    getCmd, SnmpEngine, CommunityData,
    UdpTransportTarget, ContextData, ObjectType, ObjectIdentity
)
import logging

logger = logging.getLogger(__name__)


class NetworkState(TypedDict):
    host: str
    community: str
    metrics: Dict[str, str]
    anomalies: List[str]
    action: str


def collect_snmp_metrics(state: NetworkState) -> NetworkState:
    """Collect network metrics via SNMP."""
    oids = {
        "sysDescr": "1.3.6.1.2.1.1.1.0",
        "sysUpTime": "1.3.6.1.2.1.1.3.0",
        "ifInOctets": "1.3.6.1.2.1.2.2.1.10.1",
        "ifOutOctets": "1.3.6.1.2.1.2.2.1.16.1",
    }
    metrics = {}
    for name, oid in oids.items():
        try:
            error_indication, error_status, _, var_binds = next(
                getCmd(
                    SnmpEngine(),
                    CommunityData(state["community"]),
                    UdpTransportTarget((state["host"], 161)),
                    ContextData(),
                    ObjectType(ObjectIdentity(oid))
                )
            )
            if not error_indication and not error_status:
                metrics[name] = str(var_binds[0][1])
        except Exception as e:
            logger.warning(f"SNMP collection failed for {name}: {e}")
            metrics[name] = "unavailable"
    state["metrics"] = metrics
    return state


def detect_anomalies(state: NetworkState) -> NetworkState:
    """Use LLM to analyze metrics and detect anomalies."""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    metrics_text = "\n".join([f"{k}: {v}" for k, v in state["metrics"].items()])
    prompt = f"""Analyze these network metrics and identify anomalies:

{metrics_text}

List any anomalies detected. If none, respond with 'No anomalies detected.'"""
    response = llm.invoke(prompt)
    anomalies = response.content.strip().split("\n")
    state["anomalies"] = [a for a in anomalies if a.strip()]
    return state


def determine_action(state: NetworkState) -> NetworkState:
    """Determine automated response action based on anomalies."""
    if not state["anomalies"] or state["anomalies"] == ["No anomalies detected."]:
        state["action"] = "no_action"
    else:
        state["action"] = "alert_ops_team"
    return state


def route_by_anomaly(state: NetworkState) -> str:
    """Route workflow based on anomaly detection results."""
    return state["action"]


def build_network_workflow() -> StateGraph:
    """Build LangGraph network monitoring workflow."""
    workflow = StateGraph(NetworkState)
    workflow.add_node("collect_metrics", collect_snmp_metrics)
    workflow.add_node("detect_anomalies", detect_anomalies)
    workflow.add_node("determine_action", determine_action)
    workflow.set_entry_point("collect_metrics")
    workflow.add_edge("collect_metrics", "detect_anomalies")
    workflow.add_edge("detect_anomalies", "determine_action")
    workflow.add_conditional_edges(
        "determine_action",
        route_by_anomaly,
        {"no_action": END, "alert_ops_team": END}
    )
    return workflow.compile()


if __name__ == "__main__":
    app = build_network_workflow()
    result = app.invoke({
        "host": "192.168.1.1",
        "community": "public",
        "metrics": {},
        "anomalies": [],
        "action": ""
    })
    print(f"Action: {result['action']}")
    print(f"Anomalies: {result['anomalies']}")
