"""
graph.py – Compiles the LangGraph StateGraph for the AI Study Coach.
Import `coach_app` in Streamlit to run the agent.
"""

from langgraph.graph import StateGraph, END

from .state import AgentState
from .nodes import (
    master_node,
    analyser_node,
    retriever_node,
    planner_node,
    quizzer_node,
    end_node,
)

# ─────────────────────────────────────────────
# BUILD THE GRAPH
# ─────────────────────────────────────────────

def _build_graph() -> StateGraph:
    workflow = StateGraph(AgentState)

    # Register nodes
    workflow.add_node("supervisor", master_node)
    workflow.add_node("analyser",   analyser_node)
    workflow.add_node("retriever",  retriever_node)
    workflow.add_node("planner",    planner_node)
    workflow.add_node("quizzer",    quizzer_node)
    workflow.add_node("end",        end_node)

    # Entry point
    workflow.set_entry_point("supervisor")

    # Supervisor routes dynamically based on state["next_node"]
    workflow.add_conditional_edges(
        "supervisor",
        lambda x: x["next_node"],
        {
            "analyser":  "analyser",
            "retriever": "retriever",
            "planner":   "planner",
            "quizzer":   "quizzer",
            "end":       "end",
        },
    )

    # Most specialist nodes return to supervisor so a multi-step plan can continue.
    workflow.add_edge("analyser",  "supervisor")
    workflow.add_edge("retriever", "supervisor")
    workflow.add_edge("planner",   "supervisor")

    # send quiz output through end so mixed requests can be shown together.
    workflow.add_edge("quizzer", "end")

    # End node terminates the graph
    workflow.add_edge("end", END)

    return workflow.compile()


# ─────────────────────────────────────────────
# COMPILED APP  (import this in app.py)
# ─────────────────────────────────────────────
coach_app = _build_graph()
print("[graph] AI Study Coach graph compiled successfully.")
