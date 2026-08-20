"""LangGraph workflow skeleton."""

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.supervisor import SupervisorAgent
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.state import ResearchState


class _GraphState(TypedDict):
    """Internal LangGraph state: wraps the single source of truth, `ResearchState`.

    Kept as a TypedDict (rather than passing ResearchState directly as the graph
    schema) so this doesn't depend on which Pydantic-schema behavior a given
    LangGraph version supports -- TypedDict + default overwrite-on-write is the
    most stable part of the LangGraph API. `research_state` is replaced wholesale
    by each node; that's correct here because exactly one node runs per step
    (no parallel branches writing to it concurrently).
    """

    research_state: ResearchState


def _supervisor_node(graph_state: _GraphState) -> _GraphState:
    state = SupervisorAgent().run(graph_state["research_state"])
    return {"research_state": state}


def _researcher_node(graph_state: _GraphState) -> _GraphState:
    state = ResearcherAgent().run(graph_state["research_state"])
    return {"research_state": state}


def _analyst_node(graph_state: _GraphState) -> _GraphState:
    state = AnalystAgent().run(graph_state["research_state"])
    return {"research_state": state}


def _writer_node(graph_state: _GraphState) -> _GraphState:
    state = WriterAgent().run(graph_state["research_state"])
    return {"research_state": state}


def _route_after_supervisor(graph_state: _GraphState) -> str:
    """Read the route Supervisor just recorded and send the graph there next."""

    route_history = graph_state["research_state"].route_history
    return route_history[-1] if route_history else "done"


class MultiAgentWorkflow:
    """Builds and runs the multi-agent graph.

    Keep orchestration here; keep agent internals in `agents/`. Agents are only
    instantiated when their node actually runs (not in `build()`), so building the
    graph never requires OPENAI_API_KEY -- only `run()` does.
    """

    def build(self) -> Any:
        """Create the LangGraph graph: Supervisor routes to a worker, every worker
        hands control back to Supervisor, which decides what's next (including
        stopping) -- see `SupervisorAgent` for the routing policy and guardrails.

        Return type is `Any` rather than LangGraph's precise `CompiledStateGraph[...]`
        generic on purpose: that type isn't meant to be spelled out by callers, and
        mypy --strict's overload resolution for `add_node` against a TypedDict schema
        is notoriously finicky across LangGraph versions -- the `type: ignore`s below
        are a known, verified-at-runtime workaround, not a sign the wiring is wrong.
        """

        graph = StateGraph(_GraphState)
        graph.add_node("supervisor", _supervisor_node)  # type: ignore[call-overload]
        graph.add_node("researcher", _researcher_node)  # type: ignore[call-overload]
        graph.add_node("analyst", _analyst_node)  # type: ignore[call-overload]
        graph.add_node("writer", _writer_node)  # type: ignore[call-overload]

        graph.add_edge(START, "supervisor")
        graph.add_conditional_edges(
            "supervisor",
            _route_after_supervisor,
            {
                "researcher": "researcher",
                "analyst": "analyst",
                "writer": "writer",
                "done": END,
            },
        )
        graph.add_edge("researcher", "supervisor")
        graph.add_edge("analyst", "supervisor")
        graph.add_edge("writer", "supervisor")

        return graph.compile()

    def run(self, state: ResearchState) -> ResearchState:
        """Execute the graph and return the final state.

        `recursion_limit` is an outer safety net on top of SupervisorAgent's own
        max_iterations/stall guards -- it bounds total node visits (supervisor +
        worker steps) in case those guards are ever bypassed by a code change.
        """

        compiled_graph = self.build()
        settings = get_settings()
        result: _GraphState = compiled_graph.invoke(
            {"research_state": state},
            config={"recursion_limit": settings.max_iterations * 3 + 10},
        )
        return result["research_state"]
