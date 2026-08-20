"""Cheap tests for MultiAgentWorkflow's graph wiring (no network, no API key).

These deliberately don't run the compiled graph end-to-end (that needs
OPENAI_API_KEY and costs a bit -- see docs/lab_guide.md for the integration-test
command). They only check the parts that don't touch an LLM: the graph compiles,
and the routing helper reads the Supervisor's decision correctly.
"""

from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow, _route_after_supervisor


def _new_state() -> ResearchState:
    return ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))


def test_build_compiles_without_calling_any_agent() -> None:
    # build() only registers node functions; agents are constructed lazily inside
    # them when a node actually runs, so this must succeed with no API key set.
    compiled = MultiAgentWorkflow().build()
    assert compiled is not None


def test_route_after_supervisor_reads_last_recorded_route() -> None:
    state = _new_state()
    state.record_route("analyst")
    assert _route_after_supervisor({"research_state": state}) == "analyst"


def test_route_after_supervisor_defaults_to_done_when_no_route_yet() -> None:
    state = _new_state()
    assert _route_after_supervisor({"research_state": state}) == "done"
