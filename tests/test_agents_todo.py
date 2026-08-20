"""Unit tests for SupervisorAgent's routing policy.

Replaces the old skeleton guard test (which asserted SupervisorAgent still raised
StudentTodoError) now that Bước 2 is implemented, per the NOTE(student) that used to
be here.
"""

from multi_agent_research_lab.agents.supervisor import SupervisorAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState


def _new_state() -> ResearchState:
    return ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))


def test_routes_to_researcher_when_no_notes_or_sources() -> None:
    state = SupervisorAgent().run(_new_state())
    assert state.route_history == ["researcher"]
    assert state.iteration == 1


def test_routes_through_full_pipeline_as_notes_are_filled() -> None:
    state = _new_state()
    supervisor = SupervisorAgent()

    state = supervisor.run(state)
    assert state.route_history[-1] == "researcher"

    state.research_notes = "fake research notes"
    state = supervisor.run(state)
    assert state.route_history[-1] == "analyst"

    state.analysis_notes = "fake analysis notes"
    state = supervisor.run(state)
    assert state.route_history[-1] == "writer"

    state.final_answer = "fake final answer"
    state = supervisor.run(state)
    assert state.route_history[-1] == "done"
    assert state.errors == []


def test_allows_one_retry_then_falls_back_to_done_on_stall() -> None:
    """If a worker runs but never updates state, Supervisor retries once, then bails."""

    state = _new_state()
    supervisor = SupervisorAgent()

    state = supervisor.run(state)  # attempt 1: researcher
    state = supervisor.run(state)  # attempt 2 (retry): researcher
    assert state.route_history == ["researcher", "researcher"]

    state = supervisor.run(state)  # attempt 3: stall guard fires -> done
    assert state.route_history[-1] == "done"
    assert state.errors


def test_stops_at_max_iterations_even_without_final_answer() -> None:
    state = _new_state()
    state.iteration = get_settings().max_iterations

    state = SupervisorAgent().run(state)
    assert state.route_history == ["done"]
    assert state.errors
