"""Supervisor / router skeleton."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span

_ROUTE_RESEARCHER = "researcher"
_ROUTE_ANALYST = "analyst"
_ROUTE_WRITER = "writer"
_ROUTE_DONE = "done"

# How many times a worker route may run back-to-back without progress before the
# Supervisor gives up on it and routes to "done" instead of looping forever.
_MAX_ROUTE_RETRIES = 1


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop.

    Routing policy (linear pipeline, decided purely from `state` so it's easy to test):
      1. No sources and no research notes yet -> researcher
      2. Research notes present, no analysis   -> analyst
      3. Analysis present, no final answer     -> writer
      4. Final answer present                  -> done

    Guardrails:
      - Hard stop once `state.iteration >= settings.max_iterations`, regardless of
        whether a final_answer exists (bounded run time / cost).
      - Stall guard: if the route about to run already ran back-to-back more than
        `_MAX_ROUTE_RETRIES` times without moving state forward, stop instead of
        retrying indefinitely (one retry allowed per worker, then fallback to done).
    """

    name = "supervisor"

    def run(self, state: ResearchState) -> ResearchState:
        settings = get_settings()

        with trace_span("supervisor", {"iteration": state.iteration}) as span:
            if state.iteration >= settings.max_iterations:
                if state.final_answer is None:
                    state.errors.append(
                        f"Stopped at max_iterations={settings.max_iterations} "
                        "without producing a final_answer."
                    )
                next_route = _ROUTE_DONE
            else:
                next_route = self._decide_route(state)
                if self._is_stalled(state, next_route):
                    state.errors.append(
                        f"Route '{next_route}' ran {_MAX_ROUTE_RETRIES + 1} times in a row "
                        "without progress. Falling back to done."
                    )
                    next_route = _ROUTE_DONE
            span["attributes"]["decided_route"] = next_route

        state.record_route(next_route)
        state.add_trace_event("supervisor", span)
        return state

    @staticmethod
    def _decide_route(state: ResearchState) -> str:
        if not state.research_notes and not state.sources:
            return _ROUTE_RESEARCHER
        if state.analysis_notes is None:
            return _ROUTE_ANALYST
        if state.final_answer is None:
            return _ROUTE_WRITER
        return _ROUTE_DONE

    @staticmethod
    def _is_stalled(state: ResearchState, next_route: str) -> bool:
        """True once `next_route` would run for the (retries + 2)-th time in a row."""

        if next_route == _ROUTE_DONE:
            return False
        consecutive = 0
        for route in reversed(state.route_history):
            if route != next_route:
                break
            consecutive += 1
        return consecutive > _MAX_ROUTE_RETRIES
