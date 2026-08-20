"""Analyst agent skeleton."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient

_SYSTEM_PROMPT = (
    "You are the Analyst in a multi-agent research pipeline. You receive research notes "
    "where each fact is tagged with a document_id in square brackets. Extract the key "
    "claims, note where sources agree or disagree, and flag any claim backed by only one "
    "source or weak evidence. Keep the [document_id] citations attached to each claim you "
    "restate."
)


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights."""

    name = "analyst"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self._llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        if not state.research_notes:
            raise AgentExecutionError(
                "AnalystAgent requires state.research_notes; run ResearcherAgent first."
            )

        with trace_span("analyst", {"query": state.request.query}) as span:
            response = self._llm_client.complete(
                system_prompt=_SYSTEM_PROMPT,
                user_prompt=(
                    f"Query: {state.request.query}\n\nResearch notes:\n{state.research_notes}"
                ),
            )
            state.analysis_notes = response.content

            span["attributes"].update(
                {
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": response.cost_usd,
                }
            )

        state.add_trace_event("analyst", span)
        return state
