"""Writer agent skeleton."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient

_SYSTEM_PROMPT = (
    "You are the Writer in a multi-agent research pipeline. Using the analysis notes "
    "(claims tagged with [document_id] citations), write the final answer for the "
    "audience described in the query. Keep the [document_id] citations inline. Be "
    "concise and do not restate these instructions."
)


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes."""

    name = "writer"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self._llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        if not state.analysis_notes:
            raise AgentExecutionError(
                "WriterAgent requires state.analysis_notes; run AnalystAgent first."
            )

        with trace_span("writer", {"query": state.request.query}) as span:
            response = self._llm_client.complete(
                system_prompt=_SYSTEM_PROMPT,
                user_prompt=(
                    f"Query: {state.request.query}\n"
                    f"Audience: {state.request.audience}\n\n"
                    f"Analysis notes:\n{state.analysis_notes}"
                ),
            )
            state.final_answer = response.content
            state.agent_results.append(
                AgentResult(
                    agent=AgentName.WRITER,
                    content=response.content,
                    metadata={
                        "input_tokens": response.input_tokens,
                        "output_tokens": response.output_tokens,
                        "cost_usd": response.cost_usd,
                    },
                )
            )

            span["attributes"].update(
                {
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": response.cost_usd,
                }
            )

        state.add_trace_event("writer", span)
        return state
