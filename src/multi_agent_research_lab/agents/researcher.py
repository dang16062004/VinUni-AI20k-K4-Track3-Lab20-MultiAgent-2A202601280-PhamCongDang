"""Researcher agent skeleton."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient

_SYSTEM_PROMPT = (
    "You are the Researcher in a multi-agent research pipeline. You are given a query "
    "and a set of retrieved sources, each tagged with a document_id. Write concise "
    "research notes: list the key facts relevant to the query, each followed by its "
    "document_id in square brackets, e.g. 'X reduces latency via caching [autogen]'. "
    "Only use facts supported by the given sources; do not invent citations."
)


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"

    def __init__(
        self,
        search_client: SearchClient | None = None,
        llm_client: LLMClient | None = None,
    ) -> None:
        self._search_client = search_client or SearchClient()
        self._llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        with trace_span("researcher", {"query": state.request.query}) as span:
            sources = self._search_client.search(
                state.request.query, max_results=state.request.max_sources
            )
            state.sources = sources

            sources_block = "\n\n".join(
                f"[{doc.metadata.get('document_id', i)}] {doc.title}\n{doc.snippet}"
                for i, doc in enumerate(sources)
            )
            response = self._llm_client.complete(
                system_prompt=_SYSTEM_PROMPT,
                user_prompt=f"Query: {state.request.query}\n\nSources:\n{sources_block}",
            )
            state.research_notes = response.content

            span["attributes"].update(
                {
                    "num_sources": len(sources),
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": response.cost_usd,
                }
            )

        state.add_trace_event("researcher", span)
        return state
