"""Unit tests for Analyst/Writer state handoff, using a fake LLM client (no network/cost)."""

import pytest

from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMResponse


class _FakeLLMClient:
    """Stand-in for LLMClient: returns a fixed response instead of calling OpenAI."""

    def __init__(self, content: str) -> None:
        self._content = content
        self.calls: list[tuple[str, str]] = []

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        self.calls.append((system_prompt, user_prompt))
        return LLMResponse(content=self._content, input_tokens=10, output_tokens=10, cost_usd=0.0)


def _new_state() -> ResearchState:
    return ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))


def test_analyst_requires_research_notes_first() -> None:
    with pytest.raises(AgentExecutionError):
        AnalystAgent(llm_client=_FakeLLMClient("x")).run(_new_state())


def test_analyst_fills_analysis_notes_from_research_notes() -> None:
    state = _new_state()
    state.research_notes = "fact A [doc1]"
    fake = _FakeLLMClient("claim A is supported [doc1]")

    result = AnalystAgent(llm_client=fake).run(state)

    assert result.analysis_notes == "claim A is supported [doc1]"
    assert fake.calls, "LLM should have been called"
    assert "fact A" in fake.calls[0][1]


def test_writer_requires_analysis_notes_first() -> None:
    with pytest.raises(AgentExecutionError):
        WriterAgent(llm_client=_FakeLLMClient("x")).run(_new_state())


def test_writer_fills_final_answer_and_records_agent_result() -> None:
    state = _new_state()
    state.analysis_notes = "claim A is supported [doc1]"
    fake = _FakeLLMClient("Final: claim A holds [doc1]")

    result = WriterAgent(llm_client=fake).run(state)

    assert result.final_answer == "Final: claim A holds [doc1]"
    assert result.agent_results and result.agent_results[-1].agent == "writer"
