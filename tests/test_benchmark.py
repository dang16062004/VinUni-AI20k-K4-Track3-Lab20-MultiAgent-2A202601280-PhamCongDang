"""Unit tests for benchmark helpers (no network, no API key, no cost)."""

from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark


def _fake_state_with_trace(final_answer: str, cost_per_call: float = 0.001) -> ResearchState:
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    state.add_trace_event("researcher", {"attributes": {"cost_usd": cost_per_call}})
    state.add_trace_event("writer", {"cost_usd": cost_per_call})
    state.final_answer = final_answer
    return state


def test_run_benchmark_reports_latency_cost_and_citation_coverage() -> None:
    fake_state = _fake_state_with_trace("Claim A holds [doc1]. Claim B is unclear.")

    result_state, metrics = run_benchmark("demo", "irrelevant query", lambda _: fake_state)

    assert result_state is fake_state
    assert metrics.latency_seconds >= 0
    assert metrics.estimated_cost_usd == 0.002  # both nested and flat cost_usd formats summed
    assert metrics.citation_coverage == 0.5  # 1 of 2 sentences carries a [citation]
    assert metrics.failure_rate == 0.0


def test_run_benchmark_marks_failure_when_runner_raises() -> None:
    def _boom(_: str) -> ResearchState:
        raise RuntimeError("search API down")

    result_state, metrics = run_benchmark("demo", "irrelevant query", _boom)

    assert result_state is None
    assert metrics.failure_rate == 1.0
    assert "search API down" in metrics.notes


def test_run_benchmark_marks_failure_when_no_final_answer_produced() -> None:
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    state.errors.append("stalled")

    _, metrics = run_benchmark("demo", "irrelevant query", lambda _: state)

    assert metrics.failure_rate == 1.0
    assert metrics.notes == "stalled"
