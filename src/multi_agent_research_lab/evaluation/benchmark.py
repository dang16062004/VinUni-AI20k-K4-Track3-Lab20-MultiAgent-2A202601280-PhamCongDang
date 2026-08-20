"""Benchmark skeleton for single-agent vs multi-agent."""

from __future__ import annotations

import re
from collections.abc import Callable
from time import perf_counter
from typing import Any

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState

Runner = Callable[[str], ResearchState]

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_CITATION_RE = re.compile(r"\[[A-Za-z0-9_\-]+\]")


def _payload_cost_usd(payload: dict[str, Any]) -> float | None:
    """Read `cost_usd` from a trace payload, whether it's top-level or nested under
    `attributes` (agents log it via `trace_span`, whose span dict nests attributes)."""

    value = payload.get("cost_usd")
    if value is None:
        attributes = payload.get("attributes")
        if isinstance(attributes, dict):
            value = attributes.get("cost_usd")
    return float(value) if isinstance(value, int | float) else None


def _estimate_cost_usd(state: ResearchState) -> float | None:
    """Sum the cost every LLM-calling agent logged on `state.trace`.

    This is the single source of truth for cost -- not `state.agent_results`, which
    only the Writer populates and would undercount the Researcher/Analyst calls.
    """

    costs = [
        cost
        for event in state.trace
        if isinstance(event.get("payload"), dict)
        and (cost := _payload_cost_usd(event["payload"])) is not None
    ]
    return sum(costs) if costs else None


def _citation_coverage(final_answer: str | None) -> float | None:
    """Fraction of sentences in `final_answer` that carry a `[citation]` marker.

    Evaluation-pitfall caveat: this is a crude regex proxy. It only checks that
    *some* bracketed token is present in a sentence -- not that the id is real, that
    it supports the specific claim, or that every claim needing a citation has one.
    Treat it as a cheap regression signal (did citation density drop between runs?),
    not a substitute for someone actually reading the citations.
    """

    if not final_answer:
        return None
    sentences = [s.strip() for s in _SENTENCE_RE.split(final_answer) if s.strip()]
    if not sentences:
        return None
    cited = sum(1 for s in sentences if _CITATION_RE.search(s))
    return cited / len(sentences)


def run_benchmark(
    run_name: str,
    query: str,
    runner: Runner,
    quality_score: float | None = None,
) -> tuple[ResearchState | None, BenchmarkMetrics]:
    """Run one query through `runner`, measure latency/cost, and return metrics.

    `quality_score` is intentionally NOT computed automatically here. docs/lab_guide.md
    scores quality via peer review (docs/peer_review_rubric.md, 0-10) specifically to
    avoid the common pitfall of a system (or an LLM judge from the same model family)
    grading its own output. Pass the peer-review score in once you have it, or leave
    it None and fill the report column in by hand.
    """

    started = perf_counter()
    try:
        state = runner(query)
    except Exception as exc:  # noqa: BLE001 - a bad run must not crash the whole benchmark
        latency = perf_counter() - started
        metrics = BenchmarkMetrics(
            run_name=run_name,
            latency_seconds=latency,
            quality_score=quality_score,
            failure_rate=1.0,
            notes=f"Run raised {type(exc).__name__}: {exc}",
        )
        return None, metrics

    latency = perf_counter() - started
    failed = state.final_answer is None
    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=latency,
        estimated_cost_usd=_estimate_cost_usd(state),
        quality_score=quality_score,
        citation_coverage=_citation_coverage(state.final_answer),
        failure_rate=1.0 if failed else 0.0,
        notes="" if not state.errors else "; ".join(state.errors),
    )
    return state, metrics
