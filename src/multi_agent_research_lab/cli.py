"""Command-line entrypoint for the lab starter."""

from pathlib import Path
from typing import Annotated

import typer
import yaml
from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.schemas import (
    AgentName,
    AgentResult,
    BenchmarkMetrics,
    ResearchQuery,
)
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.evaluation.report import render_markdown_report
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.storage import LocalArtifactStore

app = typer.Typer(help="Multi-Agent Research Lab starter CLI")
console = Console()

_FALLBACK_BENCHMARK_QUERIES = [
    "Research GraphRAG state-of-the-art and write a 500-word summary",
    "Compare single-agent and multi-agent workflows for customer support",
    "Summarize production guardrails for LLM agents",
]


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)


def _parse_query(query: str) -> ResearchQuery:
    try:
        return ResearchQuery(query=query)
    except ValidationError as exc:
        console.print(
            Panel.fit(
                f"Invalid query: {exc.errors()[0]['msg']}",
                title="Input Error",
                style="red",
            )
        )
        raise typer.Exit(code=1) from exc


def run_baseline(request: ResearchQuery) -> ResearchState:
    """Single-agent baseline: one LLM call, no routing/tools/retrieval."""

    state = ResearchState(request=request)
    client = LLMClient()
    response = client.complete(
        system_prompt=(
            "You are a research assistant. Answer the user's research query directly, "
            "concisely, and factually. Note any claims you are unsure about."
        ),
        user_prompt=request.query,
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
    state.add_trace_event(
        "baseline_llm_usage",
        {
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "cost_usd": response.cost_usd,
        },
    )
    return state


def run_multi_agent(request: ResearchQuery) -> ResearchState:
    """Supervisor-routed pipeline: Researcher -> Analyst -> Writer."""

    state = ResearchState(request=request)
    return MultiAgentWorkflow().run(state)


def _load_default_benchmark_queries() -> list[str]:
    config_path = Path("configs/lab_default.yaml")
    if config_path.exists():
        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        queries = data.get("benchmark", {}).get("queries")
        if queries:
            return list(queries)
    return list(_FALLBACK_BENCHMARK_QUERIES)


def _print_metric_line(label: str, metrics: BenchmarkMetrics) -> None:
    cost = "n/a" if metrics.estimated_cost_usd is None else f"${metrics.estimated_cost_usd:.5f}"
    coverage = "n/a" if metrics.citation_coverage is None else f"{metrics.citation_coverage:.0%}"
    console.print(
        f"  [cyan]{label:<11}[/cyan] latency={metrics.latency_seconds:.2f}s cost={cost} "
        f"citation_cov={coverage} failure_rate={metrics.failure_rate}"
    )


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run a minimal single-agent baseline: one LLM call, no routing/tools."""

    _init()
    state = run_baseline(_parse_query(query))

    console.print(Panel.fit(state.final_answer or "", title="Single-Agent Baseline"))
    metadata = state.agent_results[-1].metadata if state.agent_results else {}
    if metadata.get("input_tokens") is not None:
        cost = "n/a" if metadata.get("cost_usd") is None else f"${metadata['cost_usd']:.5f}"
        console.print(
            f"[dim]tokens in={metadata['input_tokens']} out={metadata['output_tokens']} "
            f"cost={cost}[/dim]"
        )


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the multi-agent workflow: Supervisor routes Researcher/Analyst/Writer."""

    _init()
    request = _parse_query(query)
    try:
        result = run_multi_agent(request)
    except StudentTodoError as exc:
        console.print(Panel.fit(str(exc), title="Expected TODO", style="yellow"))
        raise typer.Exit(code=2) from exc
    console.print(result.model_dump_json(indent=2))


@app.command()
def benchmark(
    query: Annotated[
        list[str] | None,
        typer.Option(
            "--query",
            "-q",
            help="Query to benchmark (repeatable). Default: configs/lab_default.yaml queries.",
        ),
    ] = None,
    output: Annotated[
        str, typer.Option("--output", help="Report file name, written under reports/")
    ] = "benchmark_report.md",
) -> None:
    """Run baseline vs multi-agent on the same queries and write a comparison report."""

    _init()
    queries = query or _load_default_benchmark_queries()

    metrics: list[BenchmarkMetrics] = []
    for q in queries:
        console.print(f"[bold]Query:[/bold] {q}")

        _, baseline_metrics = run_benchmark(
            f"baseline · {q[:40]}", q, lambda qq: run_baseline(ResearchQuery(query=qq))
        )
        metrics.append(baseline_metrics)
        _print_metric_line("baseline", baseline_metrics)

        _, multi_metrics = run_benchmark(
            f"multi-agent · {q[:40]}", q, lambda qq: run_multi_agent(ResearchQuery(query=qq))
        )
        metrics.append(multi_metrics)
        _print_metric_line("multi-agent", multi_metrics)

    report_md = render_markdown_report(metrics)
    path = LocalArtifactStore().write_text(output, report_md)
    console.print(Panel.fit(f"Report written to {path}", title="Benchmark", style="green"))


if __name__ == "__main__":
    app()
