"""Benchmark report rendering."""

from multi_agent_research_lab.core.schemas import BenchmarkMetrics

_GROUP_SEPARATOR = " · "  # " · " -- see cli.py's `benchmark` command for run_name format


def render_markdown_report(metrics: list[BenchmarkMetrics]) -> str:
    """Render benchmark metrics to a markdown table plus a short comparison summary."""

    lines = [
        "# Benchmark Report",
        "",
        "| Run | Latency (s) | Cost (USD) | Quality | Citation cov. | Failure rate | Notes |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for item in metrics:
        cost = "" if item.estimated_cost_usd is None else f"{item.estimated_cost_usd:.4f}"
        quality = "" if item.quality_score is None else f"{item.quality_score:.1f}"
        citation = "" if item.citation_coverage is None else f"{item.citation_coverage:.0%}"
        failure = "" if item.failure_rate is None else f"{item.failure_rate:.0%}"
        lines.append(
            f"| {item.run_name} | {item.latency_seconds:.2f} | {cost} | {quality} "
            f"| {citation} | {failure} | {item.notes} |"
        )

    if metrics:
        lines.extend(["", "## Summary", "", *_summary_lines(metrics)])

    failed = [m for m in metrics if m.failure_rate]
    if failed:
        lines.extend(["", "## Failed runs", ""])
        lines.extend(f"- **{m.run_name}**: {m.notes or 'no notes recorded'}" for m in failed)

    return "\n".join(lines) + "\n"


def _summary_lines(metrics: list[BenchmarkMetrics]) -> list[str]:
    """Group runs by the part of `run_name` before ' · ' (e.g. 'baseline', 'multi-agent',
    the convention used by `cli.py`'s `benchmark` command) and report per-group averages.
    Runs whose name doesn't follow that convention are grouped under 'all runs'.
    """

    groups: dict[str, list[BenchmarkMetrics]] = {}
    for m in metrics:
        group = (
            m.run_name.split(_GROUP_SEPARATOR, 1)[0]
            if _GROUP_SEPARATOR in m.run_name
            else "all runs"
        )
        groups.setdefault(group, []).append(m)

    lines = []
    for group, items in groups.items():
        n = len(items)
        avg_latency = sum(m.latency_seconds for m in items) / n
        costs = [m.estimated_cost_usd for m in items if m.estimated_cost_usd is not None]
        avg_cost = sum(costs) / len(costs) if costs else None
        coverages = [m.citation_coverage for m in items if m.citation_coverage is not None]
        avg_coverage = sum(coverages) / len(coverages) if coverages else None
        n_failed = sum(1 for m in items if m.failure_rate)

        cost_str = "n/a" if avg_cost is None else f"${avg_cost:.5f}"
        coverage_str = "n/a" if avg_coverage is None else f"{avg_coverage:.0%}"
        lines.append(
            f"- **{group}** ({n} run{'s' if n != 1 else ''}): "
            f"avg latency {avg_latency:.2f}s, avg cost {cost_str}, "
            f"avg citation coverage {coverage_str}, {n_failed} failed."
        )
    return lines
