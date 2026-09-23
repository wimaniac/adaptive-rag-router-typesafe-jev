"""Tạo SVG nhẹ cho Pareto, reliability và risk-coverage từ benchmark report."""

from __future__ import annotations

from collections import defaultdict
from html import escape

from adaptive_rag_router.domain.enums import RouterKind
from adaptive_rag_router.evaluation.models import BenchmarkReport

_COLORS = {
    RouterKind.JEV: "#2563eb",
    RouterKind.RULE: "#16a34a",
    RouterKind.LLM: "#dc2626",
}


def build_pareto_svg(report: BenchmarkReport) -> str:
    """Vẽ quality-cost scatter để quan sát Pareto frontier giữa các router."""

    points = [
        (item.router, item.cost_usd.mean, item.quality.mean)
        for item in report.router_metrics
        if item.cost_usd.mean is not None and item.quality.mean is not None
    ]
    max_cost = max((cost for _, cost, _ in points), default=1.0) or 1.0
    max_quality = max((quality for _, _, quality in points), default=1.0) or 1.0
    elements = _svg_frame(
        "Quality-cost Pareto",
        "Mean variable cost (USD/query)",
        "Mean answer quality",
    )
    for router, cost, quality in points:
        x = _scale(cost, 0.0, max_cost, 80, 720)
        y = _scale(quality, 0.0, max_quality, 430, 60)
        color = _COLORS[router]
        elements.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="{color}"/>')
        elements.append(
            f'<text x="{x + 10:.1f}" y="{y - 8:.1f}" font-size="13" fill="{color}">'
            f"{escape(router.value)} (${cost:.6f}, {quality:.4f})</text>"
        )
    return _finish_svg(elements)


def build_reliability_svg(report: BenchmarkReport) -> str:
    """Vẽ reliability curve tính lại từ route confidence và gold correctness."""

    elements = _svg_frame("Reliability curve", "Mean confidence", "Empirical accuracy")
    elements.append('<line x1="80" y1="430" x2="720" y2="60" stroke="#94a3b8"/>')
    for router in report.config.routers:
        bins: dict[int, list[tuple[float, float]]] = defaultdict(list)
        for record in report.records:
            if (
                record.router is not router
                or record.expected_route is None
                or record.predicted_route is None
                or not record.route_probabilities
            ):
                continue
            confidence = max(record.route_probabilities.values())
            index = min(int(confidence * report.config.ece_bins), report.config.ece_bins - 1)
            bins[index].append((confidence, float(record.expected_route == record.predicted_route)))
        points: list[tuple[float, float]] = []
        for index in sorted(bins):
            values = bins[index]
            points.append(
                (
                    sum(item[0] for item in values) / len(values),
                    sum(item[1] for item in values) / len(values),
                )
            )
        _append_series(elements, router, points)
    return _finish_svg(elements)


def build_risk_coverage_svg(report: BenchmarkReport) -> str:
    """Vẽ risk-coverage curve từ các điểm đã tính trong router metrics."""

    elements = _svg_frame("Risk-coverage", "Coverage", "Risk")
    for metrics in report.router_metrics:
        curve = metrics.risk_coverage
        if curve is None:
            continue
        _append_series(
            elements,
            metrics.router,
            [(point.coverage, point.risk) for point in curve.points],
        )
    return _finish_svg(elements)


def _svg_frame(title: str, x_label: str, y_label: str) -> list[str]:
    return [
        '<svg xmlns="http://www.w3.org/2000/svg" width="800" height="500" '
        'viewBox="0 0 800 500" role="img">',
        '<rect width="800" height="500" fill="#ffffff"/>',
        f'<text x="400" y="28" text-anchor="middle" font-size="20">{escape(title)}</text>',
        '<line x1="80" y1="430" x2="720" y2="430" stroke="#0f172a"/>',
        '<line x1="80" y1="430" x2="80" y2="60" stroke="#0f172a"/>',
        f'<text x="400" y="480" text-anchor="middle" font-size="14">{escape(x_label)}</text>',
        f'<text x="18" y="250" text-anchor="middle" font-size="14" '
        f'transform="rotate(-90 18 250)">{escape(y_label)}</text>',
    ]


def _append_series(
    elements: list[str],
    router: RouterKind,
    points: list[tuple[float, float]],
) -> None:
    if not points:
        return
    color = _COLORS[router]
    coordinates = [(_scale(x, 0.0, 1.0, 80, 720), _scale(y, 0.0, 1.0, 430, 60)) for x, y in points]
    path = " ".join(f"{x:.1f},{y:.1f}" for x, y in coordinates)
    elements.append(f'<polyline points="{path}" fill="none" stroke="{color}" stroke-width="2"/>')
    for x, y in coordinates:
        elements.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>')
    legend_y = 48 + 18 * list(RouterKind).index(router)
    elements.append(
        f'<text x="650" y="{legend_y}" font-size="12" fill="{color}">{escape(router.value)}</text>'
    )


def _scale(value: float, minimum: float, maximum: float, start: float, end: float) -> float:
    if maximum <= minimum:
        return (start + end) / 2
    ratio = min(max((value - minimum) / (maximum - minimum), 0.0), 1.0)
    return start + ratio * (end - start)


def _finish_svg(elements: list[str]) -> str:
    elements.append("</svg>")
    return "\n".join(elements) + "\n"
