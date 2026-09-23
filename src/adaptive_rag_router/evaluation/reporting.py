"""Kết xuất báo cáo benchmark dạng Markdown dễ đọc và kiểm tra."""

from __future__ import annotations

from adaptive_rag_router.evaluation.models import BenchmarkReport, BootstrapInterval


def _format_number(value: float | None, digits: int = 4) -> str:
    """Định dạng số tùy chọn nhất quán cho bảng Markdown."""

    return "—" if value is None else f"{value:.{digits}f}"


def _format_interval(interval: BootstrapInterval | None) -> str:
    """Định dạng estimate và confidence interval trên một dòng."""

    if interval is None:
        return "—"
    return f"{interval.estimate:.4f} [{interval.lower:.4f}, {interval.upper:.4f}]"


def build_markdown_report(report: BenchmarkReport) -> str:
    """Chuyển BenchmarkReport thành báo cáo Markdown tiếng Việt.

    Args:
        report: Báo cáo typed từ BenchmarkRunner.

    Returns:
        Nội dung Markdown bao gồm aggregate và paired comparison.
    """

    duration_seconds = max(
        0.0,
        (report.finished_at - report.started_at).total_seconds(),
    )
    ablation_ids = {
        str(record.metadata.get("ablation_id", "baseline")) for record in report.records
    }
    ablation_label = next(iter(ablation_ids)) if len(ablation_ids) == 1 else "mixed"
    lines = [
        f"# Báo cáo benchmark `{report.run_id}`",
        "",
        "## Tổng quan",
        "",
        f"- Split: `{report.config.split.value}`",
        f"- Số query: {report.sample_count}",
        f"- Số record: {len(report.records)}",
        f"- Live mode: {'có' if report.config.live else 'không'}",
        f"- Pipeline policy: `{ablation_label}`",
        f"- Lượt bị budget từ chối: {report.denied_by_budget}",
        f"- Dừng sớm: {'có' if report.stopped_early else 'không'}",
        f"- Lý do dừng: `{report.stop_reason or 'không'}`",
        f"- Thời lượng runner: {duration_seconds:.3f} giây",
        "",
        "## Metric theo router",
        "",
        "| Router | N | Accuracy | Macro-F1 | Quality mean | Tokens mean | Cost mean (USD) | p50/p95 latency (ms) | Retrieval | Fallback | Error | Abstain | Credits |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for metrics in report.router_metrics:
        routing = metrics.routing
        lines.append(
            "| "
            + " | ".join(
                [
                    metrics.router.value,
                    str(metrics.record_count),
                    _format_number(routing.accuracy if routing else None),
                    _format_number(routing.macro_f1 if routing else None),
                    _format_number(metrics.quality.mean),
                    _format_number(metrics.total_tokens.mean, 1),
                    _format_number(metrics.cost_usd.mean, 6),
                    f"{_format_number(metrics.latency_ms.p50, 2)} / {_format_number(metrics.latency_ms.p95, 2)}",
                    _format_number(metrics.retrieval_rate),
                    _format_number(metrics.fallback_rate),
                    _format_number(metrics.error_rate),
                    _format_number(metrics.abstention_rate),
                    str(metrics.total_external_credits),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Calibration",
            "",
            "| Router | N | Brier | NLL | ECE | Bins |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for metrics in report.router_metrics:
        calibration = metrics.calibration
        lines.append(
            "| "
            + " | ".join(
                [
                    metrics.router.value,
                    str(calibration.sample_count) if calibration else "0",
                    _format_number(calibration.brier_score if calibration else None),
                    _format_number(calibration.negative_log_likelihood if calibration else None),
                    _format_number(calibration.expected_calibration_error if calibration else None),
                    str(calibration.bin_count) if calibration else "—",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Routing nâng cao và risk-coverage",
            "",
            "Lớp critical được định nghĩa trước là route dùng `web` hoặc model tier `strong`.",
            "",
            "| Router | Tier MAE | Tier QWK | Critical AUROC | Critical AUPRC | Critical recall | AURC |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for metrics in report.router_metrics:
        ordinal = metrics.ordinal_model_tier
        critical = metrics.critical_route
        risk = metrics.risk_coverage
        lines.append(
            "| "
            + " | ".join(
                [
                    metrics.router.value,
                    _format_number(ordinal.mean_absolute_error if ordinal else None),
                    _format_number(ordinal.quadratic_weighted_kappa if ordinal else None),
                    _format_number(critical.auroc if critical else None),
                    _format_number(critical.auprc if critical else None),
                    _format_number(critical.critical_class_recall if critical else None),
                    _format_number(risk.area_under_curve if risk else None),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Answer, citation và retrieval",
            "",
            "| Router | Exact match | Token-F1 | Citation P/R | Supporting-fact recall | Context macro-F1 | Insufficient recall |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for metrics in report.router_metrics:
        answer = metrics.answer_quality
        citations = metrics.citations
        retrieval = metrics.retrieval
        lines.append(
            "| "
            + " | ".join(
                [
                    metrics.router.value,
                    _format_number(answer.exact_match if answer else None),
                    _format_number(answer.token_f1 if answer else None),
                    (
                        f"{_format_number(citations.precision)} / {_format_number(citations.recall)}"
                        if citations
                        else "—"
                    ),
                    _format_number(retrieval.supporting_fact_recall if retrieval else None),
                    _format_number(retrieval.context_quality_macro_f1 if retrieval else None),
                    _format_number(retrieval.insufficient_recall if retrieval else None),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Paired bootstrap",
            "",
            "Chênh lệch được tính theo `candidate - baseline`; giá trị âm tốt hơn cho cost và latency.",
            "",
            "| Candidate | Baseline | Quality Δ (95% CI) | Quality lower (one-sided 95%) | Cost Δ USD (95% CI) | Latency Δ ms (95% CI) |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for comparison in report.comparisons:
        lines.append(
            "| "
            + " | ".join(
                [
                    comparison.candidate.value,
                    comparison.baseline.value,
                    _format_interval(comparison.quality_difference),
                    _format_number(comparison.quality_one_sided_lower_95),
                    _format_interval(comparison.cost_difference_usd),
                    _format_interval(comparison.latency_difference_ms),
                ]
            )
            + " |"
        )

    lines.extend(["", "## Tiêu chí thành công", ""])
    if report.success_criteria is None:
        if report.config.live:
            lines.append(
                "Không đánh giá acceptance trên Tavily live track; track này chỉ đo external "
                "validity và vận hành mạng."
            )
        elif report.config.split.value != "test":
            lines.append("Chỉ held-out test split mới được đánh giá acceptance criteria.")
        else:
            lines.append("Chưa đủ dữ liệu để đánh giá.")
    else:
        lines.extend(
            [
                f"Trạng thái tổng: **{report.success_criteria.overall_status.value}**.",
                "",
                "| Tiêu chí | Trạng thái | Quan sát | Ngưỡng |",
                "|---|---|---:|---:|",
            ]
        )
        for criterion in report.success_criteria.criteria:
            lines.append(
                "| "
                + " | ".join(
                    [
                        criterion.name,
                        criterion.status.value,
                        _format_number(criterion.observed),
                        _format_number(criterion.threshold),
                    ]
                )
                + " |"
            )

    lines.extend(
        [
            "",
            "## Ghi chú tái lập",
            "",
            f"- Seed: `{report.config.seed}`",
            f"- Bootstrap resamples: `{report.config.bootstrap_resamples}`",
            f"- Confidence level: `{report.config.confidence_level}`",
            f"- ECE bins: `{report.config.ece_bins}`",
            "- Quality score do evaluator được inject cung cấp; runner không tự dùng LLM-as-judge.",
            "",
        ]
    )
    return "\n".join(lines)
