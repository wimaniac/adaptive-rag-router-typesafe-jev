"""So sánh hai benchmark runs cùng sample set để lượng hóa repeatability."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from adaptive_rag_router.domain.enums import RouterKind
from adaptive_rag_router.evaluation.metrics import calculate_answer_quality_metrics
from adaptive_rag_router.evaluation.models import BenchmarkRecord


class RouterRepeatability(BaseModel):
    """Repeatability metrics của một router trên các record ghép cặp."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    router: RouterKind
    paired_records: int = Field(ge=1)
    predicted_route_agreement: float = Field(ge=0, le=1)
    expected_route_agreement: float = Field(ge=0, le=1)
    baseline_gold_accuracy: float = Field(ge=0, le=1)
    repeat_fixed_gold_accuracy: float = Field(ge=0, le=1)
    repeat_native_gold_accuracy: float = Field(ge=0, le=1)
    status_agreement: float = Field(ge=0, le=1)
    answer_token_f1: float = Field(ge=0, le=1)
    mean_absolute_quality_delta: float | None = Field(default=None, ge=0)
    mean_cost_delta_usd: float
    mean_latency_delta_ms: float


class RepeatabilityReport(BaseModel):
    """Báo cáo so sánh hai run có cùng query-router job set."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    baseline_run_id: str
    repeat_run_id: str
    paired_query_count: int = Field(ge=1)
    baseline_only_records: int = Field(ge=0)
    repeat_only_records: int = Field(ge=0)
    gold_drift_queries: int = Field(ge=0)
    baseline_gold_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    repeat_gold_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    routers: tuple[RouterRepeatability, ...]


def compare_repeatability(
    baseline: Sequence[BenchmarkRecord],
    repeat: Sequence[BenchmarkRecord],
) -> RepeatabilityReport:
    """Ghép record theo query/router và tính agreement/delta.

    Args:
        baseline: Records của run gốc.
        repeat: Records của run lặp lại.

    Returns:
        Báo cáo typed theo router.

    Raises:
        ValueError: Khi input rỗng, trùng key hoặc không có record ghép cặp.
    """

    if not baseline or not repeat:
        raise ValueError("repeatability cần hai tập record không rỗng")
    baseline_by_key = _index_records(baseline)
    repeat_by_key = _index_records(repeat)
    paired_keys = sorted(set(baseline_by_key) & set(repeat_by_key), key=lambda key: key[0])
    if not paired_keys:
        raise ValueError("hai run không có query-router record chung")
    router_reports: list[RouterRepeatability] = []
    for router in RouterKind:
        pairs = [
            (baseline_by_key[key], repeat_by_key[key]) for key in paired_keys if key[1] is router
        ]
        if not pairs:
            continue
        quality_deltas = [
            abs(float(left.quality_score) - float(right.quality_score))
            for left, right in pairs
            if left.quality_score is not None and right.quality_score is not None
        ]
        answer_metrics = calculate_answer_quality_metrics(
            [right.answer for _, right in pairs],
            [left.answer for left, _ in pairs],
        )
        router_reports.append(
            RouterRepeatability(
                router=router,
                paired_records=len(pairs),
                predicted_route_agreement=_agreement(
                    left.predicted_route == right.predicted_route for left, right in pairs
                ),
                expected_route_agreement=_agreement(
                    left.expected_route == right.expected_route for left, right in pairs
                ),
                baseline_gold_accuracy=_agreement(
                    left.predicted_route == left.expected_route for left, _ in pairs
                ),
                repeat_fixed_gold_accuracy=_agreement(
                    right.predicted_route == left.expected_route for left, right in pairs
                ),
                repeat_native_gold_accuracy=_agreement(
                    right.predicted_route == right.expected_route for _, right in pairs
                ),
                status_agreement=_agreement(left.status is right.status for left, right in pairs),
                answer_token_f1=answer_metrics.token_f1,
                mean_absolute_quality_delta=(
                    sum(quality_deltas) / len(quality_deltas) if quality_deltas else None
                ),
                mean_cost_delta_usd=sum(right.cost_usd - left.cost_usd for left, right in pairs)
                / len(pairs),
                mean_latency_delta_ms=sum(
                    right.latency_ms - left.latency_ms for left, right in pairs
                )
                / len(pairs),
            )
        )
    baseline_run_id = baseline[0].run_id
    repeat_run_id = repeat[0].run_id
    paired_query_ids = {query_id for query_id, _ in paired_keys}
    baseline_gold = _gold_by_query(baseline_by_key, paired_query_ids)
    repeat_gold = _gold_by_query(repeat_by_key, paired_query_ids)
    return RepeatabilityReport(
        baseline_run_id=baseline_run_id,
        repeat_run_id=repeat_run_id,
        paired_query_count=len(paired_query_ids),
        baseline_only_records=len(set(baseline_by_key) - set(repeat_by_key)),
        repeat_only_records=len(set(repeat_by_key) - set(baseline_by_key)),
        gold_drift_queries=sum(
            baseline_gold[query_id] != repeat_gold[query_id] for query_id in paired_query_ids
        ),
        baseline_gold_sha256=_gold_sha256(baseline_gold),
        repeat_gold_sha256=_gold_sha256(repeat_gold),
        routers=tuple(router_reports),
    )


def write_repeatability_report(output_dir: Path, report: RepeatabilityReport) -> tuple[Path, Path]:
    """Ghi JSON và Markdown repeatability atomically.

    Args:
        output_dir: Thư mục artifact đích.
        report: Báo cáo typed cần ghi.

    Returns:
        Cặp đường dẫn JSON và Markdown.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "repeatability.json"
    markdown_path = output_dir / "repeatability.md"
    _atomic_write(json_path, report.model_dump_json(indent=2) + "\n")
    lines = [
        f"# Repeatability `{report.baseline_run_id}` vs `{report.repeat_run_id}`",
        "",
        f"- Paired queries: {report.paired_query_count}",
        f"- Baseline-only records: {report.baseline_only_records}",
        f"- Repeat-only records: {report.repeat_only_records}",
        f"- Gold-drift queries: {report.gold_drift_queries}/{report.paired_query_count}",
        f"- Baseline gold SHA-256: `{report.baseline_gold_sha256}`",
        f"- Repeat gold SHA-256: `{report.repeat_gold_sha256}`",
        "",
        "| Router | N | Route agreement | Gold agreement | Baseline gold acc. | Repeat fixed-gold acc. | Repeat native-gold acc. | Status agreement | Answer token-F1 | |Quality Δ| | Cost Δ USD | Latency Δ ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report.routers:
        quality_delta = (
            "—"
            if item.mean_absolute_quality_delta is None
            else f"{item.mean_absolute_quality_delta:.4f}"
        )
        lines.append(
            f"| {item.router.value} | {item.paired_records} | "
            f"{item.predicted_route_agreement:.4f} | {item.expected_route_agreement:.4f} | "
            f"{item.baseline_gold_accuracy:.4f} | {item.repeat_fixed_gold_accuracy:.4f} | "
            f"{item.repeat_native_gold_accuracy:.4f} | {item.status_agreement:.4f} | "
            f"{item.answer_token_f1:.4f} | {quality_delta} | "
            f"{item.mean_cost_delta_usd:.6f} | {item.mean_latency_delta_ms:.2f} |"
        )
    _atomic_write(markdown_path, "\n".join(lines) + "\n")
    return json_path, markdown_path


def _index_records(
    records: Sequence[BenchmarkRecord],
) -> dict[tuple[str, RouterKind], BenchmarkRecord]:
    indexed: dict[tuple[str, RouterKind], BenchmarkRecord] = {}
    for record in records:
        key = (record.query_id, record.router)
        if key in indexed:
            raise ValueError(f"record trùng key {key}")
        indexed[key] = record
    return indexed


def _agreement(values: Iterable[bool]) -> float:
    materialized = list(values)
    return sum(materialized) / len(materialized)


def _gold_by_query(
    records: dict[tuple[str, RouterKind], BenchmarkRecord],
    query_ids: set[str],
) -> dict[str, str | None]:
    gold: dict[str, str | None] = {}
    for query_id in query_ids:
        labels = {
            record.expected_route
            for (record_query_id, _), record in records.items()
            if record_query_id == query_id
        }
        if len(labels) != 1:
            raise ValueError(f"gold route không nhất quán giữa routers cho query {query_id!r}")
        gold[query_id] = next(iter(labels))
    return gold


def _gold_sha256(gold: dict[str, str | None]) -> str:
    payload = "\n".join(f"{query_id}\t{gold[query_id] or ''}" for query_id in sorted(gold)).encode()
    return hashlib.sha256(payload).hexdigest()


def _atomic_write(path: Path, payload: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8", newline="\n")
    temporary.replace(path)
