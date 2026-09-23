"""Phân tích confirmatory tuần tự cho giả thuyết Jev so với LLM router."""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from collections.abc import Iterable, Sequence
from enum import StrEnum
from pathlib import Path
from statistics import fmean, median

from pydantic import Field

from adaptive_rag_router.domain.enums import CompletionStatus, RouterKind
from adaptive_rag_router.evaluation.models import (
    BenchmarkRecord,
    BenchmarkSample,
    EvaluationModel,
)


class ConfirmatoryDecision(StrEnum):
    """Các trạng thái kết luận ổn định của confirmatory track."""

    CONTINUE = "continue"
    SUPPORTED = "supported"
    NOT_SUPPORTED = "not_supported"
    INCONCLUSIVE = "inconclusive"


class ConfirmatoryReport(EvaluationModel):
    """Kết quả một group-sequential look của nghiên cứu confirmatory.

    Args:
        run_id: Run ID chung qua mọi look.
        look_index: Chỉ số look bắt đầu từ một.
        planned_looks: Các mốc sample đã preregister.
        pair_count: Số query có đủ Jev và LLM records.
        alpha_per_look: Alpha một phía sau Bonferroni correction.
        quality_delta: Trung bình quality Jev trừ LLM.
        quality_lower: Cận dưới bootstrap một phía đã correction.
        quality_upper: Cận trên bootstrap một phía đã correction.
        cost_reduction: Mức giảm mean variable cost của Jev.
        p50_latency_reduction: Mức giảm p50 latency của Jev.
        p95_latency_regression: Mức tăng p95 latency của Jev.
        unhandled_error_rate: Tỷ lệ failed records trên Jev.
        decision: Quyết định tuần tự tại look hiện tại.
        criteria: Trạng thái pass/fail của các tiêu chí xác nhận.
        manifest_sha256: Hash manifest query đã khóa.
    """

    run_id: str
    look_index: int = Field(ge=1)
    planned_looks: tuple[int, ...]
    pair_count: int = Field(ge=0)
    alpha_per_look: float = Field(gt=0, lt=1)
    quality_delta: float | None = None
    quality_lower: float | None = None
    quality_upper: float | None = None
    cost_reduction: float | None = None
    p50_latency_reduction: float | None = None
    p95_latency_regression: float | None = None
    unhandled_error_rate: float | None = None
    decision: ConfirmatoryDecision
    criteria: dict[str, bool]
    manifest_sha256: str


def write_confirmatory_manifest(
    path: Path,
    samples: Sequence[BenchmarkSample],
    *,
    excluded_run_ids: Sequence[str],
) -> str:
    """Ghi manifest query xác nhận và trả SHA-256 của payload canonical.

    Args:
        path: File JSON đích.
        samples: Query đã khóa theo đúng thứ tự chạy.
        excluded_run_ids: Các run development/pilot đã loại khỏi sampling frame.

    Returns:
        SHA-256 hexadecimal của phần query manifest canonical.
    """

    query_rows = [
        {
            "query_id": sample.query_id,
            "dataset": sample.dataset,
            "stratum": sample.stratum,
            "group_id": sample.group_id,
        }
        for sample in samples
    ]
    canonical = json.dumps(query_rows, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    payload = {
        "manifest_sha256": digest,
        "sample_count": len(query_rows),
        "excluded_run_ids": list(excluded_run_ids),
        "queries": query_rows,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return digest


def analyze_confirmatory_look(
    records: Sequence[BenchmarkRecord],
    *,
    run_id: str,
    look_index: int,
    planned_looks: tuple[int, ...],
    manifest_sha256: str,
    resamples: int,
    seed: int,
    quality_margin: float = 0.02,
    minimum_cost_reduction: float = 0.20,
    minimum_p50_reduction: float = 0.15,
    maximum_p95_regression: float = 0.05,
    maximum_error_rate: float = 0.005,
    familywise_alpha: float = 0.05,
) -> ConfirmatoryReport:
    """Đánh giá một sequential look bằng paired stratified bootstrap.

    Bonferroni chia alpha đều cho mọi look, nên có thể dừng sớm khi toàn bộ tiêu
    chí đạt mà vẫn giữ family-wise one-sided error không quá ``familywise_alpha``.

    Args:
        records: Adaptive records hiện có của cùng run.
        run_id: Run ID cần phân tích.
        look_index: Look hiện tại, bắt đầu từ một.
        planned_looks: Các mốc sample đã khóa.
        manifest_sha256: Hash manifest dùng chứng minh sample không đổi.
        resamples: Số bootstrap resamples.
        seed: Seed tái lập.
        quality_margin: Non-inferiority margin dương.
        minimum_cost_reduction: Mức giảm mean cost tối thiểu.
        minimum_p50_reduction: Mức giảm p50 latency tối thiểu.
        maximum_p95_regression: Mức tăng p95 latency tối đa.
        maximum_error_rate: Tỷ lệ failed Jev tối đa.
        familywise_alpha: Tổng alpha một phía qua mọi look.

    Returns:
        Báo cáo typed với quyết định continue/supported/not-supported/inconclusive.

    Raises:
        ValueError: Khi look hoặc records không hợp lệ.
    """

    if not 1 <= look_index <= len(planned_looks):
        raise ValueError("look_index nằm ngoài planned_looks")
    if resamples < 100:
        raise ValueError("confirmatory bootstrap cần ít nhất 100 resamples")
    alpha_per_look = familywise_alpha / len(planned_looks)
    by_router = {
        router: {record.query_id: record for record in records if record.router is router}
        for router in (RouterKind.JEV, RouterKind.LLM)
    }
    pair_ids = sorted(set(by_router[RouterKind.JEV]) & set(by_router[RouterKind.LLM]))
    quality_ids = [
        query_id
        for query_id in pair_ids
        if by_router[RouterKind.JEV][query_id].quality_score is not None
        and by_router[RouterKind.LLM][query_id].quality_score is not None
    ]
    quality_delta: float | None = None
    quality_lower: float | None = None
    quality_upper: float | None = None
    if quality_ids:
        differences = [
            float(by_router[RouterKind.JEV][query_id].quality_score or 0.0)
            - float(by_router[RouterKind.LLM][query_id].quality_score or 0.0)
            for query_id in quality_ids
        ]
        strata = [by_router[RouterKind.JEV][query_id].stratum for query_id in quality_ids]
        quality_delta, quality_lower, quality_upper = _one_sided_bootstrap_bounds(
            differences,
            strata,
            alpha=alpha_per_look,
            resamples=resamples,
            seed=seed + look_index,
        )

    jev_records = list(by_router[RouterKind.JEV].values())
    llm_records = list(by_router[RouterKind.LLM].values())
    cost_reduction = _relative_reduction(
        _mean(record.cost_usd for record in jev_records),
        _mean(record.cost_usd for record in llm_records),
    )
    p50_reduction = _relative_reduction(
        _percentile([record.latency_ms for record in jev_records], 0.50),
        _percentile([record.latency_ms for record in llm_records], 0.50),
    )
    jev_p95 = _percentile([record.latency_ms for record in jev_records], 0.95)
    llm_p95 = _percentile([record.latency_ms for record in llm_records], 0.95)
    p95_regression = (
        (jev_p95 - llm_p95) / llm_p95
        if jev_p95 is not None and llm_p95 is not None and llm_p95 > 0
        else None
    )
    error_rate = (
        sum(record.status is CompletionStatus.FAILED for record in jev_records) / len(jev_records)
        if jev_records
        else None
    )
    expected_pairs = planned_looks[look_index - 1]
    criteria = {
        "complete_pairs": len(pair_ids) >= expected_pairs,
        "quality_non_inferiority": quality_lower is not None and quality_lower >= -quality_margin,
        "mean_cost_reduction": cost_reduction is not None
        and cost_reduction >= minimum_cost_reduction,
        "p50_latency_reduction": p50_reduction is not None
        and p50_reduction >= minimum_p50_reduction,
        "p95_latency_regression": p95_regression is not None
        and p95_regression <= maximum_p95_regression,
        "unhandled_error_rate": error_rate is not None and error_rate < maximum_error_rate,
    }
    if all(criteria.values()):
        decision = ConfirmatoryDecision.SUPPORTED
    elif look_index < len(planned_looks):
        decision = ConfirmatoryDecision.CONTINUE
    elif quality_upper is not None and quality_upper < -quality_margin:
        decision = ConfirmatoryDecision.NOT_SUPPORTED
    elif criteria["quality_non_inferiority"]:
        decision = ConfirmatoryDecision.NOT_SUPPORTED
    else:
        decision = ConfirmatoryDecision.INCONCLUSIVE
    return ConfirmatoryReport(
        run_id=run_id,
        look_index=look_index,
        planned_looks=planned_looks,
        pair_count=len(pair_ids),
        alpha_per_look=alpha_per_look,
        quality_delta=quality_delta,
        quality_lower=quality_lower,
        quality_upper=quality_upper,
        cost_reduction=cost_reduction,
        p50_latency_reduction=p50_reduction,
        p95_latency_regression=p95_regression,
        unhandled_error_rate=error_rate,
        decision=decision,
        criteria=criteria,
        manifest_sha256=manifest_sha256,
    )


def write_confirmatory_report(path: Path, report: ConfirmatoryReport) -> tuple[Path, Path]:
    """Ghi JSON và Markdown atomic cho một confirmatory look.

    Args:
        path: Đường dẫn JSON đích.
        report: Báo cáo typed cần ghi.

    Returns:
        Cặp đường dẫn JSON và Markdown.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path = path.with_suffix(".md")
    _atomic_write(path, report.model_dump_json(indent=2) + "\n")
    rows = "\n".join(
        f"| {name} | {'đạt' if passed else 'chưa đạt'} |"
        for name, passed in report.criteria.items()
    )
    markdown = f"""# Confirmatory look {report.look_index}

- Run: `{report.run_id}`
- Paired queries: {report.pair_count}
- Planned looks: {", ".join(str(value) for value in report.planned_looks)}
- Alpha một phía/look: {report.alpha_per_look:.6f}
- Manifest SHA-256: `{report.manifest_sha256}`
- Decision: **{report.decision}**

| Metric | Giá trị |
|---|---:|
| Quality Jev - LLM | {_format_optional(report.quality_delta)} |
| One-sided corrected lower | {_format_optional(report.quality_lower)} |
| One-sided corrected upper | {_format_optional(report.quality_upper)} |
| Mean cost reduction | {_format_percent(report.cost_reduction)} |
| p50 latency reduction | {_format_percent(report.p50_latency_reduction)} |
| p95 latency regression | {_format_percent(report.p95_latency_regression)} |
| Jev unhandled error rate | {_format_percent(report.unhandled_error_rate)} |

| Tiêu chí | Trạng thái |
|---|---|
{rows}
"""
    _atomic_write(markdown_path, markdown)
    return path, markdown_path


def _one_sided_bootstrap_bounds(
    differences: Sequence[float],
    strata: Sequence[str],
    *,
    alpha: float,
    resamples: int,
    seed: int,
) -> tuple[float, float, float]:
    """Trả mean cùng hai one-sided percentile bounds."""

    indices_by_stratum: dict[str, list[int]] = defaultdict(list)
    for index, stratum in enumerate(strata):
        indices_by_stratum[stratum].append(index)
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(resamples):
        sampled: list[float] = []
        for indices in indices_by_stratum.values():
            sampled.extend(differences[rng.choice(indices)] for _ in indices)
        estimates.append(fmean(sampled))
    estimates.sort()
    return fmean(differences), _quantile(estimates, alpha), _quantile(estimates, 1 - alpha)


def _quantile(values: Sequence[float], probability: float) -> float:
    """Nội suy tuyến tính quantile từ dãy đã sắp xếp."""

    position = (len(values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    weight = position - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def _mean(values: Iterable[float]) -> float | None:
    """Tính mean an toàn cho iterable số."""

    materialized = list(values)
    return fmean(materialized) if materialized else None


def _percentile(values: Sequence[float], probability: float) -> float | None:
    """Tính percentile tuyến tính; trả None khi rỗng."""

    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if probability == 0.5:
        return float(median(ordered))
    return _quantile(ordered, probability)


def _relative_reduction(candidate: float | None, baseline: float | None) -> float | None:
    """Tính mức giảm tương đối so với baseline dương."""

    if candidate is None or baseline is None or baseline <= 0:
        return None
    return (baseline - candidate) / baseline


def _atomic_write(path: Path, content: str) -> None:
    """Ghi UTF-8 atomic trong cùng directory."""

    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    temporary.replace(path)


def _format_optional(value: float | None) -> str:
    """Format số tùy chọn cho Markdown."""

    return "n/a" if value is None else f"{value:.6f}"


def _format_percent(value: float | None) -> str:
    """Format tỷ lệ tùy chọn cho Markdown."""

    return "n/a" if value is None else f"{value:.2%}"
