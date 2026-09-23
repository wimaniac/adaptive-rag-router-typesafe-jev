"""Kiểm thử so sánh repeatability và artifacts JSON/Markdown của hai run."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from adaptive_rag_router.domain.enums import CompletionStatus, RouterKind
from adaptive_rag_router.evaluation.models import BenchmarkRecord
from adaptive_rag_router.evaluation.repeatability import (
    RepeatabilityReport,
    compare_repeatability,
    write_repeatability_report,
)


def _record(
    run_id: str,
    query_id: str,
    router: RouterKind,
    *,
    predicted_route: str,
    answer: str,
    quality: float,
    cost: float,
    latency: float,
    status: CompletionStatus = CompletionStatus.COMPLETED,
) -> BenchmarkRecord:
    return BenchmarkRecord(
        run_id=run_id,
        query_id=query_id,
        dataset="fixture",
        stratum="fixture",
        group_id=query_id,
        router=router,
        expected_route="vector",
        predicted_route=predicted_route,
        answer=answer,
        quality_score=quality,
        cost_usd=cost,
        latency_ms=latency,
        status=status,
    )


def test_compare_repeatability_calculates_paired_router_metrics() -> None:
    baseline = (
        _record(
            "baseline",
            "q-1",
            RouterKind.JEV,
            predicted_route="vector",
            answer="alpha beta",
            quality=0.8,
            cost=0.01,
            latency=10,
        ),
        _record(
            "baseline",
            "q-1",
            RouterKind.LLM,
            predicted_route="web",
            answer="gamma",
            quality=0.7,
            cost=0.03,
            latency=30,
        ),
    )
    repeat = (
        _record(
            "repeat",
            "q-1",
            RouterKind.JEV,
            predicted_route="vector",
            answer="alpha beta",
            quality=0.75,
            cost=0.012,
            latency=12,
        ),
        _record(
            "repeat",
            "q-1",
            RouterKind.LLM,
            predicted_route="vector",
            answer="delta",
            quality=0.6,
            cost=0.025,
            latency=25,
            status=CompletionStatus.ABSTAINED,
        ),
    )

    report = compare_repeatability(baseline, repeat)

    assert report.paired_query_count == 1
    assert report.baseline_only_records == 0
    assert report.repeat_only_records == 0
    assert report.gold_drift_queries == 0
    assert len(report.baseline_gold_sha256) == 64
    assert report.baseline_gold_sha256 == report.repeat_gold_sha256
    jev = next(item for item in report.routers if item.router is RouterKind.JEV)
    llm = next(item for item in report.routers if item.router is RouterKind.LLM)
    assert jev.predicted_route_agreement == 1.0
    assert jev.baseline_gold_accuracy == 1.0
    assert jev.repeat_fixed_gold_accuracy == 1.0
    assert jev.repeat_native_gold_accuracy == 1.0
    assert jev.answer_token_f1 == 1.0
    assert jev.mean_absolute_quality_delta == pytest.approx(0.05)
    assert jev.mean_cost_delta_usd == pytest.approx(0.002)
    assert jev.mean_latency_delta_ms == pytest.approx(2.0)
    assert llm.predicted_route_agreement == 0.0
    assert llm.baseline_gold_accuracy == 0.0
    assert llm.repeat_fixed_gold_accuracy == 1.0
    assert llm.status_agreement == 0.0
    assert llm.answer_token_f1 == 0.0


def test_compare_repeatability_reports_unpaired_records_and_rejects_duplicates() -> None:
    shared = _record(
        "baseline",
        "q-1",
        RouterKind.RULE,
        predicted_route="none",
        answer="answer",
        quality=1.0,
        cost=0.0,
        latency=1.0,
    )
    baseline_only = shared.model_copy(update={"query_id": "q-baseline"})
    repeat_shared = shared.model_copy(update={"run_id": "repeat"})
    repeat_only = repeat_shared.model_copy(update={"query_id": "q-repeat"})

    report = compare_repeatability((shared, baseline_only), (repeat_shared, repeat_only))

    assert report.baseline_only_records == 1
    assert report.repeat_only_records == 1
    with pytest.raises(ValueError, match="trùng key"):
        compare_repeatability((shared, shared), (repeat_shared,))


def test_compare_repeatability_reports_gold_drift_against_fixed_baseline() -> None:
    baseline = _record(
        "baseline",
        "q-1",
        RouterKind.JEV,
        predicted_route="vector",
        answer="answer",
        quality=1.0,
        cost=0.01,
        latency=1.0,
    )
    repeat = baseline.model_copy(
        update={
            "run_id": "repeat",
            "expected_route": "web",
            "predicted_route": "vector",
        }
    )

    report = compare_repeatability((baseline,), (repeat,))

    assert report.gold_drift_queries == 1
    assert report.baseline_gold_sha256 != report.repeat_gold_sha256
    assert report.routers[0].repeat_fixed_gold_accuracy == 1.0
    assert report.routers[0].repeat_native_gold_accuracy == 0.0


def test_write_repeatability_report_writes_valid_artifacts(tmp_path: Path) -> None:
    baseline = _record(
        "baseline",
        "q-1",
        RouterKind.JEV,
        predicted_route="vector",
        answer="same answer",
        quality=0.8,
        cost=0.01,
        latency=10,
    )
    repeat = baseline.model_copy(update={"run_id": "repeat"})
    report = compare_repeatability((baseline,), (repeat,))

    json_path, markdown_path = write_repeatability_report(tmp_path, report)

    restored = RepeatabilityReport.model_validate_json(json_path.read_text(encoding="utf-8"))
    assert restored == report
    assert json.loads(json_path.read_text(encoding="utf-8"))["paired_query_count"] == 1
    assert "| jev | 1 |" in markdown_path.read_text(encoding="utf-8")
