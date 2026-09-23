"""Kiểm thử metric routing, calibration, aggregate và paired bootstrap."""

from __future__ import annotations

import math

import pytest

from adaptive_rag_router.domain.enums import CompletionStatus, RouterKind
from adaptive_rag_router.evaluation.bootstrap import paired_stratified_bootstrap_ci
from adaptive_rag_router.evaluation.metrics import (
    aggregate_router_metrics,
    calculate_answer_quality_metrics,
    calculate_calibration_metrics,
    calculate_citation_metrics,
    calculate_one_vs_rest_metrics,
    calculate_ordinal_metrics,
    calculate_retrieval_metrics,
    calculate_risk_coverage_curve,
    calculate_routing_metrics,
    evaluate_success_criteria,
    summarize_numeric,
)
from adaptive_rag_router.evaluation.models import (
    BenchmarkRecord,
    CriterionStatus,
    SuccessCriteriaInputs,
    SuccessCriteriaThresholds,
)


def test_aggregate_router_metrics_wires_extended_benchmark_evidence() -> None:
    records = [
        BenchmarkRecord(
            run_id="run",
            query_id="critical",
            dataset="fixture",
            stratum="s",
            group_id="g-1",
            router=RouterKind.JEV,
            expected_route="web:strong",
            predicted_route="web:strong",
            route_probabilities={"none:economy": 0.1, "web:strong": 0.9},
            reference_answer="Paris",
            answer="Paris",
            quality_score=1.0,
            status=CompletionStatus.COMPLETED,
            metadata={
                "citations": ["doc-1"],
                "supporting_document_ids": ["doc-1"],
                "retrieved_document_ids": ["doc-1", "doc-x"],
                "expected_context_quality": "sufficient",
                "predicted_context_quality": "sufficient",
                "used_retrieval": True,
                "used_strong_fallback": True,
                "total_tokens": 120,
            },
        ),
        BenchmarkRecord(
            run_id="run",
            query_id="standard",
            dataset="fixture",
            stratum="s",
            group_id="g-2",
            router=RouterKind.JEV,
            expected_route="none:economy",
            predicted_route="none:economy",
            route_probabilities={"none:economy": 0.8, "web:strong": 0.2},
            reference_answer="blue",
            answer="blue",
            quality_score=1.0,
            status=CompletionStatus.COMPLETED,
            metadata={"total_tokens": 20},
        ),
    ]

    metrics = aggregate_router_metrics(RouterKind.JEV, records)

    assert metrics.ordinal_model_tier is not None
    assert metrics.ordinal_model_tier.mean_absolute_error == 0.0
    assert metrics.critical_route is not None
    assert metrics.critical_route.critical_class_recall == 1.0
    assert metrics.answer_quality is not None
    assert metrics.answer_quality.token_f1 == pytest.approx(1.0)
    assert metrics.citations is not None and metrics.citations.f1 == 1.0
    assert metrics.retrieval is not None and metrics.retrieval.supporting_fact_recall == 1.0
    assert metrics.total_tokens.mean == 70.0
    assert metrics.retrieval_rate == 0.5
    assert metrics.fallback_rate == 0.5


def test_routing_metrics_multiclass() -> None:
    metrics = calculate_routing_metrics(
        ["a", "a", "b", "b"],
        ["a", "b", "b", "b"],
    )

    assert metrics.accuracy == pytest.approx(0.75)
    assert metrics.macro_f1 == pytest.approx((2 / 3 + 0.8) / 2)
    assert metrics.balanced_accuracy == pytest.approx(0.75)
    assert metrics.confusion_matrix["a"] == {"a": 1, "b": 1}


def test_calibration_metrics_known_values() -> None:
    metrics = calculate_calibration_metrics(
        ["a", "b"],
        [{"a": 0.8, "b": 0.2}, {"a": 0.1, "b": 0.9}],
        bin_count=5,
    )

    assert metrics.brier_score == pytest.approx(0.05)
    assert metrics.negative_log_likelihood == pytest.approx((-math.log(0.8) - math.log(0.9)) / 2)
    assert metrics.expected_calibration_error == pytest.approx(0.15)


def test_numeric_summary_percentiles() -> None:
    summary = summarize_numeric([1.0, 2.0, 3.0, 4.0])

    assert summary.mean == pytest.approx(2.5)
    assert summary.p50 == pytest.approx(2.5)
    assert summary.p95 == pytest.approx(3.85)
    assert summary.p99 == pytest.approx(3.97)


def test_paired_stratified_bootstrap_constant_difference() -> None:
    interval = paired_stratified_bootstrap_ci(
        [0.8, 0.9, 0.7, 0.6],
        [0.7, 0.8, 0.6, 0.5],
        ["x", "x", "y", "y"],
        resamples=200,
        seed=7,
    )

    assert interval.estimate == pytest.approx(0.1)
    assert interval.lower == pytest.approx(0.1)
    assert interval.upper == pytest.approx(0.1)
    assert interval.pair_count == 4


def test_ordinal_metrics_respect_label_distance() -> None:
    metrics = calculate_ordinal_metrics(
        ["low", "medium", "high", "high"],
        ["low", "high", "medium", "high"],
        ordered_labels=("low", "medium", "high"),
    )

    assert metrics.mean_absolute_error == pytest.approx(0.5)
    assert -1.0 <= metrics.quadratic_weighted_kappa <= 1.0
    perfect = calculate_ordinal_metrics(
        ["low", "medium", "high"],
        ["low", "medium", "high"],
        ordered_labels=("low", "medium", "high"),
    )
    assert perfect.quadratic_weighted_kappa == 1.0


def test_one_vs_rest_metrics_include_critical_recall() -> None:
    metrics = calculate_one_vs_rest_metrics(
        ["web", "none", "web", "vector"],
        [0.9, 0.8, 0.7, 0.1],
        positive_label="web",
        threshold=0.75,
    )

    assert metrics.auroc == pytest.approx(0.75)
    assert metrics.auprc == pytest.approx(5 / 6)
    assert metrics.critical_class_recall == pytest.approx(0.5)


def test_one_vs_rest_marks_undefined_metrics_without_positive_class() -> None:
    metrics = calculate_one_vs_rest_metrics(
        ["none", "vector"],
        [0.1, 0.2],
        positive_label="web",
    )

    assert metrics.auroc is None
    assert metrics.auprc is None
    assert metrics.critical_class_recall is None


def test_answer_quality_supports_multiple_references() -> None:
    metrics = calculate_answer_quality_metrics(
        ["The Eiffel Tower", "Paris France"],
        [["Eiffel tower", "Tour Eiffel"], "Paris"],
    )

    assert metrics.exact_match == pytest.approx(0.5)
    assert metrics.token_f1 == pytest.approx((1.0 + 2 / 3) / 2)


def test_citation_metrics_are_micro_averaged_and_deduplicated() -> None:
    metrics = calculate_citation_metrics(
        [("d1", "bad", "d1"), ()],
        [("d1", "d2"), ("d3",)],
    )

    assert metrics.predicted_citation_count == 2
    assert metrics.reference_citation_count == 3
    assert metrics.supported_citation_count == 1
    assert metrics.precision == pytest.approx(0.5)
    assert metrics.recall == pytest.approx(1 / 3)
    assert metrics.f1 == pytest.approx(0.4)


def test_risk_coverage_curve_orders_by_confidence() -> None:
    metrics = calculate_risk_coverage_curve(
        [0.9, 0.8, 0.2],
        [1.0, 0.0, 0.0],
    )

    assert [point.coverage for point in metrics.points] == pytest.approx([1 / 3, 2 / 3, 1.0])
    assert [point.risk for point in metrics.points] == pytest.approx([0.0, 0.5, 2 / 3])
    assert metrics.area_under_curve == pytest.approx(5 / 18)


def test_retrieval_metrics_cover_facts_context_and_insufficient_recall() -> None:
    metrics = calculate_retrieval_metrics(
        [("a", "b"), ("c",), ()],
        [("a",), ("c", "x"), ()],
        ["insufficient", "partial", "sufficient"],
        ["insufficient", "sufficient", "sufficient"],
    )

    assert metrics.supporting_fact_recall == pytest.approx(2 / 3)
    assert metrics.context_quality_macro_f1 == pytest.approx((1.0 + 0.0 + 2 / 3) / 3)
    assert metrics.insufficient_recall == 1.0


def test_success_criteria_pass_when_all_locked_thresholds_are_met() -> None:
    result = evaluate_success_criteria(
        SuccessCriteriaInputs(
            quality_difference_ci_lower=-0.01,
            jev_mean_cost_usd=0.7,
            llm_mean_cost_usd=1.0,
            jev_p50_latency_ms=80,
            llm_p50_latency_ms=100,
            jev_p95_latency_ms=104,
            llm_p95_latency_ms=100,
            critical_class_recall=0.91,
            expected_calibration_error=0.07,
            unhandled_router_schema_error_rate=0.004,
            jev_quality=0.82,
            rule_quality=0.80,
            rule_mean_cost_usd=0.8,
        )
    )

    assert result.overall_status == CriterionStatus.PASSED
    assert all(criterion.status == CriterionStatus.PASSED for criterion in result.criteria)


def test_success_criteria_preserve_not_evaluated_state() -> None:
    result = evaluate_success_criteria(SuccessCriteriaInputs())

    assert result.overall_status == CriterionStatus.NOT_EVALUATED
    assert all(criterion.status == CriterionStatus.NOT_EVALUATED for criterion in result.criteria)


def test_success_criteria_use_typed_preregistered_thresholds() -> None:
    result = evaluate_success_criteria(
        SuccessCriteriaInputs(quality_difference_ci_lower=-0.03),
        SuccessCriteriaThresholds(quality_non_inferiority_margin=0.04),
    )

    quality = next(
        criterion for criterion in result.criteria if criterion.name == "quality_non_inferiority"
    )
    assert quality.threshold == pytest.approx(-0.04)
    assert quality.status == CriterionStatus.PASSED
