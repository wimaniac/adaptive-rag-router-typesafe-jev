"""Kiểm thử preflight benchmark hoàn toàn offline và không làm lộ secret."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from pydantic import SecretStr

from adaptive_rag_router.config import Settings
from adaptive_rag_router.domain.enums import (
    CompletionStatus,
    ModelTier,
    RetrievalSource,
    RouterKind,
)
from adaptive_rag_router.evaluation import (
    BenchmarkConfig,
    BenchmarkRecord,
    BenchmarkSample,
    CounterfactualGoldAction,
    CounterfactualOutcome,
    CounterfactualRunResult,
    DatasetSplit,
)
from adaptive_rag_router.interfaces.configuration import BenchmarkExecutionPlan
from adaptive_rag_router.interfaces.preflight import build_benchmark_preflight


def _sample(query_id: str) -> BenchmarkSample:
    return BenchmarkSample(
        query_id=query_id,
        query=f"Question {query_id}",
        dataset="fixture",
        stratum="fixture|fact",
        group_id=f"group-{query_id}",
    )


def _plan(tmp_path: Path) -> tuple[BenchmarkExecutionPlan, Settings]:
    mock_web = tmp_path / "mock.jsonl"
    mock_web.write_text("{}\n", encoding="utf-8")
    qdrant = tmp_path / "qdrant"
    qdrant.mkdir()
    benchmark = BenchmarkConfig(
        run_id="formal-preflight",
        routers=(RouterKind.JEV, RouterKind.RULE, RouterKind.LLM),
        split=DatasetSplit.TEST,
        bootstrap_resamples=100,
        output_dir=tmp_path / "artifacts" / "benchmarks",
    )
    plan = BenchmarkExecutionPlan(
        benchmark=benchmark,
        samples=(_sample("q-1"), _sample("q-2")),
        calibration_samples=(_sample("q-cal"),),
        project_root=tmp_path,
        mock_web_path=mock_web,
        vector_top_k=12,
        evidence_limit=8,
        max_repair_rounds=2,
        max_output_tokens=128,
        web_max_results=5,
        web_provider="mock",
    )
    settings = Settings(
        qdrant_path=qdrant,
        typesafe_api_key=SecretStr("typesafe-secret"),
        deepseek_api_key=SecretStr("deepseek-secret"),
    )
    return plan, settings


def test_preflight_reports_readiness_and_exact_phase_sizes(tmp_path: Path) -> None:
    plan, settings = _plan(tmp_path)

    report = build_benchmark_preflight(
        plan,
        settings,
        live=False,
        qdrant_probe=lambda _path, _collection: 36_582,
    )

    assert report.ready is True
    assert report.qdrant_point_count == 36_582
    assert report.ablation_id == "baseline"
    assert report.pipeline_policy["repair_enabled"] is True
    assert report.key_readiness == {"typesafe": True, "deepseek": True, "tavily": True}
    assert [(phase.name, phase.planned, phase.completed) for phase in report.phases] == [
        ("calibration_counterfactual", 1, 0),
        ("calibration_adaptive", 3, 0),
        ("main_counterfactual", 2, 0),
        ("main_adaptive", 6, 0),
    ]
    assert "typesafe-secret" not in repr(report)
    assert "deepseek-secret" not in repr(report)


def test_preflight_estimates_formal_cost_from_typed_smoke_artifacts(tmp_path: Path) -> None:
    plan, settings = _plan(tmp_path)
    smoke = plan.benchmark.output_dir / "smoke-dev-3"
    smoke.mkdir(parents=True)
    outcome = CounterfactualOutcome(
        query_id="smoke-q",
        retrieval_source=RetrievalSource.NONE,
        model_tier=ModelTier.ECONOMY,
        quality_score=0.8,
        cost_usd=0.03,
        latency_ms=10,
    )
    counterfactual = CounterfactualRunResult(
        query_id="smoke-q",
        outcomes=(outcome,),
        gold_action=CounterfactualGoldAction(
            query_id="smoke-q",
            retrieval_source=RetrievalSource.NONE,
            model_tier=ModelTier.ECONOMY,
            quality_score=0.8,
            cost_usd=0.03,
            latency_ms=10,
            met_quality_floor=True,
            no_good_route=False,
        ),
    )
    (smoke / "counterfactual.jsonl").write_text(
        counterfactual.model_dump_json() + "\n",
        encoding="utf-8",
    )
    record = BenchmarkRecord(
        run_id="smoke-dev-3",
        query_id="smoke-q",
        dataset="fixture",
        stratum="fixture|fact",
        group_id="group-smoke",
        router=RouterKind.JEV,
        cost_usd=0.02,
        status=CompletionStatus.COMPLETED,
    )
    (smoke / "adaptive-records-checkpoint.jsonl").write_text(
        record.model_dump_json() + "\n",
        encoding="utf-8",
    )

    report = build_benchmark_preflight(
        plan,
        settings,
        live=False,
        qdrant_probe=lambda _path, _collection: 36_582,
    )

    assert report.estimate_reference_queries == 1
    assert report.estimate_reference_cost_usd == pytest.approx(0.05)
    assert report.estimated_deepseek_cost_low_usd == pytest.approx(0.15)
    assert report.estimated_deepseek_cost_high_usd == pytest.approx(0.2625)


def test_preflight_counts_only_resumable_records(tmp_path: Path) -> None:
    plan, settings = _plan(tmp_path)
    run_directory = plan.benchmark.output_dir / plan.benchmark.run_id
    run_directory.mkdir(parents=True)
    completed = BenchmarkRecord(
        run_id=plan.benchmark.run_id,
        query_id="q-1",
        dataset="fixture",
        stratum="fixture|fact",
        group_id="group-q-1",
        router=RouterKind.JEV,
        status=CompletionStatus.COMPLETED,
    )
    failed = completed.model_copy(
        update={"router": RouterKind.RULE, "status": CompletionStatus.FAILED}
    )
    (run_directory / "adaptive-records-checkpoint.jsonl").write_text(
        completed.model_dump_json() + "\n" + failed.model_dump_json() + "\n",
        encoding="utf-8",
    )

    report = build_benchmark_preflight(
        plan,
        settings,
        live=False,
        qdrant_probe=lambda _path, _collection: 36_582,
    )
    adaptive = next(phase for phase in report.phases if phase.name == "main_adaptive")

    assert adaptive.completed == 1
    assert adaptive.remaining == 5


def test_preflight_validates_frozen_replay_without_planning_calibration(
    tmp_path: Path,
) -> None:
    plan, settings = _plan(tmp_path)
    source_directory = plan.benchmark.output_dir / "baseline-source"
    source_directory.mkdir(parents=True)
    counterfactual_path = source_directory / "counterfactual.jsonl"
    frozen_results = []
    for sample in plan.samples:
        outcome = CounterfactualOutcome(
            query_id=sample.query_id,
            retrieval_source=RetrievalSource.NONE,
            model_tier=ModelTier.ECONOMY,
            quality_score=0.8,
            cost_usd=0.01,
            latency_ms=10,
        )
        frozen_results.append(
            CounterfactualRunResult(
                query_id=sample.query_id,
                outcomes=(outcome,),
                gold_action=CounterfactualGoldAction(
                    query_id=sample.query_id,
                    retrieval_source=RetrievalSource.NONE,
                    model_tier=ModelTier.ECONOMY,
                    quality_score=0.8,
                    cost_usd=0.01,
                    latency_ms=10,
                    met_quality_floor=True,
                    no_good_route=False,
                ),
            )
        )
    counterfactual_path.write_text(
        "".join(result.model_dump_json() + "\n" for result in frozen_results),
        encoding="utf-8",
    )
    calibration_path = source_directory / "calibration-models.json"
    calibration_path.write_text("{}\n", encoding="utf-8")
    replay_plan = replace(
        plan,
        replay_source_run_id="baseline-source",
        replay_counterfactual_path=counterfactual_path,
        replay_calibration_path=calibration_path,
    )

    report = build_benchmark_preflight(
        replay_plan,
        settings,
        live=False,
        qdrant_probe=lambda _path, _collection: 36_582,
    )

    assert report.ready is True
    assert report.calibration_sample_count == 0
    assert report.calibration_artifact_ready is True
    assert report.replay_source_run_id == "baseline-source"
    assert report.replay_counterfactual_path == str(counterfactual_path)
    assert [(phase.name, phase.planned, phase.completed) for phase in report.phases] == [
        ("calibration_counterfactual", 0, 0),
        ("calibration_adaptive", 0, 0),
        ("main_counterfactual", 2, 2),
        ("main_adaptive", 6, 0),
    ]
    assert any("biên bảo thủ" in warning for warning in report.warnings)


def test_preflight_confirmatory_plans_only_adaptive_pairs(tmp_path: Path) -> None:
    plan, settings = _plan(tmp_path)
    calibration_path = tmp_path / "calibration-models.json"
    calibration_path.write_text("{}\n", encoding="utf-8")
    confirmatory_plan = replace(
        plan,
        calibration_samples=(),
        confirmatory_enabled=True,
        confirmatory_looks=(1, 2),
        confirmatory_calibration_path=calibration_path,
    )

    report = build_benchmark_preflight(
        confirmatory_plan,
        settings,
        live=False,
        qdrant_probe=lambda _path, _collection: 36_582,
    )

    assert report.ready is True
    assert report.confirmatory is True
    assert report.confirmatory_looks == (1, 2)
    assert report.calibration_sample_count == 0
    assert report.calibration_artifact_ready is True
    assert [(phase.name, phase.planned) for phase in report.phases] == [
        ("calibration_counterfactual", 0),
        ("calibration_adaptive", 0),
        ("main_counterfactual", 0),
        ("main_adaptive", 6),
    ]
