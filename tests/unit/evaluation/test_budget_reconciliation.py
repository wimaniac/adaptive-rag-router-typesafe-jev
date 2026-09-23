"""Kiểm thử migrate cost artifacts cũ vào provider ledger không gọi API."""

from pathlib import Path

import pytest

from adaptive_rag_router.domain.enums import (
    CompletionStatus,
    ModelTier,
    RetrievalSource,
    RouterKind,
)
from adaptive_rag_router.evaluation.budget_reconciliation import reconcile_provider_budget
from adaptive_rag_router.evaluation.cost_budget import FileProviderBudgetLedger
from adaptive_rag_router.evaluation.models import (
    BenchmarkRecord,
    CounterfactualGoldAction,
    CounterfactualOutcome,
    CounterfactualRunResult,
)
from adaptive_rag_router.evaluation.safety import SafetyAssessmentResult


@pytest.mark.asyncio
async def test_reconcile_provider_budget_counts_legacy_artifacts_conservatively(
    tmp_path: Path,
) -> None:
    """Counterfactual, adaptive và safety cost được map đúng provider."""

    benchmark_root = tmp_path / "benchmarks"
    run_directory = benchmark_root / "run-a"
    run_directory.mkdir(parents=True)
    outcome = CounterfactualOutcome(
        query_id="q-1",
        retrieval_source=RetrievalSource.NONE,
        model_tier=ModelTier.ECONOMY,
        quality_score=0.5,
        cost_usd=0.2,
        latency_ms=1.0,
    )
    counterfactual = CounterfactualRunResult(
        query_id="q-1",
        outcomes=(outcome,),
        gold_action=CounterfactualGoldAction(
            query_id="q-1",
            retrieval_source=RetrievalSource.NONE,
            model_tier=ModelTier.ECONOMY,
            quality_score=0.5,
            cost_usd=0.2,
            latency_ms=1.0,
            met_quality_floor=False,
            no_good_route=True,
        ),
    )
    (run_directory / "counterfactual.jsonl").write_text(
        counterfactual.model_dump_json() + "\n",
        encoding="utf-8",
    )
    record = BenchmarkRecord(
        run_id="run-a",
        query_id="q-1",
        dataset="fixture",
        stratum="fixture",
        group_id="g-1",
        router=RouterKind.JEV,
        cost_usd=0.1,
        status=CompletionStatus.COMPLETED,
    )
    (run_directory / "adaptive-records-checkpoint.jsonl").write_text(
        record.model_dump_json() + "\n",
        encoding="utf-8",
    )
    archived_directory = benchmark_root / "run-b"
    archived_directory.mkdir()
    archived_record = record.model_copy(update={"cost_usd": 0.05})
    (archived_directory / "records.jsonl").write_text(
        archived_record.model_dump_json() + "\n",
        encoding="utf-8",
    )
    safety_path = tmp_path / "safety.jsonl"
    safety = SafetyAssessmentResult(
        variant_id="s-1",
        router=RouterKind.LLM,
        passed=True,
        quality_matches=True,
        accepted_ids_match=True,
        conflicting_ids_match=True,
        injection_ids_match=True,
        cost_usd=0.03,
    )
    safety_path.write_text(safety.model_dump_json() + "\n", encoding="utf-8")
    ledger = FileProviderBudgetLedger(
        tmp_path / "provider.json",
        limits_usd={"deepseek": 2.79, "typesafe-jev": 3.0},
    )

    first = await reconcile_provider_budget(
        benchmark_root,
        ("run-a", "run-b"),
        ledger,
        safety_results_path=safety_path,
    )
    second = await reconcile_provider_budget(
        benchmark_root,
        ("run-a", "run-b"),
        ledger,
        safety_results_path=safety_path,
    )

    assert first.counterfactual_units == 1
    assert first.adaptive_units == 2
    assert first.safety_units == 1
    assert first.consumed_usd == pytest.approx({"deepseek": 0.38, "typesafe-jev": 0.15})
    assert second.consumed_usd == pytest.approx(first.consumed_usd)
