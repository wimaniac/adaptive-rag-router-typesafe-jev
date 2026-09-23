"""Kiểm thử grouped split và suy ra gold action counterfactual."""

from __future__ import annotations

from adaptive_rag_router.domain.enums import ModelTier, RetrievalSource
from adaptive_rag_router.evaluation.counterfactual import (
    CounterfactualRunner,
    select_counterfactual_gold_action,
)
from adaptive_rag_router.evaluation.models import BenchmarkSample, CounterfactualOutcome
from adaptive_rag_router.evaluation.split import create_grouped_benchmark_splits


def _sample(index: int) -> BenchmarkSample:
    return BenchmarkSample(
        query_id=f"q-{index:04d}",
        query=f"Question {index}",
        dataset="fixture",
        stratum=f"domain-{index % 5}",
        group_id=f"group-{index:04d}",
    )


def _six_outcomes() -> list[CounterfactualOutcome]:
    outcomes: list[CounterfactualOutcome] = []
    for source_index, source in enumerate(RetrievalSource):
        for tier_index, tier in enumerate(ModelTier):
            outcomes.append(
                CounterfactualOutcome(
                    query_id="q-1",
                    retrieval_source=source,
                    model_tier=tier,
                    quality_score=0.70 + source_index * 0.05 + tier_index * 0.03,
                    cost_usd=0.001 + source_index * 0.001 + tier_index * 0.002,
                    latency_ms=100 + source_index * 10 + tier_index * 20,
                )
            )
    return outcomes


def test_grouped_split_has_exact_default_sizes() -> None:
    samples = [_sample(index) for index in range(1_000)]

    splits = create_grouped_benchmark_splits(samples, seed=123)

    assert len(splits.dev) == 200
    assert len(splits.calibration) == 200
    assert len(splits.test) == 600
    group_sets = [
        {sample.group_id for sample in split}
        for split in (splits.dev, splits.calibration, splits.test)
    ]
    assert group_sets[0].isdisjoint(group_sets[1])
    assert group_sets[0].isdisjoint(group_sets[2])
    assert group_sets[1].isdisjoint(group_sets[2])


def test_counterfactual_selects_cheapest_branch_above_floor() -> None:
    outcomes = _six_outcomes()

    result = select_counterfactual_gold_action(outcomes, quality_floor=0.74)

    assert result.retrieval_source == RetrievalSource.VECTOR
    assert result.model_tier == ModelTier.ECONOMY
    assert result.met_quality_floor is True
    assert result.no_good_route is False


def test_counterfactual_marks_no_good_route_and_picks_best_quality() -> None:
    outcomes = _six_outcomes()

    result = select_counterfactual_gold_action(outcomes, quality_floor=0.99)

    assert result.retrieval_source == RetrievalSource.WEB
    assert result.model_tier == ModelTier.STRONG
    assert result.met_quality_floor is False
    assert result.no_good_route is True


async def test_counterfactual_runner_executes_exactly_six_offline_branches() -> None:
    calls: list[tuple[RetrievalSource, ModelTier]] = []

    async def run_branch(
        sample: BenchmarkSample,
        source: RetrievalSource,
        tier: ModelTier,
    ) -> CounterfactualOutcome:
        calls.append((source, tier))
        quality = 0.8 if source == RetrievalSource.VECTOR else 0.7
        cost = 0.01 if tier == ModelTier.ECONOMY else 0.02
        return CounterfactualOutcome(
            query_id=sample.query_id,
            retrieval_source=source,
            model_tier=tier,
            quality_score=quality,
            cost_usd=cost,
            latency_ms=10,
        )

    result = await CounterfactualRunner(run_branch).run(
        _sample(1),
        quality_floor=0.75,
    )

    assert len(calls) == 6
    assert len(set(calls)) == 6
    assert len(result.outcomes) == 6
    assert result.gold_action.retrieval_source == RetrievalSource.VECTOR
    assert result.gold_action.model_tier == ModelTier.ECONOMY
