"""Suy ra gold action từ sáu nhánh counterfactual theo quality floor và chi phí."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence

from adaptive_rag_router.domain.enums import ModelTier, RetrievalSource
from adaptive_rag_router.evaluation.models import (
    BenchmarkSample,
    CounterfactualGoldAction,
    CounterfactualOutcome,
    CounterfactualRunResult,
)

BRANCH_ORDER = tuple(
    (source, tier)
    for source in (
        RetrievalSource.NONE,
        RetrievalSource.VECTOR,
        RetrievalSource.WEB,
    )
    for tier in (ModelTier.ECONOMY, ModelTier.STRONG)
)
EXPECTED_BRANCHES = frozenset(BRANCH_ORDER)

type CounterfactualBranchCallable = Callable[
    [BenchmarkSample, RetrievalSource, ModelTier],
    Awaitable[CounterfactualOutcome],
]


def select_counterfactual_gold_action(
    outcomes: Sequence[CounterfactualOutcome],
    *,
    quality_floor: float,
    require_all_branches: bool = True,
) -> CounterfactualGoldAction:
    """Chọn nhánh rẻ nhất đạt quality floor, rồi ưu tiên latency thấp.

    Nếu không nhánh thành công nào đạt floor, hàm chọn nhánh có quality cao nhất,
    dùng cost và latency để phá hòa, đồng thời đánh dấu ``no_good_route``.

    Args:
        outcomes: Kết quả counterfactual của cùng một query.
        quality_floor: Ngưỡng chất lượng trong [0, 1].
        require_all_branches: Có bắt buộc đủ 3 nguồn x 2 model tier hay không.

    Returns:
        Gold action ổn định để dùng làm nhãn benchmark.

    Raises:
        ValueError: Khi outcomes sai query, trùng/thiếu nhánh hoặc tất cả thất bại.
    """

    if not 0 <= quality_floor <= 1:
        raise ValueError("quality_floor phải nằm trong [0, 1]")
    if not outcomes:
        raise ValueError("outcomes không được rỗng")

    query_ids = {outcome.query_id for outcome in outcomes}
    if len(query_ids) != 1:
        raise ValueError("mọi counterfactual outcome phải thuộc cùng query_id")

    observed_branches = [(outcome.retrieval_source, outcome.model_tier) for outcome in outcomes]
    if len(set(observed_branches)) != len(observed_branches):
        raise ValueError("counterfactual outcomes chứa branch trùng lặp")
    if require_all_branches and set(observed_branches) != EXPECTED_BRANCHES:
        missing = EXPECTED_BRANCHES - set(observed_branches)
        extra = set(observed_branches) - EXPECTED_BRANCHES
        raise ValueError(f"cần đúng sáu branches; thiếu={missing}, thừa={extra}")

    succeeded = [outcome for outcome in outcomes if outcome.succeeded]
    if not succeeded:
        raise ValueError("không có counterfactual branch nào chạy thành công")

    qualified = [outcome for outcome in succeeded if outcome.quality_score >= quality_floor]
    if qualified:
        selected = min(
            qualified,
            key=lambda outcome: (
                outcome.cost_usd,
                outcome.latency_ms,
                -outcome.quality_score,
                outcome.retrieval_source.value,
                outcome.model_tier.value,
            ),
        )
        met_quality_floor = True
        no_good_route = False
    else:
        selected = min(
            succeeded,
            key=lambda outcome: (
                -outcome.quality_score,
                outcome.cost_usd,
                outcome.latency_ms,
                outcome.retrieval_source.value,
                outcome.model_tier.value,
            ),
        )
        met_quality_floor = False
        no_good_route = True

    return CounterfactualGoldAction(
        query_id=selected.query_id,
        retrieval_source=selected.retrieval_source,
        model_tier=selected.model_tier,
        quality_score=selected.quality_score,
        cost_usd=selected.cost_usd,
        latency_ms=selected.latency_ms,
        met_quality_floor=met_quality_floor,
        no_good_route=no_good_route,
    )


class CounterfactualRunner:
    """Chạy đủ sáu nhánh offline qua callable được inject.

    Runner không phụ thuộc generator, retriever hay network provider. Callable
    chịu trách nhiệm chạy một branch và trả outcome đã chấm quality.

    Args:
        branch_callable: Async callable nhận sample, retrieval source và model tier.
        max_concurrency: Số branch được phép chạy đồng thời.
    """

    def __init__(
        self,
        branch_callable: CounterfactualBranchCallable,
        *,
        max_concurrency: int = 6,
    ) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency phải dương")
        self._branch_callable = branch_callable
        self._max_concurrency = max_concurrency

    async def run(
        self,
        sample: BenchmarkSample,
        *,
        quality_floor: float,
    ) -> CounterfactualRunResult:
        """Chạy sáu branch và suy ra gold action cho một sample.

        Args:
            sample: Benchmark sample cần chạy counterfactual.
            quality_floor: Ngưỡng quality dùng chọn branch rẻ nhất.

        Returns:
            CounterfactualRunResult theo branch order ổn định.

        Raises:
            ValueError: Khi callable trả outcome không khớp branch được yêu cầu.
        """

        semaphore = asyncio.Semaphore(self._max_concurrency)

        async def execute(
            source: RetrievalSource,
            tier: ModelTier,
        ) -> CounterfactualOutcome:
            """Chạy một branch dưới semaphore và kiểm tra identity."""

            async with semaphore:
                outcome = await self._branch_callable(sample, source, tier)
            if (
                outcome.query_id != sample.query_id
                or outcome.retrieval_source != source
                or outcome.model_tier != tier
            ):
                raise ValueError("counterfactual callable trả outcome sai branch")
            return outcome

        outcomes = tuple(
            await asyncio.gather(*(execute(source, tier) for source, tier in BRANCH_ORDER))
        )
        gold_action = select_counterfactual_gold_action(
            outcomes,
            quality_floor=quality_floor,
        )
        return CounterfactualRunResult(
            query_id=sample.query_id,
            outcomes=outcomes,
            gold_action=gold_action,
        )
