"""Điều phối benchmark qua async pipeline được inject và không gọi provider trực tiếp."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC, datetime
from time import perf_counter
from typing import Protocol

from adaptive_rag_router.domain.enums import CompletionStatus, RouterKind
from adaptive_rag_router.domain.errors import CreditBudgetExceededError
from adaptive_rag_router.evaluation.bootstrap import paired_stratified_bootstrap_ci
from adaptive_rag_router.evaluation.metrics import (
    aggregate_router_metrics,
    evaluate_success_criteria,
)
from adaptive_rag_router.evaluation.models import (
    BenchmarkCheckpoint,
    BenchmarkConfig,
    BenchmarkRecord,
    BenchmarkReport,
    BenchmarkSample,
    DatasetSplit,
    PairedComparison,
    PipelineRunResult,
    SuccessCriteriaInputs,
)

type PipelineCallable = Callable[[BenchmarkSample, RouterKind], Awaitable[PipelineRunResult]]


class LiveBudgetHook(Protocol):
    """Hook để adapter live tự quản lý credit mà runner không biết provider."""

    async def reserve(self, sample: BenchmarkSample, router: RouterKind) -> bool:
        """Đặt trước budget cho một lượt chạy, trả False để từ chối an toàn."""

    async def commit(self, record: BenchmarkRecord) -> None:
        """Ghi nhận chi phí/credit thực tế sau khi lượt chạy kết thúc."""


class CheckpointHook(Protocol):
    """Hook lưu tiến độ ra storage do caller quản lý."""

    async def save(self, checkpoint: BenchmarkCheckpoint) -> None:
        """Lưu một snapshot tiến độ có thể tiếp tục về sau."""


class BenchmarkRecordStore(Protocol):
    """Storage lưu records đã hoàn tất để benchmark dài có thể resume."""

    async def load(self) -> tuple[BenchmarkRecord, ...]:
        """Đọc records checkpoint hiện có; file chưa tồn tại trả tuple rỗng."""

    async def save(self, records: Sequence[BenchmarkRecord]) -> None:
        """Ghi atomic toàn bộ records checkpoint hiện tại."""


def _paired_comparison(
    candidate: RouterKind,
    baseline: RouterKind,
    records: Sequence[BenchmarkRecord],
    config: BenchmarkConfig,
) -> PairedComparison:
    """Ghép record theo query ID và tính bootstrap CI cho ba metric."""

    baseline_by_query = {record.query_id: record for record in records if record.router == baseline}
    candidate_by_query = {
        record.query_id: record for record in records if record.router == candidate
    }
    paired_ids = sorted(set(baseline_by_query) & set(candidate_by_query))
    if not paired_ids:
        return PairedComparison(candidate=candidate, baseline=baseline)

    strata = [candidate_by_query[query_id].stratum for query_id in paired_ids]
    cost_interval = paired_stratified_bootstrap_ci(
        [candidate_by_query[query_id].cost_usd for query_id in paired_ids],
        [baseline_by_query[query_id].cost_usd for query_id in paired_ids],
        strata,
        resamples=config.bootstrap_resamples,
        confidence_level=config.confidence_level,
        seed=config.seed,
    )
    latency_interval = paired_stratified_bootstrap_ci(
        [candidate_by_query[query_id].latency_ms for query_id in paired_ids],
        [baseline_by_query[query_id].latency_ms for query_id in paired_ids],
        strata,
        resamples=config.bootstrap_resamples,
        confidence_level=config.confidence_level,
        seed=config.seed + 1,
    )

    quality_ids = [
        query_id
        for query_id in paired_ids
        if candidate_by_query[query_id].quality_score is not None
        and baseline_by_query[query_id].quality_score is not None
    ]
    quality_interval = None
    quality_one_sided_lower_95 = None
    if quality_ids:
        candidate_quality = [candidate_by_query[query_id].quality_score for query_id in quality_ids]
        baseline_quality = [baseline_by_query[query_id].quality_score for query_id in quality_ids]
        assert all(value is not None for value in candidate_quality)
        assert all(value is not None for value in baseline_quality)
        quality_interval = paired_stratified_bootstrap_ci(
            [float(value) for value in candidate_quality if value is not None],
            [float(value) for value in baseline_quality if value is not None],
            [candidate_by_query[query_id].stratum for query_id in quality_ids],
            resamples=config.bootstrap_resamples,
            confidence_level=config.confidence_level,
            seed=config.seed + 2,
        )
        quality_one_sided_lower_95 = paired_stratified_bootstrap_ci(
            [float(value) for value in candidate_quality if value is not None],
            [float(value) for value in baseline_quality if value is not None],
            [candidate_by_query[query_id].stratum for query_id in quality_ids],
            resamples=config.bootstrap_resamples,
            confidence_level=0.90,
            seed=config.seed + 3,
        ).lower
    return PairedComparison(
        candidate=candidate,
        baseline=baseline,
        quality_difference=quality_interval,
        quality_one_sided_lower_95=quality_one_sided_lower_95,
        cost_difference_usd=cost_interval,
        latency_difference_ms=latency_interval,
    )


def build_benchmark_report(
    config: BenchmarkConfig,
    records: Sequence[BenchmarkRecord],
    *,
    started_at: datetime,
    finished_at: datetime,
    denied_by_budget: int = 0,
    stopped_early: bool = False,
    stop_reason: str | None = None,
) -> BenchmarkReport:
    """Tổng hợp record thành report typed với paired comparisons.

    Args:
        config: Cấu hình benchmark gốc.
        records: Các record đã hoàn tất hoặc thất bại có kiểm soát.
        started_at: Thời điểm bắt đầu run.
        finished_at: Thời điểm kết thúc run.
        denied_by_budget: Số lượt bị live budget hook từ chối.
        stopped_early: Run có dừng trước khi dispatch toàn bộ record hay không.
        stop_reason: Lý do dừng có kiểm soát, nếu có.

    Returns:
        BenchmarkReport sẵn sàng ghi artifact.
    """

    metrics = tuple(
        aggregate_router_metrics(router, records, ece_bins=config.ece_bins)
        for router in config.routers
    )
    comparisons = tuple(
        _paired_comparison(router, config.comparison_baseline, records, config)
        for router in config.routers
        if router != config.comparison_baseline
    )
    success_criteria = None
    if config.split is DatasetSplit.TEST and not config.live and config.publish_acceptance:
        quality_lower = next(
            (
                comparison.quality_one_sided_lower_95
                for comparison in comparisons
                if comparison.candidate is RouterKind.JEV and comparison.baseline is RouterKind.LLM
            ),
            None,
        )
        by_router = {item.router: item for item in metrics}
        jev = by_router.get(RouterKind.JEV)
        llm = by_router.get(RouterKind.LLM)
        rule = by_router.get(RouterKind.RULE)
        success_criteria = evaluate_success_criteria(
            SuccessCriteriaInputs(
                quality_difference_ci_lower=quality_lower,
                jev_mean_cost_usd=jev.cost_usd.mean if jev else None,
                llm_mean_cost_usd=llm.cost_usd.mean if llm else None,
                jev_p50_latency_ms=jev.latency_ms.p50 if jev else None,
                llm_p50_latency_ms=llm.latency_ms.p50 if llm else None,
                jev_p95_latency_ms=jev.latency_ms.p95 if jev else None,
                llm_p95_latency_ms=llm.latency_ms.p95 if llm else None,
                critical_class_recall=(
                    jev.critical_route.critical_class_recall if jev and jev.critical_route else None
                ),
                expected_calibration_error=(
                    jev.calibration.expected_calibration_error if jev and jev.calibration else None
                ),
                unhandled_router_schema_error_rate=jev.error_rate if jev else None,
                jev_quality=jev.quality.mean if jev else None,
                rule_quality=rule.quality.mean if rule else None,
                rule_mean_cost_usd=rule.cost_usd.mean if rule else None,
            ),
            config.success_thresholds,
        )
    return BenchmarkReport(
        run_id=config.run_id,
        config=config,
        started_at=started_at,
        finished_at=finished_at,
        sample_count=len({record.query_id for record in records}),
        records=tuple(records),
        router_metrics=metrics,
        comparisons=comparisons,
        success_criteria=success_criteria,
        denied_by_budget=denied_by_budget,
        stopped_early=stopped_early,
        stop_reason=stop_reason,
    )


class BenchmarkRunner:
    """Chạy benchmark trên async pipeline/callable được inject.

    Runner không import hoặc gọi DeepSeek, Jev, Tavily hay vector store. Caller
    cung cấp adapter trả ``PipelineRunResult`` và optional hooks cho live budget,
    checkpoint. Nhờ đó cùng runner dùng được cho fixture, frozen replay và live run.

    Args:
        pipeline: Async callable nhận sample và router.
        samples: Dataset mặc định để ``run(config)`` khớp public interface.
        budget_hook: Hook budget chỉ được kích hoạt khi ``config.live=True``.
        checkpoint_hook: Hook nhận snapshot tiến độ định kỳ.
        record_store: Storage lưu từng batch record để resume sau crash.
    """

    def __init__(
        self,
        pipeline: PipelineCallable,
        *,
        samples: Sequence[BenchmarkSample] | None = None,
        budget_hook: LiveBudgetHook | None = None,
        checkpoint_hook: CheckpointHook | None = None,
        record_store: BenchmarkRecordStore | None = None,
    ) -> None:
        self._pipeline = pipeline
        self._samples = tuple(samples) if samples is not None else None
        self._budget_hook = budget_hook
        self._checkpoint_hook = checkpoint_hook
        self._record_store = record_store

    async def _execute_one(
        self,
        sample: BenchmarkSample,
        router: RouterKind,
        config: BenchmarkConfig,
        semaphore: asyncio.Semaphore,
    ) -> tuple[BenchmarkRecord, bool]:
        """Chạy một cặp query-router và chuyển lỗi thành typed record."""

        async with semaphore:
            if self._budget_hook is not None:
                try:
                    allowed = await self._budget_hook.reserve(sample, router)
                except CreditBudgetExceededError:
                    allowed = False
                if not allowed:
                    return BenchmarkRecord(
                        run_id=config.run_id,
                        query_id=sample.query_id,
                        dataset=sample.dataset,
                        stratum=sample.stratum,
                        group_id=sample.group_id,
                        router=router,
                        expected_route=sample.expected_route,
                        reference_answer=sample.reference_answer,
                        status=CompletionStatus.FAILED,
                        error=("live_budget_denied" if config.live else "cost_budget_denied"),
                        metadata={"budget_denied": True, "controlled_stop": True},
                    ), True

            started = perf_counter()
            denied_by_budget = False
            try:
                result = await self._pipeline(sample, router)
            except CreditBudgetExceededError:
                elapsed_ms = (perf_counter() - started) * 1000.0
                record = BenchmarkRecord(
                    run_id=config.run_id,
                    query_id=sample.query_id,
                    dataset=sample.dataset,
                    stratum=sample.stratum,
                    group_id=sample.group_id,
                    router=router,
                    expected_route=sample.expected_route,
                    reference_answer=sample.reference_answer,
                    latency_ms=elapsed_ms,
                    status=CompletionStatus.FAILED,
                    error="credit_budget_exhausted",
                    metadata={"budget_denied": True, "controlled_stop": True},
                )
                denied_by_budget = True
            except Exception as error:
                if config.fail_fast:
                    raise
                elapsed_ms = (perf_counter() - started) * 1000.0
                record = BenchmarkRecord(
                    run_id=config.run_id,
                    query_id=sample.query_id,
                    dataset=sample.dataset,
                    stratum=sample.stratum,
                    group_id=sample.group_id,
                    router=router,
                    expected_route=sample.expected_route,
                    reference_answer=sample.reference_answer,
                    latency_ms=elapsed_ms,
                    status=CompletionStatus.FAILED,
                    error=type(error).__name__,
                )
            else:
                elapsed_ms = (perf_counter() - started) * 1000.0
                record = BenchmarkRecord(
                    run_id=config.run_id,
                    query_id=sample.query_id,
                    dataset=sample.dataset,
                    stratum=sample.stratum,
                    group_id=sample.group_id,
                    router=router,
                    expected_route=sample.expected_route,
                    predicted_route=result.predicted_route,
                    route_probabilities=result.route_probabilities,
                    reference_answer=sample.reference_answer,
                    answer=result.answer,
                    quality_score=result.quality_score,
                    cost_usd=result.cost_usd,
                    latency_ms=result.latency_ms or elapsed_ms,
                    status=result.status,
                    external_credits=result.external_credits,
                    metadata=result.metadata,
                )

            if self._budget_hook is not None:
                try:
                    await self._budget_hook.commit(record)
                except CreditBudgetExceededError:
                    record = record.model_copy(
                        update={
                            "metadata": {
                                **record.metadata,
                                "budget_exhausted_after_commit": True,
                                "controlled_stop": True,
                            }
                        }
                    )
                    denied_by_budget = True
            return record, denied_by_budget

    async def run(
        self,
        config: BenchmarkConfig,
        samples: Sequence[BenchmarkSample] | None = None,
    ) -> BenchmarkReport:
        """Chạy tất cả router trên các sample và trả report tổng hợp.

        Args:
            config: Cấu hình concurrency, metrics, checkpoint và live mode.
            samples: Sample thuộc split đã chọn. Nếu bỏ qua, dùng sample đã inject
                vào constructor; runner không tự đọc dataset.

        Returns:
            BenchmarkReport chứa record, aggregate và paired bootstrap CI.
        """

        source_samples = samples if samples is not None else self._samples
        if source_samples is None:
            raise ValueError("cần truyền samples vào constructor hoặc run()")
        started_at = datetime.now(UTC)
        selected_samples = (
            list(source_samples[: config.max_samples])
            if config.max_samples
            else list(source_samples)
        )
        semaphore = asyncio.Semaphore(config.max_concurrency)
        all_jobs: list[tuple[BenchmarkSample, RouterKind]] = []
        for sample_index, sample in enumerate(selected_samples):
            routers = (
                tuple(reversed(config.routers))
                if config.alternate_router_order and sample_index % 2 == 1
                else config.routers
            )
            all_jobs.extend((sample, router) for router in routers)
        allowed_keys = {(sample.query_id, router) for sample, router in all_jobs}
        resumed_records = await self._record_store.load() if self._record_store is not None else ()
        records_by_key: dict[tuple[str, RouterKind], BenchmarkRecord] = {}
        for record in resumed_records:
            key = (record.query_id, record.router)
            if record.run_id != config.run_id or key not in allowed_keys:
                raise ValueError("record checkpoint không khớp benchmark run hiện tại")
            if key in records_by_key:
                raise ValueError("record checkpoint chứa query-router trùng lặp")
            # Provider/schema failure được retry khi resume; record thành công
            # không được gọi lại để tránh phát sinh cost lần hai.
            if record.status is not CompletionStatus.FAILED:
                records_by_key[key] = record
                if self._budget_hook is not None:
                    await self._budget_hook.commit(record)
        records = list(records_by_key.values())
        jobs = [
            (sample, router)
            for sample, router in all_jobs
            if (sample.query_id, router) not in records_by_key
        ]
        denied_by_budget = 0
        planned_records = len(all_jobs)
        stopped_early = False
        stop_reason: str | None = None
        next_checkpoint_at = (len(records) // config.checkpoint_every + 1) * config.checkpoint_every
        cursor = 0

        while cursor < len(jobs):
            batch = jobs[cursor : cursor + config.max_concurrency]
            batch_results = await asyncio.gather(
                *(self._execute_one(sample, router, config, semaphore) for sample, router in batch)
            )
            cursor += len(batch)
            budget_denied_in_batch = False
            for record, denied in batch_results:
                records.append(record)
                denied_by_budget += int(denied)
                budget_denied_in_batch = budget_denied_in_batch or denied

            if self._record_store is not None:
                await self._record_store.save(records)

            if self._checkpoint_hook is not None and len(records) >= next_checkpoint_at:
                await self._checkpoint_hook.save(
                    BenchmarkCheckpoint(
                        run_id=config.run_id,
                        completed_records=len(records),
                        planned_records=planned_records,
                        denied_by_budget=denied_by_budget,
                        failed_records=sum(
                            item.status == CompletionStatus.FAILED for item in records
                        ),
                    )
                )
                while next_checkpoint_at <= len(records):
                    next_checkpoint_at += config.checkpoint_every

            should_stop = (
                config.stop_on_budget_denial and budget_denied_in_batch and cursor < len(jobs)
            )
            if should_stop:
                stopped_early = True
                stop_reason = "credit_budget_exhausted" if config.live else "cost_budget_exhausted"
                break

        sample_order = {sample.query_id: index for index, sample in enumerate(selected_samples)}
        router_order = {router: index for index, router in enumerate(config.routers)}
        records.sort(
            key=lambda record: (
                sample_order[record.query_id],
                router_order[record.router],
            )
        )
        if self._checkpoint_hook is not None:
            await self._checkpoint_hook.save(
                BenchmarkCheckpoint(
                    run_id=config.run_id,
                    completed_records=len(records),
                    planned_records=planned_records,
                    denied_by_budget=denied_by_budget,
                    failed_records=sum(item.status == CompletionStatus.FAILED for item in records),
                    stopped_early=stopped_early,
                    stop_reason=stop_reason,
                )
            )

        return build_benchmark_report(
            config,
            records,
            started_at=started_at,
            finished_at=datetime.now(UTC),
            denied_by_budget=denied_by_budget,
            stopped_early=stopped_early,
            stop_reason=stop_reason,
        )
