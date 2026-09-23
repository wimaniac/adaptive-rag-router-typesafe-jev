"""Kiểm thử async runner, budget/checkpoint hooks và artifact offline."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from adaptive_rag_router.domain.enums import CompletionStatus, RouterKind
from adaptive_rag_router.domain.errors import CreditBudgetExceededError
from adaptive_rag_router.evaluation.artifacts import BenchmarkArtifactWriter
from adaptive_rag_router.evaluation.models import (
    BenchmarkCheckpoint,
    BenchmarkConfig,
    BenchmarkRecord,
    BenchmarkSample,
    DatasetSplit,
    PipelineRunResult,
)
from adaptive_rag_router.evaluation.runner import BenchmarkRunner


class RecordingCheckpointHook:
    """Fixture lưu checkpoint trong bộ nhớ."""

    def __init__(self) -> None:
        self.checkpoints: list[BenchmarkCheckpoint] = []

    async def save(self, checkpoint: BenchmarkCheckpoint) -> None:
        self.checkpoints.append(checkpoint)


class DenyRuleBudgetHook:
    """Fixture từ chối riêng rule router và ghi các record đã commit."""

    def __init__(self) -> None:
        self.committed: list[BenchmarkRecord] = []

    async def reserve(self, sample: BenchmarkSample, router: RouterKind) -> bool:
        del sample
        return router != RouterKind.RULE

    async def commit(self, record: BenchmarkRecord) -> None:
        self.committed.append(record)


class InMemoryRecordStore:
    """Record store fixture giữ snapshot gần nhất trong bộ nhớ."""

    def __init__(self, records: tuple[BenchmarkRecord, ...] = ()) -> None:
        self.records = records
        self.save_calls = 0

    async def load(self) -> tuple[BenchmarkRecord, ...]:
        """Trả snapshot hiện tại."""

        return self.records

    async def save(self, records: Sequence[BenchmarkRecord]) -> None:
        """Thay snapshot và đếm số lần checkpoint."""

        self.records = tuple(records)
        self.save_calls += 1


def _samples() -> list[BenchmarkSample]:
    return [
        BenchmarkSample(
            query_id=f"q-{index}",
            query=f"Question {index}",
            dataset="fixture",
            stratum=f"s-{index % 2}",
            group_id=f"g-{index}",
            expected_route="vector",
        )
        for index in range(4)
    ]


@pytest.mark.asyncio
async def test_runner_builds_metrics_comparisons_and_checkpoints() -> None:
    async def pipeline(
        sample: BenchmarkSample,
        router: RouterKind,
    ) -> PipelineRunResult:
        del sample
        quality = 0.8 if router == RouterKind.JEV else 0.79
        cost = 0.01 if router == RouterKind.JEV else 0.02
        return PipelineRunResult(
            answer="ok",
            predicted_route="vector",
            route_probabilities={"none": 0.1, "vector": 0.8, "web": 0.1},
            quality_score=quality,
            cost_usd=cost,
            latency_ms=10 if router == RouterKind.JEV else 20,
        )

    checkpoint_hook = RecordingCheckpointHook()
    config = BenchmarkConfig(
        run_id="offline-run",
        routers=(RouterKind.JEV, RouterKind.LLM),
        bootstrap_resamples=100,
        checkpoint_every=3,
    )
    report = await BenchmarkRunner(
        pipeline,
        checkpoint_hook=checkpoint_hook,
    ).run(config, _samples())

    assert len(report.records) == 8
    assert report.sample_count == 4
    assert report.router_metrics[0].routing is not None
    assert report.router_metrics[0].routing.accuracy == 1.0
    assert report.comparisons[0].quality_difference is not None
    assert report.comparisons[0].quality_difference.estimate == pytest.approx(0.01)
    assert report.success_criteria is not None
    assert checkpoint_hook.checkpoints[-1].completed_records == 8

    dev_report = await BenchmarkRunner(pipeline).run(
        config.model_copy(update={"run_id": "dev-run", "split": DatasetSplit.DEV}),
        _samples()[:1],
    )
    assert dev_report.success_criteria is None

    pilot_report = await BenchmarkRunner(pipeline).run(
        config.model_copy(update={"run_id": "pilot", "publish_acceptance": False}),
        _samples()[:1],
    )
    assert pilot_report.success_criteria is None


@pytest.mark.asyncio
async def test_runner_resumes_completed_records_without_provider_replay() -> None:
    calls: list[str] = []

    async def pipeline(
        sample: BenchmarkSample,
        router: RouterKind,
    ) -> PipelineRunResult:
        del router
        calls.append(sample.query_id)
        return PipelineRunResult(answer="ok", predicted_route="vector")

    samples = _samples()
    cached = BenchmarkRecord(
        run_id="resume-run",
        query_id=samples[0].query_id,
        dataset=samples[0].dataset,
        stratum=samples[0].stratum,
        group_id=samples[0].group_id,
        router=RouterKind.JEV,
        expected_route=samples[0].expected_route,
        answer="cached",
        status=CompletionStatus.COMPLETED,
    )
    store = InMemoryRecordStore((cached,))
    config = BenchmarkConfig(
        run_id="resume-run",
        routers=(RouterKind.JEV,),
        comparison_baseline=RouterKind.JEV,
        bootstrap_resamples=100,
        max_concurrency=1,
    )

    first = await BenchmarkRunner(pipeline, record_store=store).run(config, samples)
    second = await BenchmarkRunner(pipeline, record_store=store).run(config, samples)

    assert len(first.records) == 4
    assert len(second.records) == 4
    assert calls == ["q-1", "q-2", "q-3"]
    assert store.save_calls == 3


@pytest.mark.asyncio
async def test_runner_alternates_router_order_by_query() -> None:
    calls: list[tuple[str, RouterKind]] = []

    async def pipeline(
        sample: BenchmarkSample,
        router: RouterKind,
    ) -> PipelineRunResult:
        calls.append((sample.query_id, router))
        return PipelineRunResult(answer="ok")

    config = BenchmarkConfig(
        run_id="alternate-order",
        routers=(RouterKind.JEV, RouterKind.LLM),
        bootstrap_resamples=100,
        max_concurrency=1,
        max_samples=2,
        alternate_router_order=True,
        publish_acceptance=False,
    )

    await BenchmarkRunner(pipeline).run(config, _samples())

    assert calls == [
        ("q-0", RouterKind.JEV),
        ("q-0", RouterKind.LLM),
        ("q-1", RouterKind.LLM),
        ("q-1", RouterKind.JEV),
    ]


@pytest.mark.asyncio
async def test_runner_reconciles_resumed_records_into_budget_hook() -> None:
    """Record resume phải được commit idempotent vào project-wide ledger mới."""

    sample = _samples()[0]
    resumed = BenchmarkRecord(
        run_id="resume-budget",
        query_id=sample.query_id,
        dataset=sample.dataset,
        stratum=sample.stratum,
        group_id=sample.group_id,
        router=RouterKind.JEV,
        cost_usd=0.01,
        status=CompletionStatus.COMPLETED,
    )

    class RecordingBudget:
        """Budget hook fixture ghi record được reconcile."""

        def __init__(self) -> None:
            self.committed: list[BenchmarkRecord] = []

        async def reserve(self, sample: BenchmarkSample, router: RouterKind) -> bool:
            del sample, router
            return True

        async def commit(self, record: BenchmarkRecord) -> None:
            self.committed.append(record)

    async def pipeline(sample: BenchmarkSample, router: RouterKind) -> PipelineRunResult:
        del sample, router
        raise AssertionError("completed record không được replay")

    budget = RecordingBudget()
    report = await BenchmarkRunner(
        pipeline,
        samples=(sample,),
        budget_hook=budget,
        record_store=InMemoryRecordStore((resumed,)),
    ).run(
        BenchmarkConfig(
            run_id="resume-budget",
            routers=(RouterKind.JEV,),
            comparison_baseline=RouterKind.JEV,
            bootstrap_resamples=100,
        )
    )

    assert report.records == (resumed,)
    assert budget.committed == [resumed]


@pytest.mark.asyncio
async def test_live_budget_hook_denies_without_calling_pipeline() -> None:
    calls: list[RouterKind] = []

    async def pipeline(
        sample: BenchmarkSample,
        router: RouterKind,
    ) -> PipelineRunResult:
        del sample
        calls.append(router)
        return PipelineRunResult(answer="ok", predicted_route="vector")

    budget_hook = DenyRuleBudgetHook()
    config = BenchmarkConfig(
        run_id="live-run",
        routers=(RouterKind.JEV, RouterKind.RULE),
        comparison_baseline=RouterKind.JEV,
        bootstrap_resamples=100,
        live=True,
    )
    report = await BenchmarkRunner(pipeline, budget_hook=budget_hook).run(
        config,
        _samples()[:1],
    )

    assert calls == [RouterKind.JEV]
    assert report.denied_by_budget == 1
    assert report.success_criteria is None
    denied = next(record for record in report.records if record.router == RouterKind.RULE)
    assert denied.status == CompletionStatus.FAILED
    assert denied.error == "live_budget_denied"
    assert len(budget_hook.committed) == 1


@pytest.mark.asyncio
async def test_live_runner_stops_dispatch_and_checkpoints_after_budget_denial() -> None:
    calls: list[tuple[str, RouterKind]] = []

    async def pipeline(
        sample: BenchmarkSample,
        router: RouterKind,
    ) -> PipelineRunResult:
        calls.append((sample.query_id, router))
        return PipelineRunResult(answer="ok", predicted_route="vector")

    budget_hook = DenyRuleBudgetHook()
    checkpoint_hook = RecordingCheckpointHook()
    config = BenchmarkConfig(
        run_id="stopped-live-run",
        routers=(RouterKind.JEV, RouterKind.RULE),
        comparison_baseline=RouterKind.JEV,
        bootstrap_resamples=100,
        live=True,
        max_concurrency=1,
        checkpoint_every=1,
    )
    report = await BenchmarkRunner(
        pipeline,
        budget_hook=budget_hook,
        checkpoint_hook=checkpoint_hook,
    ).run(config, _samples())

    assert calls == [("q-0", RouterKind.JEV)]
    assert len(report.records) == 2
    assert report.stopped_early is True
    assert report.stop_reason == "credit_budget_exhausted"
    assert checkpoint_hook.checkpoints[-1].completed_records == 2
    assert checkpoint_hook.checkpoints[-1].planned_records == 8
    assert checkpoint_hook.checkpoints[-1].stopped_early is True


@pytest.mark.asyncio
async def test_live_runner_stops_on_pipeline_credit_budget_exception() -> None:
    calls: list[str] = []

    async def pipeline(
        sample: BenchmarkSample,
        router: RouterKind,
    ) -> PipelineRunResult:
        del router
        calls.append(sample.query_id)
        raise CreditBudgetExceededError("fixture exhausted")

    config = BenchmarkConfig(
        run_id="pipeline-budget-stop",
        routers=(RouterKind.JEV,),
        comparison_baseline=RouterKind.JEV,
        bootstrap_resamples=100,
        live=True,
        max_concurrency=1,
    )
    report = await BenchmarkRunner(pipeline).run(config, _samples())

    assert calls == ["q-0"]
    assert len(report.records) == 1
    assert report.records[0].error == "credit_budget_exhausted"
    assert report.stopped_early is True


@pytest.mark.asyncio
async def test_artifact_writer_outputs_jsonl_markdown_and_parquet(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")

    async def pipeline(
        sample: BenchmarkSample,
        router: RouterKind,
    ) -> PipelineRunResult:
        del sample, router
        return PipelineRunResult(
            answer="answer",
            predicted_route="vector",
            route_probabilities={"vector": 1.0},
            quality_score=1.0,
            cost_usd=0.001,
            latency_ms=5,
        )

    config = BenchmarkConfig(
        run_id="artifact-run",
        routers=(RouterKind.JEV,),
        comparison_baseline=RouterKind.JEV,
        bootstrap_resamples=100,
    )
    report = await BenchmarkRunner(pipeline).run(config, _samples()[:1])

    paths = BenchmarkArtifactWriter(tmp_path).write_all(report)

    assert paths.records_jsonl.is_file()
    assert paths.records_parquet.is_file()
    assert paths.report_markdown.is_file()
    assert paths.pareto_svg.is_file()
    assert paths.reliability_svg.is_file()
    assert paths.risk_coverage_svg.is_file()
    assert "Quality-cost Pareto" in paths.pareto_svg.read_text(encoding="utf-8")
    payload = json.loads(paths.records_jsonl.read_text(encoding="utf-8"))
    assert payload["router"] == "jev"
    markdown = paths.report_markdown.read_text(encoding="utf-8")
    assert "Báo cáo benchmark" in markdown
    assert "Quality lower (one-sided 95%)" in markdown
    assert "jev" in markdown
