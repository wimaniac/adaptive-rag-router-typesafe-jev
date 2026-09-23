"""Kiểm thử bề mặt Typer CLI bằng fake coroutine, không dùng network."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from adaptive_rag_router.config import Settings
from adaptive_rag_router.domain.enums import CompletionStatus, RouterKind
from adaptive_rag_router.domain.models import AnswerResponse
from adaptive_rag_router.evaluation import (
    BenchmarkConfig,
    BenchmarkReport,
    BenchmarkSample,
    DatasetSplit,
)
from adaptive_rag_router.interfaces import cli
from adaptive_rag_router.interfaces.preflight import BenchmarkPreflightReport

runner = CliRunner()


def test_query_command_prints_json(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_execute_query(*args: object, **kwargs: object) -> AnswerResponse:
        return AnswerResponse(
            query_id="q-1",
            answer="ok",
            status=CompletionStatus.COMPLETED,
        )

    monkeypatch.setattr(cli, "execute_query", fake_execute_query)

    result = runner.invoke(
        cli.app,
        ["query", "--engine", "rule", "--query", "hello"],
    )

    assert result.exit_code == 0
    assert '"answer": "ok"' in result.stdout


def test_live_command_rejects_budget_above_350() -> None:
    result = runner.invoke(
        cli.app,
        [
            "benchmark-live",
            "--config",
            "missing.toml",
            "--tavily-credit-budget",
            "351",
        ],
    )

    assert result.exit_code != 0


def test_preflight_command_returns_nonzero_when_not_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = BenchmarkPreflightReport(
        run_id="not-ready",
        live=True,
        split="test",
        sample_count=100,
        calibration_sample_count=0,
        routers=("jev", "rule", "llm"),
        ablation_id="baseline",
        pipeline_policy={
            "context_gate_enabled": True,
            "repair_enabled": True,
            "strong_fallback_enabled": True,
        },
        qdrant_ready=True,
        qdrant_point_count=36_582,
        mock_web_ready=True,
        calibration_artifact_ready=False,
        replay_source_run_id=None,
        replay_counterfactual_path=None,
        confirmatory=False,
        confirmatory_looks=(),
        key_readiness={"typesafe": True, "deepseek": True, "tavily": True},
        phases=(),
        estimated_deepseek_cost_low_usd=None,
        estimated_deepseek_cost_high_usd=None,
        estimate_reference_run=None,
        estimate_reference_queries=None,
        estimate_reference_cost_usd=None,
        tavily_credit_budget=350,
        cost_budget_usd=None,
        cost_budget_consumed_usd=None,
        cost_budget_remaining_usd=None,
        provider_budget_path=None,
        provider_budget_limits_usd=None,
        provider_budget_consumed_usd=None,
        provider_budget_remaining_usd=None,
        ready=False,
    )
    monkeypatch.setattr(cli, "load_benchmark_execution_plan", lambda *args, **kwargs: object())
    monkeypatch.setattr(cli, "get_settings", Settings)
    monkeypatch.setattr(cli, "_settings_for_plan", lambda settings, plan: settings)
    monkeypatch.setattr(cli, "build_benchmark_preflight", lambda *args, **kwargs: report)

    result = runner.invoke(
        cli.app,
        ["benchmark-preflight", "--config", "live.toml", "--live"],
    )

    assert result.exit_code == 2
    assert '"ready": false' in result.stdout


def test_read_report_blocks_path_traversal(tmp_path: Path) -> None:
    settings = Settings(artifacts_dir=tmp_path)

    try:
        cli.read_report("../secret", settings)
    except Exception as error:
        assert "run-id" in str(error)
    else:
        raise AssertionError("path traversal phải bị từ chối")


def test_read_report_returns_existing_markdown(tmp_path: Path) -> None:
    report_path = tmp_path / "benchmarks" / "run-1" / "report.md"
    report_path.parent.mkdir(parents=True)
    report_path.write_text("# Report", encoding="utf-8")

    report = cli.read_report("run-1", Settings(artifacts_dir=tmp_path))

    assert report == "# Report"


@pytest.mark.asyncio
async def test_formal_test_calibrates_before_creating_held_out_gold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    qdrant_path = tmp_path / "qdrant"
    qdrant_path.mkdir()
    benchmark = BenchmarkConfig(
        run_id="formal-order",
        routers=(RouterKind.RULE,),
        comparison_baseline=RouterKind.RULE,
        split=DatasetSplit.TEST,
        bootstrap_resamples=100,
        output_dir=tmp_path / "artifacts",
    )
    sample = BenchmarkSample(
        query_id="q-1",
        query="question",
        dataset="fixture",
        stratum="fixture|fact",
        group_id="g-1",
    )
    plan = SimpleNamespace(
        benchmark=benchmark,
        samples=(sample,),
        mock_web_path=tmp_path / "mock.jsonl",
        web_provider="mock",
        counterfactual_quality_floor=0.7,
    )

    class FakeBundle:
        def __init__(self) -> None:
            self.retrievers: dict[object, object] = {}
            self.generator = object()
            self.pipeline = object()
            self.credit_meter = None
            self.tavily_ledger = None

        async def close(self) -> None:
            events.append("close")

    async def fake_fit(*args: object, **kwargs: object) -> dict[RouterKind, object]:
        events.append("calibration")
        return {}

    async def fake_attach(
        samples: tuple[BenchmarkSample, ...],
        *args: object,
        **kwargs: object,
    ) -> tuple[BenchmarkSample, ...]:
        events.append("held-out")
        return samples

    class FakeRunner:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        async def run(self, config: BenchmarkConfig) -> BenchmarkReport:
            events.append("adaptive")
            now = datetime.now(UTC)
            return BenchmarkReport(
                run_id=config.run_id,
                config=config,
                started_at=now,
                finished_at=now,
                sample_count=0,
                records=(),
                router_metrics=(),
            )

    class FakeWriter:
        def __init__(self, output_dir: Path) -> None:
            del output_dir

        def write_all(self, report: BenchmarkReport) -> SimpleNamespace:
            del report
            return SimpleNamespace()

    monkeypatch.setattr(cli, "load_benchmark_execution_plan", lambda *args, **kwargs: plan)
    monkeypatch.setattr(cli, "_settings_for_plan", lambda settings, loaded: settings)
    monkeypatch.setattr(cli, "build_runtime", lambda *args, **kwargs: FakeBundle())
    monkeypatch.setattr(cli, "_fit_formal_calibration", fake_fit)
    monkeypatch.setattr(cli, "attach_counterfactual_gold", fake_attach)
    monkeypatch.setattr(cli, "BenchmarkRunner", FakeRunner)
    monkeypatch.setattr(cli, "BenchmarkArtifactWriter", FakeWriter)

    await cli.execute_benchmark(
        tmp_path / "benchmark.toml",
        live=False,
        settings=Settings(qdrant_path=qdrant_path),
    )

    assert events[:3] == ["calibration", "held-out", "adaptive"]
    assert events[-1] == "close"
