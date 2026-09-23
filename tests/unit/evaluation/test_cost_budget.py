"""Kiểm thử hard budget USD, idempotency và checkpoint của benchmark pilot."""

from pathlib import Path

import pytest

from adaptive_rag_router.domain.enums import CompletionStatus, RouterKind
from adaptive_rag_router.domain.errors import ConfigurationError, CreditBudgetExceededError
from adaptive_rag_router.evaluation.cost_budget import (
    CombinedBudgetHook,
    CombinedUsdLedger,
    FileProviderBudgetLedger,
    FileUsdBudgetLedger,
    ProviderScopedUsdLedger,
    ProviderUsdBudgetHook,
)
from adaptive_rag_router.evaluation.models import BenchmarkRecord, BenchmarkSample


@pytest.mark.asyncio
async def test_cost_ledger_reserves_commits_and_resumes_idempotently(tmp_path: Path) -> None:
    """Ledger resume không cộng trùng một unit đã commit."""

    path = tmp_path / "cost.json"
    ledger = FileUsdBudgetLedger(path, limit_usd=2.79)

    assert await ledger.reserve("unit-1", reserve_usd=0.50) is True
    await ledger.commit("unit-1", 0.12)
    await ledger.commit("unit-1", 0.12)

    resumed = FileUsdBudgetLedger(path, limit_usd=2.79)
    assert resumed.consumed_usd == pytest.approx(0.12)
    assert resumed.remaining_usd == pytest.approx(2.67)


@pytest.mark.asyncio
async def test_cost_ledger_denies_when_reserve_would_cross_limit(tmp_path: Path) -> None:
    """Reserve chặn đơn vị mới trước khi phần còn lại xuống dưới vùng đệm."""

    ledger = FileUsdBudgetLedger(tmp_path / "cost.json", limit_usd=1.0)
    await ledger.commit("unit-1", 0.70)

    assert await ledger.reserve("unit-2", reserve_usd=0.31) is False
    assert await ledger.reserve("unit-2", reserve_usd=0.30) is True


@pytest.mark.asyncio
async def test_cost_ledger_rejects_mismatched_resume_or_actual_overrun(tmp_path: Path) -> None:
    """Checkpoint lệch cost bị từ chối và actual overrun được ghi trước khi báo lỗi."""

    path = tmp_path / "cost.json"
    ledger = FileUsdBudgetLedger(path, limit_usd=1.0)
    await ledger.commit("unit-1", 0.40)
    with pytest.raises(ConfigurationError, match="Cost ledger lệch"):
        await ledger.commit("unit-1", 0.41)
    with pytest.raises(CreditBudgetExceededError, match="vượt hard limit"):
        await ledger.commit("unit-2", 0.61)

    assert FileUsdBudgetLedger(path, limit_usd=1.0).consumed_usd == pytest.approx(1.01)


@pytest.mark.asyncio
async def test_cost_ledger_replaces_zero_cost_failed_retry(tmp_path: Path) -> None:
    """Retry thành công được thay placeholder 0 mà không cộng trùng unit."""

    path = tmp_path / "cost.json"
    ledger = FileUsdBudgetLedger(path, limit_usd=1.0)
    await ledger.commit("retry-unit", 0.0)
    await ledger.commit("retry-unit", 0.07)

    assert ledger.consumed_usd == pytest.approx(0.07)
    assert len(ledger._entries) == 1


@pytest.mark.asyncio
async def test_combined_budget_requires_every_hook_and_commits_all() -> None:
    """Hook ghép không bỏ qua Tavily hoặc USD guardrail."""

    class Hook:
        def __init__(self, allowed: bool) -> None:
            self.allowed = allowed
            self.commits = 0

        async def reserve(self, sample: BenchmarkSample, router: RouterKind) -> bool:
            del sample, router
            return self.allowed

        async def commit(self, record: BenchmarkRecord) -> None:
            del record
            self.commits += 1

    first = Hook(True)
    second = Hook(False)
    sample = BenchmarkSample(
        query_id="q-1",
        query="question",
        dataset="fixture",
        stratum="fixture",
        group_id="g-1",
    )
    record = BenchmarkRecord(
        run_id="run",
        query_id="q-1",
        dataset="fixture",
        stratum="fixture",
        group_id="g-1",
        router=RouterKind.JEV,
        status=CompletionStatus.COMPLETED,
    )
    combined = CombinedBudgetHook(first, second)

    assert await combined.reserve(sample, RouterKind.JEV) is False
    await combined.commit(record)
    assert first.commits == second.commits == 1


@pytest.mark.asyncio
async def test_provider_ledger_enforces_independent_project_caps(tmp_path: Path) -> None:
    """Project ledger reserve và commit độc lập cho DeepSeek/Jev."""

    path = tmp_path / "providers.json"
    ledger = FileProviderBudgetLedger(
        path,
        limits_usd={"deepseek": 2.79, "typesafe-jev": 3.0},
    )

    assert await ledger.reserve(
        "unit-1",
        reserves_usd={"deepseek": 0.5, "typesafe-jev": 0.05},
    )
    await ledger.commit("unit-1", {"deepseek": 0.2, "typesafe-jev": 0.01})
    await ledger.commit("unit-1", {"deepseek": 0.2, "typesafe-jev": 0.01})

    resumed = FileProviderBudgetLedger(
        path,
        limits_usd={"deepseek": 2.79, "typesafe-jev": 3.0},
    )
    assert resumed.consumed_usd == pytest.approx({"deepseek": 0.2, "typesafe-jev": 0.01})
    assert resumed.remaining_usd == pytest.approx({"deepseek": 2.59, "typesafe-jev": 2.99})
    with pytest.raises(ConfigurationError, match="Provider cost ledger lệch"):
        await resumed.commit("unit-1", {"deepseek": 0.21, "typesafe-jev": 0.01})


@pytest.mark.asyncio
async def test_provider_scoped_and_combined_ledgers_namespace_counterfactual(
    tmp_path: Path,
) -> None:
    """Counterfactual commit đồng thời vào run ledger và project DeepSeek cap."""

    run = FileUsdBudgetLedger(tmp_path / "run.json", limit_usd=1.0)
    project = FileProviderBudgetLedger(
        tmp_path / "project.json",
        limits_usd={"deepseek": 2.79, "typesafe-jev": 3.0},
    )
    scoped = ProviderScopedUsdLedger(
        project,
        provider="deepseek",
        namespace="run-a",
    )
    combined = CombinedUsdLedger(run, scoped)

    assert await combined.reserve("counterfactual:main:q-1", reserve_usd=0.25)
    await combined.commit("counterfactual:main:q-1", 0.12)

    assert run.consumed_usd == pytest.approx(0.12)
    assert project.consumed_usd == pytest.approx({"deepseek": 0.12, "typesafe-jev": 0.0})


@pytest.mark.asyncio
async def test_provider_hook_uses_breakdown_and_conservative_legacy_fallback(
    tmp_path: Path,
) -> None:
    """Adaptive hook dùng breakdown mới và double-count an toàn artifact cũ."""

    ledger = FileProviderBudgetLedger(
        tmp_path / "project.json",
        limits_usd={"deepseek": 2.79, "typesafe-jev": 3.0},
    )
    hook = ProviderUsdBudgetHook(
        ledger,
        run_id="run",
        deepseek_reserve_usd=0.5,
        jev_reserve_usd=0.05,
    )
    sample = BenchmarkSample(
        query_id="q-new",
        query="question",
        dataset="fixture",
        stratum="fixture",
        group_id="g-new",
    )
    assert await hook.reserve(sample, RouterKind.JEV)
    exact = BenchmarkRecord(
        run_id="run",
        query_id="q-new",
        dataset="fixture",
        stratum="fixture",
        group_id="g-new",
        router=RouterKind.JEV,
        cost_usd=0.21,
        status=CompletionStatus.COMPLETED,
        metadata={"provider_costs_usd": {"deepseek": 0.2, "typesafe-jev": 0.01}},
    )
    legacy = exact.model_copy(
        update={
            "query_id": "q-old",
            "group_id": "g-old",
            "cost_usd": 0.1,
            "metadata": {},
        }
    )

    await hook.commit(exact)
    await hook.commit(legacy)

    assert ledger.consumed_usd == pytest.approx({"deepseek": 0.3, "typesafe-jev": 0.11})
