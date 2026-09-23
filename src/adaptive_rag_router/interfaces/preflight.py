"""Kiểm tra readiness, checkpoint và cost estimate trước benchmark có chi phí."""

from __future__ import annotations

import warnings as pywarnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from adaptive_rag_router.config import Settings
from adaptive_rag_router.domain.enums import CompletionStatus, RouterKind
from adaptive_rag_router.domain.errors import ConfigurationError
from adaptive_rag_router.evaluation import BenchmarkRecord, CounterfactualRunResult, DatasetSplit
from adaptive_rag_router.evaluation.cost_budget import (
    FileProviderBudgetLedger,
    FileUsdBudgetLedger,
)
from adaptive_rag_router.interfaces.configuration import BenchmarkExecutionPlan


@dataclass(frozen=True, slots=True)
class PreflightPhase:
    """Tiến độ một phase có checkpoint của benchmark.

    Args:
        name: Tên phase ổn định.
        checkpoint_path: Đường dẫn checkpoint dự kiến.
        planned: Số query hoặc query-router records theo đơn vị phase.
        completed: Số entry hợp lệ hiện có.
    """

    name: str
    checkpoint_path: str
    planned: int
    completed: int

    @property
    def remaining(self) -> int:
        """Trả số entry còn lại, không bao giờ âm."""

        return max(0, self.planned - self.completed)


@dataclass(frozen=True, slots=True)
class BenchmarkPreflightReport:
    """Báo cáo offline trước khi cho phép benchmark gọi provider.

    Args:
        run_id: Run ID sẽ dùng cho artifact/checkpoint.
        live: Có phải Tavily live track hay không.
        split: Dataset split được chạy.
        sample_count: Số query trong phase chính.
        calibration_sample_count: Số query calibration chạy trước held-out.
        routers: Danh sách router theo thứ tự cấu hình.
        ablation_id: Baseline hoặc component ablation identifier.
        pipeline_policy: Typed switches sẽ áp cho adaptive pipeline.
        qdrant_ready: Qdrant local path đã tồn tại.
        qdrant_point_count: Số point exact trong collection, nếu đọc được.
        mock_web_ready: Frozen web fixture đã tồn tại.
        calibration_artifact_ready: Artifact calibration bắt buộc cho live đã có.
        replay_source_run_id: Run nguồn cung cấp frozen gold/calibration, nếu có.
        replay_counterfactual_path: Frozen counterfactual path của replay, nếu có.
        confirmatory: Có phải paired budget-constrained track hay không.
        confirmatory_looks: Các mốc group-sequential đã khóa.
        key_readiness: Trạng thái API key cần thiết, không chứa secret.
        phases: Tiến độ từng checkpoint.
        estimated_deepseek_cost_low_usd: Nội suy trực tiếp từ smoke, nếu có.
        estimated_deepseek_cost_high_usd: Biên bảo thủ 1,75 lần, nếu có.
        estimate_reference_run: Run smoke dùng làm bằng chứng nội suy.
        estimate_reference_queries: Số query trong run tham chiếu.
        estimate_reference_cost_usd: Tổng cost ghi trong run tham chiếu.
        tavily_credit_budget: Hard cap live track, nếu có.
        cost_budget_usd: Hard cap USD offline, nếu có.
        cost_budget_consumed_usd: Cost đã checkpoint trong ledger.
        cost_budget_remaining_usd: Phần USD còn lại theo ledger.
        provider_budget_path: Project-wide ledger path, nếu được cấu hình.
        provider_budget_limits_usd: Hard cap riêng theo provider.
        provider_budget_consumed_usd: Cost project-wide đã reconcile theo provider.
        provider_budget_remaining_usd: Phần hard cap còn lại theo provider.
        ready: Mọi prerequisite bắt buộc đã sẵn sàng.
        warnings: Cảnh báo vận hành không chứa secret.
    """

    run_id: str
    live: bool
    split: str
    sample_count: int
    calibration_sample_count: int
    routers: tuple[str, ...]
    ablation_id: str
    pipeline_policy: dict[str, bool]
    qdrant_ready: bool
    qdrant_point_count: int | None
    mock_web_ready: bool
    calibration_artifact_ready: bool | None
    replay_source_run_id: str | None
    replay_counterfactual_path: str | None
    confirmatory: bool
    confirmatory_looks: tuple[int, ...]
    key_readiness: dict[str, bool]
    phases: tuple[PreflightPhase, ...]
    estimated_deepseek_cost_low_usd: float | None
    estimated_deepseek_cost_high_usd: float | None
    estimate_reference_run: str | None
    estimate_reference_queries: int | None
    estimate_reference_cost_usd: float | None
    tavily_credit_budget: int | None
    cost_budget_usd: float | None
    cost_budget_consumed_usd: float | None
    cost_budget_remaining_usd: float | None
    provider_budget_path: str | None
    provider_budget_limits_usd: dict[str, float] | None
    provider_budget_consumed_usd: dict[str, float] | None
    provider_budget_remaining_usd: dict[str, float] | None
    ready: bool
    warnings: tuple[str, ...] = ()


def build_benchmark_preflight(
    plan: BenchmarkExecutionPlan,
    settings: Settings,
    *,
    live: bool,
    estimate_reference_run: str = "smoke-dev-3",
    qdrant_probe: Callable[[Path, str], int | None] | None = None,
) -> BenchmarkPreflightReport:
    """Dựng preflight report chỉ từ config, filesystem và artifact đã có.

    Hàm không khởi tạo model, SDK client hoặc gọi network/provider.

    Args:
        plan: Execution plan đã parse và validate.
        settings: Settings đã resolve về project root.
        live: Có kiểm tra prerequisite của Tavily live track hay không.
        estimate_reference_run: Run smoke dùng để nội suy chi phí formal.
        qdrant_probe: Probe inject cho test; mặc định đọc exact collection count.

    Returns:
        Báo cáo readiness và tiến độ có thể serialize bằng ``dataclasses.asdict``.

    Raises:
        ConfigurationError: Khi checkpoint tồn tại nhưng hỏng schema.
    """

    sample_count = min(
        len(plan.samples),
        plan.benchmark.max_samples or len(plan.samples),
    )
    replay_source_run_id = getattr(plan, "replay_source_run_id", None)
    replay_counterfactual_path = getattr(plan, "replay_counterfactual_path", None)
    replay_calibration_path = getattr(plan, "replay_calibration_path", None)
    replay_enabled = not live and replay_source_run_id is not None
    confirmatory_enabled = not live and getattr(plan, "confirmatory_enabled", False)
    confirmatory_calibration_path = getattr(plan, "confirmatory_calibration_path", None)
    calibration_count = (
        len(plan.calibration_samples)
        if not live
        and plan.benchmark.split is DatasetSplit.TEST
        and not replay_enabled
        and not confirmatory_enabled
        else 0
    )
    run_directory = plan.benchmark.output_dir / plan.benchmark.run_id
    router_count = len(plan.benchmark.routers)
    selected_samples = plan.samples[:sample_count]
    main_query_ids = {sample.query_id for sample in selected_samples}
    calibration_query_ids = {sample.query_id for sample in plan.calibration_samples}
    router_set = set(plan.benchmark.routers)
    calibration_counterfactual_path = run_directory / "calibration-counterfactual.jsonl"
    main_counterfactual_path = (
        replay_counterfactual_path
        if replay_enabled and replay_counterfactual_path is not None
        else run_directory / "counterfactual.jsonl"
    )
    phases = (
        PreflightPhase(
            name="calibration_counterfactual",
            checkpoint_path=str(
                replay_calibration_path
                if replay_enabled and replay_calibration_path is not None
                else calibration_counterfactual_path
            ),
            planned=calibration_count,
            completed=(
                0
                if replay_enabled
                else _counterfactual_checkpoint_count(
                    calibration_counterfactual_path,
                    calibration_query_ids,
                )
            ),
        ),
        PreflightPhase(
            name="calibration_adaptive",
            checkpoint_path=str(run_directory / "calibration-records-checkpoint.jsonl"),
            planned=calibration_count * router_count,
            completed=_record_checkpoint_count(
                run_directory / "calibration-records-checkpoint.jsonl",
                allowed_query_ids=calibration_query_ids,
                allowed_routers=router_set,
                run_id=f"{plan.benchmark.run_id}-calibration",
            ),
        ),
        PreflightPhase(
            name="main_counterfactual",
            checkpoint_path=str(main_counterfactual_path),
            planned=0 if confirmatory_enabled else sample_count,
            completed=(
                0
                if confirmatory_enabled
                else _counterfactual_checkpoint_count(
                    main_counterfactual_path,
                    main_query_ids,
                )
            ),
        ),
        PreflightPhase(
            name="main_adaptive",
            checkpoint_path=str(run_directory / "adaptive-records-checkpoint.jsonl"),
            planned=sample_count * router_count,
            completed=_record_checkpoint_count(
                run_directory / "adaptive-records-checkpoint.jsonl",
                allowed_query_ids=main_query_ids,
                allowed_routers=router_set,
                run_id=plan.benchmark.run_id,
            ),
        ),
    )

    key_readiness = {
        "typesafe": RouterKind.JEV not in plan.benchmark.routers
        or settings.typesafe_api_key is not None,
        "deepseek": settings.deepseek_api_key is not None,
        "tavily": not live or settings.tavily_api_key is not None,
    }
    probe = qdrant_probe or _qdrant_point_count
    qdrant_point_count = (
        probe(settings.qdrant_path, settings.vector_collection)
        if settings.qdrant_path.is_dir()
        else None
    )
    qdrant_ready = qdrant_point_count is not None and qdrant_point_count > 0
    mock_web_ready = plan.mock_web_path is not None and plan.mock_web_path.is_file()
    if live:
        calibration_ready = (
            plan.calibration_artifact_path is not None and plan.calibration_artifact_path.is_file()
        )
    elif replay_enabled:
        calibration_ready = (
            replay_calibration_path is not None and replay_calibration_path.is_file()
        )
    elif confirmatory_enabled:
        calibration_ready = (
            confirmatory_calibration_path is not None and confirmatory_calibration_path.is_file()
        )
    else:
        calibration_ready = None

    warnings: list[str] = []
    if qdrant_point_count is not None and qdrant_point_count > 20_000:
        warnings.append(
            "Qdrant local vượt 20.000 points; kết quả đúng nhưng formal retrieval có thể chậm."
        )
    if replay_enabled:
        warnings.append(
            "Replay dùng frozen gold/calibration; ước lượng từ smoke cho adaptive phase "
            "là biên bảo thủ."
        )
    if confirmatory_enabled:
        warnings.append(
            "Confirmatory estimate chỉ áp cho adaptive Jev/LLM; không chạy calibration "
            "hoặc six-branch counterfactual mới."
        )

    confirmatory_reference_run_id = getattr(plan, "confirmatory_reference_run_id", None)
    effective_reference_run = (
        confirmatory_reference_run_id
        if confirmatory_enabled and confirmatory_reference_run_id is not None
        else estimate_reference_run
    )
    reference = (
        _adaptive_cost_reference(
            plan.benchmark.output_dir / effective_reference_run,
            plan.benchmark.routers,
        )
        if confirmatory_enabled
        else _smoke_cost_reference(plan.benchmark.output_dir / effective_reference_run)
    )
    estimated_low: float | None = None
    estimated_high: float | None = None
    reference_queries: int | None = None
    reference_cost: float | None = None
    if reference is not None and not live:
        reference_queries, reference_cost = reference
        target_queries = sample_count + calibration_count
        estimated_low = reference_cost / reference_queries * target_queries
        estimated_high = estimated_low * (1.15 if confirmatory_enabled else 1.75)
        warnings.append(
            "Ước lượng DeepSeek là nội suy từ run tham chiếu, không phải provider-side hard cap."
        )
    elif not live:
        warnings.append("Không tìm thấy smoke artifact hợp lệ để ước lượng DeepSeek cost.")

    cost_consumed: float | None = None
    cost_remaining: float | None = None
    if plan.cost_budget_usd is not None:
        ledger = FileUsdBudgetLedger(
            run_directory / "cost-budget-checkpoint.json",
            limit_usd=plan.cost_budget_usd,
        )
        cost_consumed = ledger.consumed_usd
        cost_remaining = ledger.remaining_usd
        if estimated_high is not None and estimated_high > plan.cost_budget_usd:
            warnings.append(
                "Ước lượng cost biên cao vượt hard budget; run có thể dừng sớm tại checkpoint."
            )

    provider_limits: dict[str, float] | None = None
    provider_consumed: dict[str, float] | None = None
    provider_remaining: dict[str, float] | None = None
    if (
        plan.provider_budget_path is not None
        and plan.deepseek_limit_usd is not None
        and plan.jev_limit_usd is not None
    ):
        provider_ledger = FileProviderBudgetLedger(
            plan.provider_budget_path,
            limits_usd={
                "deepseek": plan.deepseek_limit_usd,
                "typesafe-jev": plan.jev_limit_usd,
            },
        )
        provider_limits = provider_ledger.limits_usd
        provider_consumed = provider_ledger.consumed_usd
        provider_remaining = provider_ledger.remaining_usd

    prerequisites = [qdrant_ready, mock_web_ready, *key_readiness.values()]
    if live:
        prerequisites.append(bool(calibration_ready))
    elif replay_enabled:
        main_counterfactual = next(phase for phase in phases if phase.name == "main_counterfactual")
        prerequisites.extend(
            [
                bool(calibration_ready),
                main_counterfactual.completed == main_counterfactual.planned,
            ]
        )
    elif confirmatory_enabled:
        prerequisites.append(bool(calibration_ready))
    return BenchmarkPreflightReport(
        run_id=plan.benchmark.run_id,
        live=live,
        split=plan.benchmark.split.value,
        sample_count=sample_count,
        calibration_sample_count=calibration_count,
        routers=tuple(router.value for router in plan.benchmark.routers),
        ablation_id=plan.pipeline_policy.ablation_id,
        pipeline_policy=plan.pipeline_policy.model_dump(),
        qdrant_ready=qdrant_ready,
        qdrant_point_count=qdrant_point_count,
        mock_web_ready=mock_web_ready,
        calibration_artifact_ready=calibration_ready,
        replay_source_run_id=replay_source_run_id,
        replay_counterfactual_path=(
            str(replay_counterfactual_path) if replay_counterfactual_path is not None else None
        ),
        confirmatory=confirmatory_enabled,
        confirmatory_looks=getattr(plan, "confirmatory_looks", ()),
        key_readiness=key_readiness,
        phases=phases,
        estimated_deepseek_cost_low_usd=estimated_low,
        estimated_deepseek_cost_high_usd=estimated_high,
        estimate_reference_run=effective_reference_run if reference is not None else None,
        estimate_reference_queries=reference_queries,
        estimate_reference_cost_usd=reference_cost,
        tavily_credit_budget=plan.tavily_credit_budget if live else None,
        cost_budget_usd=plan.cost_budget_usd,
        cost_budget_consumed_usd=cost_consumed,
        cost_budget_remaining_usd=cost_remaining,
        provider_budget_path=(
            str(plan.provider_budget_path) if plan.provider_budget_path is not None else None
        ),
        provider_budget_limits_usd=provider_limits,
        provider_budget_consumed_usd=provider_consumed,
        provider_budget_remaining_usd=provider_remaining,
        ready=all(prerequisites),
        warnings=tuple(warnings),
    )


def _counterfactual_checkpoint_count(path: Path, allowed_query_ids: set[str]) -> int:
    if not path.is_file():
        return 0
    seen: set[str] = set()
    _line_number = 0
    try:
        with path.open("r", encoding="utf-8") as handle:
            for _line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                result = CounterfactualRunResult.model_validate_json(line)
                if result.query_id not in allowed_query_ids:
                    raise ValueError("query_id ngoài split hiện tại")
                if result.query_id in seen:
                    raise ValueError("query_id trùng lặp")
                seen.add(result.query_id)
    except (OSError, ValueError) as error:
        raise ConfigurationError(f"Checkpoint hỏng tại dòng {_line_number}: {path}") from error
    return len(seen)


def _qdrant_point_count(path: Path, collection_name: str) -> int | None:
    """Đọc exact point count từ Qdrant local mà không tải embedding model."""

    try:
        from qdrant_client import QdrantClient

        with pywarnings.catch_warnings():
            pywarnings.simplefilter("ignore", UserWarning)
            client = QdrantClient(path=str(path))
        try:
            if not client.collection_exists(collection_name=collection_name):
                return None
            return int(client.count(collection_name=collection_name, exact=True).count)
        finally:
            client.close()
    except Exception:
        return None


def _record_checkpoint_count(
    path: Path,
    *,
    allowed_query_ids: set[str],
    allowed_routers: set[RouterKind],
    run_id: str,
) -> int:
    if not path.is_file():
        return 0
    seen: set[tuple[str, RouterKind]] = set()
    completed = 0
    _line_number = 0
    try:
        with path.open("r", encoding="utf-8") as handle:
            for _line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                record = BenchmarkRecord.model_validate_json(line)
                key = (record.query_id, record.router)
                if (
                    record.run_id != run_id
                    or record.query_id not in allowed_query_ids
                    or record.router not in allowed_routers
                ):
                    raise ValueError("record ngoài benchmark job hiện tại")
                if key in seen:
                    raise ValueError("query-router trùng lặp")
                seen.add(key)
                completed += int(record.status is not CompletionStatus.FAILED)
    except (OSError, ValueError) as error:
        raise ConfigurationError(f"Checkpoint hỏng tại dòng {_line_number}: {path}") from error
    return completed


def _smoke_cost_reference(run_directory: Path) -> tuple[int, float] | None:
    counterfactual_path = run_directory / "counterfactual.jsonl"
    records_path = run_directory / "adaptive-records-checkpoint.jsonl"
    if not counterfactual_path.is_file() or not records_path.is_file():
        return None
    query_ids: set[str] = set()
    total_cost = 0.0
    try:
        with counterfactual_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                result = CounterfactualRunResult.model_validate_json(line)
                query_ids.add(result.query_id)
                total_cost += sum(outcome.cost_usd for outcome in result.outcomes)
        with records_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = BenchmarkRecord.model_validate_json(line)
                query_ids.add(record.query_id)
                total_cost += record.cost_usd
    except (OSError, ValueError):
        return None
    if not query_ids or total_cost <= 0:
        return None
    return len(query_ids), total_cost


def _adaptive_cost_reference(
    run_directory: Path,
    routers: tuple[RouterKind, ...],
) -> tuple[int, float] | None:
    """Đọc riêng adaptive cost của các router confirmatory từ run tham chiếu."""

    records_path = run_directory / "records.jsonl"
    if not records_path.is_file():
        return None
    allowed = set(routers)
    by_query: dict[str, dict[RouterKind, float]] = {}
    try:
        with records_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = BenchmarkRecord.model_validate_json(line)
                if record.router in allowed:
                    by_query.setdefault(record.query_id, {})[record.router] = record.cost_usd
    except (OSError, ValueError):
        return None
    complete = [costs for costs in by_query.values() if set(costs) == allowed]
    if not complete:
        return None
    return len(complete), sum(sum(costs.values()) for costs in complete)
