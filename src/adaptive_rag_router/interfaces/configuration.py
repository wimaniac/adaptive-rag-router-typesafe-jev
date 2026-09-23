"""Đọc TOML benchmark thành cấu hình typed và kiểm tra dữ liệu local cần thiết."""

from __future__ import annotations

import json
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any

from adaptive_rag_router.application import PipelinePolicy
from adaptive_rag_router.domain.enums import RouterKind
from adaptive_rag_router.domain.errors import ConfigurationError
from adaptive_rag_router.evaluation import (
    BenchmarkConfig,
    BenchmarkSample,
    DatasetBuildConfig,
    DatasetSplit,
    LocalDatasetSpec,
    SuccessCriteriaThresholds,
    prepare_benchmark_dataset,
    stratified_sample,
)

_DATASET_SUFFIXES = (".jsonl", ".ndjson", ".json", ".parquet")


@dataclass(frozen=True, slots=True)
class BenchmarkExecutionPlan:
    """Cấu hình đã resolve để CLI có thể chạy benchmark.

    Args:
        benchmark: Cấu hình runner và metrics.
        samples: Các mẫu thuộc split được chọn.
        calibration_samples: Mẫu calibration tách biệt để fit probability scaler.
        project_root: Root chứa ``pyproject.toml``.
        mock_web_path: Snapshot web bắt buộc cho formal benchmark.
        vector_top_k: Số point lấy từ Qdrant trước deduplicate.
        evidence_limit: Số passage tối đa đưa vào pipeline.
        max_repair_rounds: Số vòng repair tối đa.
        max_output_tokens: Giới hạn token cho answer generator.
        web_max_results: Số kết quả web tối đa mỗi retrieval round.
        web_provider: ``mock`` cho formal hoặc ``tavily`` cho live.
        tavily_credit_budget: Hard cap credit của live run.
        tavily_requests_per_minute: Giới hạn cửa sổ trượt.
        tavily_max_concurrency: Số Tavily call đồng thời tối đa.
        tavily_max_retries: Số lần retry tối đa sau HTTP 429.
        counterfactual_quality_floor: Ngưỡng quality để tạo gold route.
        calibration_artifact_path: Artifact formal đã khóa để live replay sử dụng.
        cost_budget_usd: Hard cap USD dùng chung cho mọi phase offline.
        provider_budget_path: Ledger project-wide dùng chung giữa mọi run.
        deepseek_limit_usd: Hard cap project-wide của DeepSeek.
        jev_limit_usd: Hard cap project-wide của TypeSafe Jev.
        jev_reserve_usd: Reserve Jev trước mỗi adaptive Jev workflow.
        counterfactual_reserve_usd: Reserve trước một query sáu nhánh.
        adaptive_reserve_usd: Reserve trước một query-router adaptive.
        pipeline_policy: Typed switches của baseline hoặc ablation run.
        replay_source_run_id: Baseline run cung cấp frozen gold/calibration.
        replay_counterfactual_path: Frozen held-out counterfactual artifact.
        replay_calibration_path: Frozen calibration model artifact.
        confirmatory_enabled: Bật paired budget-constrained confirmatory track.
        confirmatory_looks: Các mốc group-sequential đã preregister.
        confirmatory_familywise_alpha: Tổng one-sided alpha qua mọi look.
        confirmatory_excluded_run_ids: Run pilot phải loại khỏi sampling frame.
        confirmatory_calibration_path: Frozen calibration chỉ dùng giữ threshold.
        confirmatory_reference_run_id: Run dùng ước lượng adaptive pair cost.
    """

    benchmark: BenchmarkConfig
    samples: tuple[BenchmarkSample, ...]
    calibration_samples: tuple[BenchmarkSample, ...]
    project_root: Path
    mock_web_path: Path | None
    vector_top_k: int
    evidence_limit: int
    max_repair_rounds: int
    max_output_tokens: int
    web_max_results: int
    web_provider: str
    tavily_credit_budget: int = 350
    tavily_requests_per_minute: int = 90
    tavily_max_concurrency: int = 4
    tavily_max_retries: int = 4
    counterfactual_quality_floor: float = 0.70
    calibration_artifact_path: Path | None = None
    cost_budget_usd: float | None = None
    provider_budget_path: Path | None = None
    deepseek_limit_usd: float | None = None
    jev_limit_usd: float | None = None
    jev_reserve_usd: float = 0.05
    counterfactual_reserve_usd: float = 0.25
    adaptive_reserve_usd: float = 0.50
    pipeline_policy: PipelinePolicy = field(default_factory=PipelinePolicy)
    replay_source_run_id: str | None = None
    replay_counterfactual_path: Path | None = None
    replay_calibration_path: Path | None = None
    confirmatory_enabled: bool = False
    confirmatory_looks: tuple[int, ...] = ()
    confirmatory_familywise_alpha: float = 0.05
    confirmatory_excluded_run_ids: tuple[str, ...] = ()
    confirmatory_calibration_path: Path | None = None
    confirmatory_reference_run_id: str | None = None


def load_benchmark_execution_plan(
    config_path: Path,
    *,
    live: bool = False,
    tavily_credit_budget: int | None = None,
) -> BenchmarkExecutionPlan:
    """Đọc một config formal hoặc live và resolve dataset local.

    Với live config hiện tại, tập 100 câu được lấy xác định từ held-out test
    của ``configs/benchmark.toml``. Có thể trỏ sang formal config khác bằng
    ``datasets.formal_config``.

    Args:
        config_path: File TOML cần đọc.
        live: Bật cấu hình Tavily live subset.
        tavily_credit_budget: Override CLI; luôn bị giới hạn tối đa 350.

    Returns:
        Execution plan typed với đường dẫn tuyệt đối.

    Raises:
        ConfigurationError: Khi TOML, guardrail hoặc dữ liệu local không hợp lệ.
    """

    resolved_config = config_path.expanduser().resolve()
    raw = _read_toml(resolved_config)
    project_root = _find_project_root(resolved_config.parent)
    if live:
        return _load_live_plan(
            raw,
            resolved_config,
            project_root,
            tavily_credit_budget=tavily_credit_budget,
        )
    if tavily_credit_budget is not None:
        raise ConfigurationError("tavily_credit_budget chỉ hợp lệ với benchmark-live")
    return _load_formal_plan(raw, project_root)


def _load_formal_plan(
    raw: Mapping[str, Any],
    project_root: Path,
) -> BenchmarkExecutionPlan:
    run = _section(raw, "run")
    datasets = _section(raw, "datasets")
    retrieval = _section(raw, "retrieval")
    generation = _section(raw, "generation")
    counterfactual = _section(raw, "counterfactual", required=False)
    acceptance = _section(raw, "acceptance", required=False)
    cost_budget = _section(raw, "cost_budget", required=False)
    ablation = _section(raw, "ablation", required=False)
    replay = _section(raw, "replay", required=False)
    confirmatory = _section(raw, "confirmatory", required=False)

    ragrouter_path = _resolve_dataset_file(
        _resolve_path(project_root, _text(datasets, "ragrouter_path")),
        label="RAGRouter-Bench",
        exclude_names={"mock_web.jsonl"},
    )
    crag_path = _resolve_dataset_file(
        _resolve_path(project_root, _text(datasets, "crag_path")),
        label="CRAG",
        exclude_names={"mock_web.jsonl"},
    )
    mock_web_path = _resolve_path(project_root, _text(datasets, "mock_web_path"))
    if not mock_web_path.is_file():
        raise ConfigurationError(
            f"Không tìm thấy frozen web fixture: {mock_web_path}. "
            "Benchmark formal không được tự gọi Tavily."
        )

    seed = _integer(run, "seed", default=42)
    dev_size = _integer(run, "dev_size", default=200)
    calibration_size = _integer(run, "calibration_size", default=200)
    test_size = _integer(run, "test_size", default=600)
    build_config = DatasetBuildConfig(
        ragrouter_bench=LocalDatasetSpec(
            name="ragrouter-bench",
            path=ragrouter_path,
            sample_size=_integer(datasets, "ragrouter_sample_size", default=700),
            query_field=_optional_text(datasets, "ragrouter_query_field") or "query",
            id_field=_optional_text(datasets, "ragrouter_id_field") or "query_id",
            stratum_fields=_string_tuple(
                datasets.get("ragrouter_stratum_fields"),
                default=("domain", "query_type"),
            ),
            group_field=_optional_text(datasets, "ragrouter_group_field") or "group_id",
            reference_answer_field=(
                _optional_text(datasets, "ragrouter_reference_answer_field") or "answer"
            ),
            expected_route_field=None,
        ),
        crag=LocalDatasetSpec(
            name="crag",
            path=crag_path,
            sample_size=_integer(datasets, "crag_sample_size", default=300),
            query_field=_optional_text(datasets, "crag_query_field") or "query",
            id_field=_optional_text(datasets, "crag_id_field") or "interaction_id",
            stratum_fields=_string_tuple(
                datasets.get("crag_stratum_fields"),
                default=("domain", "category", "popularity", "temporal_dynamism"),
            ),
            group_field=_optional_text(datasets, "crag_group_field") or "group_id",
            reference_answer_field=(
                _optional_text(datasets, "crag_reference_answer_field") or "answer"
            ),
            expected_route_field=None,
        ),
        frozen_web_records_path=mock_web_path,
        seed=seed,
        dev_size=dev_size,
        calibration_size=calibration_size,
        test_size=test_size,
    )
    try:
        prepared = prepare_benchmark_dataset(build_config)
    except (FileNotFoundError, ValueError) as error:
        raise ConfigurationError(f"Không thể chuẩn bị benchmark dataset: {error}") from error

    split = DatasetSplit(_optional_text(run, "split") or DatasetSplit.TEST.value)
    samples_by_split = {
        DatasetSplit.DEV: prepared.splits.dev,
        DatasetSplit.CALIBRATION: prepared.splits.calibration,
        DatasetSplit.TEST: prepared.splits.test,
    }
    max_samples = _optional_integer(run, "max_samples")
    calibration_max_samples = _optional_integer(run, "calibration_max_samples")
    stratified_subset = _boolean(run, "stratified_subset", default=False)
    selected_samples = samples_by_split[split]
    calibration_samples = prepared.splits.calibration
    confirmatory_enabled = _boolean(confirmatory, "enabled", default=False)
    confirmatory_excluded_run_ids = _string_tuple(
        confirmatory.get("excluded_run_ids"),
        default=(),
    )
    output_dir = _resolve_path(
        project_root,
        _optional_text(run, "output_dir") or "artifacts/benchmarks",
    )
    confirmatory_calibration_path: Path | None = None
    confirmatory_reference_run_id: str | None = None
    confirmatory_looks: tuple[int, ...] = ()
    confirmatory_familywise_alpha = _number(
        confirmatory,
        "familywise_alpha",
        default=0.05,
        minimum=0.001,
        maximum=0.20,
    )
    if confirmatory_enabled:
        calibration_source_run_id = _text(confirmatory, "calibration_source_run_id")
        confirmatory_reference_run_id = calibration_source_run_id
        _validate_run_id(calibration_source_run_id)
        confirmatory_calibration_path = (
            output_dir / calibration_source_run_id / "calibration-models.json"
        )
        if not confirmatory_calibration_path.is_file():
            raise ConfigurationError(
                "Confirmatory track thiếu frozen calibration artifact: "
                f"{confirmatory_calibration_path}"
            )
        if not confirmatory_excluded_run_ids:
            raise ConfigurationError("confirmatory.excluded_run_ids không được rỗng")
        excluded_query_ids = _load_excluded_query_ids(
            output_dir,
            confirmatory_excluded_run_ids,
        )
        selected_samples = tuple(
            sample for sample in selected_samples if sample.query_id not in excluded_query_ids
        )
        confirmatory_looks = _integer_tuple(confirmatory.get("looks"), key="looks")
    if stratified_subset:
        if max_samples is None:
            raise ConfigurationError("stratified_subset=true cần run.max_samples")
        if max_samples > len(selected_samples):
            raise ConfigurationError("run.max_samples lớn hơn split được chọn")
        selected_samples = stratified_sample(selected_samples, max_samples, seed=seed + 101)
        if calibration_max_samples is not None:
            if calibration_max_samples > len(calibration_samples):
                raise ConfigurationError("run.calibration_max_samples lớn hơn calibration split")
            calibration_samples = stratified_sample(
                calibration_samples,
                calibration_max_samples,
                seed=seed + 102,
            )
    elif calibration_max_samples is not None:
        raise ConfigurationError(
            "run.calibration_max_samples chỉ hợp lệ khi stratified_subset=true"
        )
    cost_budget_usd = _optional_number(cost_budget, "limit_usd", minimum=0.01)
    provider_budget_path, deepseek_limit_usd, jev_limit_usd = _provider_budget_config(
        cost_budget,
        project_root,
    )
    max_concurrency = _integer(run, "max_concurrency", default=4)
    if cost_budget_usd is not None and max_concurrency != 1:
        raise ConfigurationError("Benchmark có cost budget bắt buộc max_concurrency=1")

    benchmark = BenchmarkConfig(
        run_id=_optional_text(run, "name") or "formal-1000",
        routers=_router_tuple(run.get("routers")),
        split=split,
        bootstrap_resamples=_integer(run, "bootstrap_iterations", default=10_000),
        seed=seed,
        max_concurrency=max_concurrency,
        max_samples=max_samples,
        checkpoint_every=_integer(run, "checkpoint_every", default=25),
        alternate_router_order=confirmatory_enabled,
        publish_acceptance=_boolean(run, "publish_acceptance", default=True),
        success_thresholds=SuccessCriteriaThresholds(
            quality_non_inferiority_margin=_number(
                acceptance,
                "quality_non_inferiority_margin",
                default=0.02,
                minimum=0.0,
                maximum=1.0,
            ),
            minimum_cost_reduction=_number(
                acceptance,
                "minimum_cost_reduction",
                default=0.20,
                minimum=0.0,
                maximum=1.0,
            ),
            minimum_p50_latency_reduction=_number(
                acceptance,
                "minimum_p50_latency_reduction",
                default=0.15,
                minimum=0.0,
                maximum=1.0,
            ),
            maximum_p95_latency_regression=_number(
                acceptance,
                "maximum_p95_latency_regression",
                default=0.05,
                minimum=0.0,
                maximum=1.0,
            ),
            critical_recall=_number(
                acceptance,
                "critical_recall",
                default=0.90,
                minimum=0.0,
                maximum=1.0,
            ),
            maximum_ece=_number(
                acceptance,
                "maximum_ece",
                default=0.08,
                minimum=0.0,
                maximum=1.0,
            ),
            maximum_unhandled_error_rate=_number(
                acceptance,
                "maximum_unhandled_error_rate",
                default=0.005,
                minimum=0.0,
                maximum=1.0,
            ),
        ),
        output_dir=_resolve_path(
            project_root,
            _optional_text(run, "output_dir") or "artifacts/benchmarks",
        ),
    )
    provider = _optional_text(retrieval, "web_provider") or "mock"
    if provider != "mock":
        raise ConfigurationError("Benchmark formal bắt buộc dùng retrieval.web_provider='mock'")
    replay_source_run_id = _optional_text(replay, "source_run_id")
    replay_counterfactual_path: Path | None = None
    replay_calibration_path: Path | None = None
    if replay_source_run_id is not None:
        if split is not DatasetSplit.TEST:
            raise ConfigurationError("replay.source_run_id chỉ hợp lệ với test split")
        _validate_run_id(replay_source_run_id)
        source_directory = benchmark.output_dir / replay_source_run_id
        replay_counterfactual_path = source_directory / "counterfactual.jsonl"
        replay_calibration_path = source_directory / "calibration-models.json"
        missing = [
            path
            for path in (replay_counterfactual_path, replay_calibration_path)
            if not path.is_file()
        ]
        if missing:
            raise ConfigurationError(f"Replay source thiếu frozen artifacts: {missing}")
    if confirmatory_enabled:
        if split is not DatasetSplit.TEST:
            raise ConfigurationError("Confirmatory track bắt buộc dùng test split")
        if replay_source_run_id is not None:
            raise ConfigurationError("Confirmatory track không dùng [replay] counterfactual")
        if set(benchmark.routers) != {RouterKind.JEV, RouterKind.LLM}:
            raise ConfigurationError("Confirmatory track chỉ chạy routers = ['jev', 'llm']")
        if benchmark.max_concurrency != 1:
            raise ConfigurationError("Confirmatory track bắt buộc max_concurrency=1")
        if benchmark.publish_acceptance:
            raise ConfigurationError("Confirmatory track cần publish_acceptance=false")
        if benchmark.max_samples is None or not confirmatory_looks:
            raise ConfigurationError("Confirmatory track cần max_samples và các look")
        if confirmatory_looks[-1] != benchmark.max_samples:
            raise ConfigurationError("Look cuối phải bằng run.max_samples")
    return BenchmarkExecutionPlan(
        benchmark=benchmark,
        samples=selected_samples,
        calibration_samples=calibration_samples,
        project_root=project_root,
        mock_web_path=mock_web_path,
        vector_top_k=_integer(retrieval, "vector_top_k", default=12),
        evidence_limit=_integer(retrieval, "evidence_limit", default=8),
        max_repair_rounds=_integer(retrieval, "max_repair_rounds", default=2),
        max_output_tokens=_integer(generation, "max_output_tokens", default=1024),
        web_max_results=_integer(retrieval, "max_results", default=5),
        web_provider="mock",
        counterfactual_quality_floor=_number(
            counterfactual,
            "quality_floor",
            default=0.70,
            minimum=0.0,
            maximum=1.0,
        ),
        cost_budget_usd=cost_budget_usd,
        provider_budget_path=provider_budget_path,
        deepseek_limit_usd=deepseek_limit_usd,
        jev_limit_usd=jev_limit_usd,
        jev_reserve_usd=_number(
            cost_budget,
            "jev_reserve_usd",
            default=0.05,
            minimum=0.001,
            maximum=100.0,
        ),
        counterfactual_reserve_usd=_number(
            cost_budget,
            "counterfactual_reserve_usd",
            default=0.25,
            minimum=0.001,
            maximum=100.0,
        ),
        adaptive_reserve_usd=_number(
            cost_budget,
            "adaptive_reserve_usd",
            default=0.50,
            minimum=0.001,
            maximum=100.0,
        ),
        pipeline_policy=_pipeline_policy(ablation),
        replay_source_run_id=replay_source_run_id,
        replay_counterfactual_path=replay_counterfactual_path,
        replay_calibration_path=replay_calibration_path,
        confirmatory_enabled=confirmatory_enabled,
        confirmatory_looks=confirmatory_looks,
        confirmatory_familywise_alpha=confirmatory_familywise_alpha,
        confirmatory_excluded_run_ids=confirmatory_excluded_run_ids,
        confirmatory_calibration_path=confirmatory_calibration_path,
        confirmatory_reference_run_id=confirmatory_reference_run_id,
    )


def _load_live_plan(
    raw: Mapping[str, Any],
    config_path: Path,
    project_root: Path,
    *,
    tavily_credit_budget: int | None,
) -> BenchmarkExecutionPlan:
    run = _section(raw, "run")
    retrieval = _section(raw, "retrieval")
    tavily = _section(raw, "tavily")
    cost_budget = _section(raw, "cost_budget", required=False)
    ablation = _section(raw, "ablation", required=False)
    datasets = _section(raw, "datasets", required=False)
    provider = _optional_text(retrieval, "web_provider") or "tavily"
    if provider != "tavily":
        raise ConfigurationError("benchmark-live bắt buộc dùng retrieval.web_provider='tavily'")
    search_depth = _optional_text(retrieval, "search_depth") or "basic"
    if search_depth != "basic":
        raise ConfigurationError("MVP chỉ cho phép Tavily Basic Search")
    max_results = _integer(retrieval, "max_results", default=5)
    if not 1 <= max_results <= 5:
        raise ConfigurationError("retrieval.max_results phải nằm trong khoảng 1..5")

    configured_budget = _integer(tavily, "credit_budget", default=350)
    selected_budget = tavily_credit_budget or configured_budget
    if not 1 <= selected_budget <= 350:
        raise ConfigurationError("Tavily credit budget phải nằm trong khoảng 1..350")
    rpm = _integer(tavily, "requests_per_minute", default=90)
    concurrency = _integer(tavily, "max_concurrency", default=4)
    retries = _integer(tavily, "max_retries", default=4)
    if not 1 <= rpm <= 90:
        raise ConfigurationError("Tavily requests_per_minute phải nằm trong khoảng 1..90")
    if not 1 <= concurrency <= 4:
        raise ConfigurationError("Tavily max_concurrency phải nằm trong khoảng 1..4")
    if not 0 <= retries <= 4:
        raise ConfigurationError("Tavily max_retries phải nằm trong khoảng 0..4")
    cost_budget_usd = _optional_number(cost_budget, "limit_usd", minimum=0.01)
    provider_budget_path, deepseek_limit_usd, jev_limit_usd = _provider_budget_config(
        cost_budget,
        project_root,
    )
    if cost_budget_usd is not None and concurrency != 1:
        raise ConfigurationError("Live benchmark có cost budget bắt buộc max_concurrency=1")

    formal_reference = _optional_text(datasets, "formal_config") or "configs/benchmark.toml"
    formal_path = _resolve_path(project_root, formal_reference)
    if formal_path == config_path:
        raise ConfigurationError("datasets.formal_config không được trỏ lại live config")
    formal_plan = load_benchmark_execution_plan(formal_path)
    sample_size = _integer(run, "sample_size", default=100)
    if sample_size > len(formal_plan.samples):
        raise ConfigurationError(
            f"Live subset cần {sample_size} mẫu nhưng held-out split chỉ có "
            f"{len(formal_plan.samples)} mẫu"
        )
    seed = _integer(run, "seed", default=formal_plan.benchmark.seed)
    live_samples = stratified_sample(formal_plan.samples, sample_size, seed=seed)
    benchmark = BenchmarkConfig(
        run_id=_optional_text(run, "name") or "tavily-live-100",
        routers=_router_tuple(run.get("routers")),
        split=DatasetSplit.TEST,
        bootstrap_resamples=_integer(run, "bootstrap_iterations", default=10_000),
        seed=seed,
        max_concurrency=concurrency,
        max_samples=sample_size,
        live=True,
        checkpoint_every=_integer(run, "checkpoint_every", default=10),
        success_thresholds=formal_plan.benchmark.success_thresholds,
        output_dir=_resolve_path(
            project_root,
            _optional_text(run, "output_dir") or "artifacts/benchmarks",
        ),
    )
    return BenchmarkExecutionPlan(
        benchmark=benchmark,
        samples=live_samples,
        calibration_samples=(),
        project_root=project_root,
        mock_web_path=formal_plan.mock_web_path,
        vector_top_k=formal_plan.vector_top_k,
        evidence_limit=formal_plan.evidence_limit,
        max_repair_rounds=_integer(retrieval, "max_repair_rounds", default=2),
        max_output_tokens=formal_plan.max_output_tokens,
        web_max_results=max_results,
        web_provider="tavily",
        tavily_credit_budget=selected_budget,
        tavily_requests_per_minute=rpm,
        tavily_max_concurrency=concurrency,
        tavily_max_retries=retries,
        counterfactual_quality_floor=formal_plan.counterfactual_quality_floor,
        calibration_artifact_path=(
            formal_plan.benchmark.output_dir
            / formal_plan.benchmark.run_id
            / "calibration-models.json"
        ),
        cost_budget_usd=cost_budget_usd,
        provider_budget_path=provider_budget_path,
        deepseek_limit_usd=deepseek_limit_usd,
        jev_limit_usd=jev_limit_usd,
        jev_reserve_usd=_number(
            cost_budget,
            "jev_reserve_usd",
            default=0.05,
            minimum=0.001,
            maximum=100.0,
        ),
        counterfactual_reserve_usd=_number(
            cost_budget,
            "counterfactual_reserve_usd",
            default=0.25,
            minimum=0.001,
            maximum=100.0,
        ),
        adaptive_reserve_usd=_number(
            cost_budget,
            "adaptive_reserve_usd",
            default=0.50,
            minimum=0.001,
            maximum=100.0,
        ),
        pipeline_policy=_pipeline_policy(ablation),
    )


def _provider_budget_config(
    section: Mapping[str, Any],
    project_root: Path,
) -> tuple[Path | None, float | None, float | None]:
    """Đọc bộ ba cấu hình provider budget bắt buộc đi cùng nhau.

    Args:
        section: Bảng TOML ``cost_budget``.
        project_root: Root dùng resolve ledger path.

    Returns:
        Provider ledger path, DeepSeek limit và Jev limit hoặc ba giá trị ``None``.

    Raises:
        ConfigurationError: Khi chỉ cấu hình một phần hard cap project-wide.
    """

    raw_path = _optional_text(section, "provider_ledger_path")
    deepseek_limit = _optional_number(section, "deepseek_limit_usd", minimum=0.01)
    jev_limit = _optional_number(section, "jev_limit_usd", minimum=0.01)
    values_present = (raw_path is not None, deepseek_limit is not None, jev_limit is not None)
    if any(values_present) and not all(values_present):
        raise ConfigurationError(
            "provider_ledger_path, deepseek_limit_usd và jev_limit_usd phải đi cùng nhau"
        )
    if raw_path is None:
        return None, None, None
    return _resolve_path(project_root, raw_path), deepseek_limit, jev_limit


def _pipeline_policy(section: Mapping[str, Any]) -> PipelinePolicy:
    """Parse typed component switches cho baseline/ablation run."""

    return PipelinePolicy(
        context_gate_enabled=_boolean(section, "context_gate_enabled", default=True),
        repair_enabled=_boolean(section, "repair_enabled", default=True),
        strong_fallback_enabled=_boolean(section, "strong_fallback_enabled", default=True),
    )


def _validate_run_id(run_id: str) -> None:
    """Từ chối path traversal trong replay run ID."""

    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
    if not run_id or any(character not in allowed for character in run_id):
        raise ConfigurationError(f"run-id không hợp lệ: {run_id!r}")


def _load_excluded_query_ids(
    output_dir: Path,
    run_ids: tuple[str, ...],
) -> set[str]:
    """Đọc query IDs đã dùng để chống pilot leakage trong confirmatory track.

    Args:
        output_dir: Root benchmark artifacts.
        run_ids: Các run development/pilot cần loại.

    Returns:
        Tập query ID hợp nhất từ ``records.jsonl`` của các run.

    Raises:
        ConfigurationError: Khi run ID/path/schema không hợp lệ.
    """

    excluded: set[str] = set()
    for run_id in run_ids:
        _validate_run_id(run_id)
        records_path = output_dir / run_id / "records.jsonl"
        if not records_path.is_file():
            raise ConfigurationError(
                f"Confirmatory exclusion run thiếu records.jsonl: {records_path}"
            )
        try:
            with records_path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    payload = json.loads(line)
                    query_id = payload.get("query_id") if isinstance(payload, dict) else None
                    if not isinstance(query_id, str) or not query_id.strip():
                        raise ValueError(f"query_id không hợp lệ tại dòng {line_number}")
                    excluded.add(query_id.strip())
        except (OSError, ValueError, json.JSONDecodeError) as error:
            raise ConfigurationError(
                f"Không đọc được confirmatory exclusion artifact: {records_path}"
            ) from error
    return excluded


def _read_toml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigurationError(f"Không tìm thấy file config: {path}")
    try:
        with path.open("rb") as file_handle:
            payload = tomllib.load(file_handle)
    except tomllib.TOMLDecodeError as error:
        raise ConfigurationError(f"TOML không hợp lệ tại {path}: {error}") from error
    return payload


def _find_project_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return Path.cwd().resolve()


def _resolve_path(project_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _resolve_dataset_file(
    path: Path,
    *,
    label: str,
    exclude_names: set[str],
) -> Path:
    if path.is_file():
        if path.suffix.lower() not in _DATASET_SUFFIXES:
            raise ConfigurationError(f"{label} không dùng định dạng dataset được hỗ trợ: {path}")
        return path
    if not path.exists():
        raise ConfigurationError(f"Không tìm thấy dataset {label}: {path}")
    if not path.is_dir():
        raise ConfigurationError(f"Đường dẫn dataset {label} không phải file/thư mục: {path}")
    candidates = sorted(
        candidate
        for candidate in path.iterdir()
        if candidate.is_file()
        and candidate.suffix.lower() in _DATASET_SUFFIXES
        and candidate.name not in exclude_names
        and not candidate.name.endswith(".index.json")
    )
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise ConfigurationError(
            f"Thư mục dataset {label} không có file JSON/JSONL/Parquet: {path}"
        )
    raise ConfigurationError(
        f"Thư mục dataset {label} có nhiều file; hãy đặt đường dẫn file cụ thể trong config: {path}"
    )


def _section(
    raw: Mapping[str, Any],
    name: str,
    *,
    required: bool = True,
) -> Mapping[str, Any]:
    value = raw.get(name)
    if value is None and not required:
        return {}
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"Thiếu hoặc sai section [{name}] trong config")
    return value


def _text(section: Mapping[str, Any], key: str) -> str:
    value = _optional_text(section, key)
    if value is None:
        raise ConfigurationError(f"Thiếu giá trị {key!r} trong config")
    return value


def _optional_text(section: Mapping[str, Any], key: str) -> str | None:
    value = section.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"Giá trị {key!r} phải là chuỗi không rỗng")
    return value.strip()


def _integer(section: Mapping[str, Any], key: str, *, default: int) -> int:
    value = section.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError(f"Giá trị {key!r} phải là số nguyên")
    return value


def _optional_integer(section: Mapping[str, Any], key: str) -> int | None:
    value = section.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError(f"Giá trị {key!r} phải là số nguyên")
    return value


def _integer_tuple(value: Any, *, key: str) -> tuple[int, ...]:
    """Parse danh sách số nguyên dương tăng nghiêm ngặt."""

    if not isinstance(value, list) or not value:
        raise ConfigurationError(f"Giá trị {key!r} phải là danh sách số nguyên không rỗng")
    if any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in value):
        raise ConfigurationError(f"Giá trị {key!r} chỉ gồm số nguyên dương")
    result = tuple(value)
    if any(left >= right for left, right in pairwise(result)):
        raise ConfigurationError(f"Giá trị {key!r} phải tăng nghiêm ngặt")
    return result


def _boolean(section: Mapping[str, Any], key: str, *, default: bool) -> bool:
    value = section.get(key, default)
    if not isinstance(value, bool):
        raise ConfigurationError(f"Giá trị {key!r} phải là boolean")
    return value


def _optional_number(
    section: Mapping[str, Any],
    key: str,
    *,
    minimum: float,
) -> float | None:
    value = section.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigurationError(f"Giá trị {key!r} phải là số")
    numeric = float(value)
    if numeric < minimum:
        raise ConfigurationError(f"Giá trị {key!r} phải >= {minimum}")
    return numeric


def _number(
    section: Mapping[str, Any],
    key: str,
    *,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    value = section.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigurationError(f"Giá trị {key!r} phải là số")
    numeric = float(value)
    if not minimum <= numeric <= maximum:
        raise ConfigurationError(f"Giá trị {key!r} phải nằm trong [{minimum}, {maximum}]")
    return numeric


def _string_tuple(value: Any, *, default: tuple[str, ...]) -> tuple[str, ...]:
    if value is None:
        return default
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise ConfigurationError("Danh sách stratum fields phải gồm các chuỗi không rỗng")
    return tuple(item.strip() for item in value)


def _router_tuple(value: Any) -> tuple[RouterKind, ...]:
    if value is None:
        return (RouterKind.JEV, RouterKind.RULE, RouterKind.LLM)
    if not isinstance(value, list) or not value:
        raise ConfigurationError("run.routers phải là danh sách không rỗng")
    try:
        return tuple(RouterKind(item) for item in value)
    except (TypeError, ValueError) as error:
        raise ConfigurationError("run.routers chỉ nhận jev, rule hoặc llm") from error
