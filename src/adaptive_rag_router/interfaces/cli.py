"""Định nghĩa CLI Typer cho query, benchmark, Tavily live, report và ingestion."""

from __future__ import annotations

import asyncio
import json
import re
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Annotated, Never

import typer
from pydantic import ValidationError

from adaptive_rag_router.config import Settings, get_settings
from adaptive_rag_router.domain.enums import RouterKind
from adaptive_rag_router.domain.errors import AdaptiveRAGError, ConfigurationError
from adaptive_rag_router.domain.models import AnswerResponse, QueryRequest
from adaptive_rag_router.evaluation import (
    ArtifactPaths,
    BenchmarkArtifactWriter,
    BenchmarkRecord,
    BenchmarkRunner,
    ConfirmatoryDecision,
    ConfirmatoryReport,
    DatasetSplit,
    SafetyAssessmentResult,
    SafetyVariant,
    TemperatureCalibrationModel,
    analyze_confirmatory_look,
    build_human_audit_bundle,
    build_safety_markdown_report,
    build_safety_variants,
    compare_repeatability,
    fit_router_calibrators,
    load_calibration_models,
    reconcile_provider_budget,
    run_safety_exploratory,
    write_calibration_models,
    write_confirmatory_manifest,
    write_confirmatory_report,
    write_human_audit_bundle,
    write_repeatability_report,
    write_safety_report,
    write_safety_results,
    write_safety_variants,
)
from adaptive_rag_router.evaluation.cost_budget import (
    BudgetHook,
    CombinedBudgetHook,
    CombinedUsdLedger,
    FileProviderBudgetLedger,
    FileUsdBudgetLedger,
    ProviderScopedUsdLedger,
    ProviderUsdBudgetHook,
    UsdBudgetHook,
    UsdLedger,
)
from adaptive_rag_router.interfaces.configuration import (
    BenchmarkExecutionPlan,
    load_benchmark_execution_plan,
)
from adaptive_rag_router.interfaces.counterfactual import (
    StaticBranchEvaluator,
    attach_counterfactual_gold,
)
from adaptive_rag_router.interfaces.data_preparation import prepare_benchmark_sources
from adaptive_rag_router.interfaces.ingestion import ingest_source_path
from adaptive_rag_router.interfaces.preflight import build_benchmark_preflight
from adaptive_rag_router.interfaces.runtime import (
    CalibratedBenchmarkAdapter,
    FileBenchmarkRecordStore,
    FileCheckpointHook,
    PipelineBenchmarkAdapter,
    RuntimeBundle,
    TavilyBudgetHook,
    build_runtime,
)
from adaptive_rag_router.retrieval import BgeM3EmbeddingAdapter, QdrantVectorRetriever


def _configure_windows_utf8() -> None:
    if sys.platform != "win32":
        return
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")


_configure_windows_utf8()

app = typer.Typer(
    name="rag-router",
    help="Adaptive RAG Router dùng Jev, rule-based hoặc DeepSeek LLM decision engine.",
    no_args_is_help=True,
)


@app.command("query")
def query_command(
    query_text: Annotated[
        str,
        typer.Option("--query", help="Truy vấn cần xử lý."),
    ],
    engine: Annotated[
        RouterKind,
        typer.Option("--engine", help="Decision engine: jev, rule hoặc llm."),
    ] = RouterKind.JEV,
    live_web: Annotated[
        bool,
        typer.Option(
            "--live-web/--mock-web",
            help="Cho phép Tavily thật; mặc định dùng frozen/mock web.",
        ),
    ] = False,
    include_trace: Annotated[
        bool,
        typer.Option("--trace/--no-trace", help="Đính kèm typed decision trace."),
    ] = False,
) -> None:
    """Xử lý một query bằng engine được chọn và in JSON response."""

    try:
        response = asyncio.run(
            execute_query(
                query_text,
                engine=engine,
                live_web=live_web,
                include_trace=include_trace,
            )
        )
    except Exception as error:
        _exit_with_error(error)
    typer.echo(response.model_dump_json(indent=2))


@app.command("benchmark")
def benchmark_command(
    config: Annotated[
        Path,
        typer.Option("--config", help="TOML của formal offline benchmark."),
    ] = Path("configs/benchmark.toml"),
) -> None:
    """Chạy benchmark formal với frozen web, không gọi Tavily."""

    try:
        run_id, paths, _ = asyncio.run(execute_benchmark(config, live=False))
    except Exception as error:
        _exit_with_error(error)
    _echo_artifacts(run_id, paths)


@app.command("benchmark-live")
def benchmark_live_command(
    config: Annotated[
        Path,
        typer.Option("--config", help="TOML của Tavily live subset."),
    ] = Path("configs/live.toml"),
    tavily_credit_budget: Annotated[
        int | None,
        typer.Option(
            "--tavily-credit-budget",
            min=1,
            max=350,
            help="Hard cap credit; không bao giờ vượt 350 trong MVP.",
        ),
    ] = None,
) -> None:
    """Chạy live subset với Basic Search và credit checkpoint."""

    try:
        run_id, paths, credits = asyncio.run(
            execute_benchmark(
                config,
                live=True,
                tavily_credit_budget=tavily_credit_budget,
            )
        )
    except Exception as error:
        _exit_with_error(error)
    _echo_artifacts(run_id, paths, tavily_credits_consumed=credits)


@app.command("benchmark-confirmatory")
def benchmark_confirmatory_command(
    config: Annotated[
        Path,
        typer.Option("--config", help="TOML paired group-sequential confirmatory track."),
    ] = Path("configs/confirmatory.toml"),
) -> None:
    """Chạy paired Jev-vs-LLM theo các look và dừng sớm khi đủ bằng chứng."""

    try:
        report, paths = asyncio.run(execute_confirmatory_benchmark(config))
    except Exception as error:
        _exit_with_error(error)
    typer.echo(
        json.dumps(
            {
                "run_id": report.run_id,
                "look_index": report.look_index,
                "pair_count": report.pair_count,
                "decision": report.decision,
                "confirmatory_json": str(paths[0]),
                "confirmatory_markdown": str(paths[1]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


@app.command("prepare-confirmatory")
def prepare_confirmatory_command(
    config: Annotated[
        Path,
        typer.Option("--config", help="TOML confirmatory cần khóa manifest."),
    ] = Path("configs/confirmatory.toml"),
) -> None:
    """Khóa manifest fresh held-out queries mà không gọi provider."""

    try:
        plan = load_benchmark_execution_plan(config)
        manifest_path, manifest_hash = _prepare_confirmatory_manifest(plan)
    except Exception as error:
        _exit_with_error(error)
    typer.echo(
        json.dumps(
            {
                "run_id": plan.benchmark.run_id,
                "sample_count": len(plan.samples),
                "manifest_sha256": manifest_hash,
                "manifest_path": str(manifest_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


@app.command("report-confirmatory")
def report_confirmatory_command(
    config: Annotated[
        Path,
        typer.Option("--config", help="TOML confirmatory cần tái tạo look reports."),
    ] = Path("configs/confirmatory.toml"),
) -> None:
    """Tái tạo mọi look report đã hoàn tất mà không gọi provider."""

    try:
        report, paths = refresh_confirmatory_reports(config)
    except Exception as error:
        _exit_with_error(error)
    typer.echo(
        json.dumps(
            {
                "run_id": report.run_id,
                "look_index": report.look_index,
                "pair_count": report.pair_count,
                "decision": report.decision,
                "confirmatory_json": str(paths[0]),
                "confirmatory_markdown": str(paths[1]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


@app.command("benchmark-preflight")
def benchmark_preflight_command(
    config: Annotated[
        Path,
        typer.Option("--config", help="TOML formal hoặc Tavily live cần kiểm tra."),
    ] = Path("configs/benchmark.toml"),
    live: Annotated[
        bool,
        typer.Option("--live/--formal", help="Kiểm tra prerequisite của live track."),
    ] = False,
    tavily_credit_budget: Annotated[
        int | None,
        typer.Option("--tavily-credit-budget", min=1, max=350),
    ] = None,
) -> None:
    """Kiểm tra readiness, checkpoint và cost estimate mà không gọi provider."""

    try:
        plan = load_benchmark_execution_plan(
            config,
            live=live,
            tavily_credit_budget=tavily_credit_budget if live else None,
        )
        active_settings = _settings_for_plan(get_settings(), plan)
        report = build_benchmark_preflight(plan, active_settings, live=live)
    except Exception as error:
        _exit_with_error(error)
    typer.echo(json.dumps(asdict(report), ensure_ascii=False, indent=2))
    if not report.ready:
        raise typer.Exit(code=2)


@app.command("report")
def report_command(
    run_id: Annotated[
        str,
        typer.Option("--run-id", help="Run ID đã ghi trong artifacts/benchmarks."),
    ],
) -> None:
    """In báo cáo Markdown đã tạo của một benchmark run."""

    try:
        report = read_report(run_id, get_settings())
    except Exception as error:
        _exit_with_error(error)
    typer.echo(report)


@app.command("compare-runs")
def compare_runs_command(
    baseline_run: Annotated[
        str,
        typer.Option("--baseline-run", help="Run gốc dùng làm baseline repeatability."),
    ],
    repeat_run: Annotated[
        str,
        typer.Option("--repeat-run", help="Run lặp lại trên cùng sample set."),
    ],
) -> None:
    """So sánh hai run theo query-router và ghi repeatability artifacts."""

    try:
        settings = _absolutize_settings(get_settings(), Path.cwd())
        read_report(baseline_run, settings)
        read_report(repeat_run, settings)
        benchmark_root = settings.artifacts_dir / "benchmarks"
        baseline = _load_benchmark_records(benchmark_root / baseline_run / "records.jsonl")
        repeat = _load_benchmark_records(benchmark_root / repeat_run / "records.jsonl")
        report = compare_repeatability(baseline, repeat)
        json_path, markdown_path = write_repeatability_report(
            benchmark_root / repeat_run / f"repeatability-vs-{baseline_run}",
            report,
        )
    except Exception as error:
        _exit_with_error(error)
    typer.echo(
        json.dumps(
            {
                "baseline_run": baseline_run,
                "repeat_run": repeat_run,
                "repeatability_json": str(json_path),
                "repeatability_markdown": str(markdown_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


@app.command("reconcile-cost-budget")
def reconcile_cost_budget_command(
    run_ids: Annotated[
        list[str],
        typer.Option("--run-id", help="Run cần migrate; lặp option cho nhiều run."),
    ],
    ledger_path: Annotated[
        Path,
        typer.Option("--ledger", help="Project-wide provider budget checkpoint."),
    ] = Path("artifacts/benchmarks/provider-cost-budget.json"),
    deepseek_limit_usd: Annotated[
        float,
        typer.Option("--deepseek-limit", min=0.01, help="Hard cap DeepSeek toàn project."),
    ] = 2.79,
    jev_limit_usd: Annotated[
        float,
        typer.Option("--jev-limit", min=0.01, help="Hard cap TypeSafe Jev toàn project."),
    ] = 3.0,
    safety_results: Annotated[
        Path | None,
        typer.Option("--safety-results", help="Safety JSONL cần tính vào project budget."),
    ] = None,
) -> None:
    """Reconcile cost artifacts cũ vào hard cap riêng DeepSeek/Jev, không gọi API."""

    try:
        settings = _absolutize_settings(get_settings(), Path.cwd())
        ledger = FileProviderBudgetLedger(
            ledger_path.expanduser().resolve(),
            limits_usd={
                "deepseek": deepseek_limit_usd,
                "typesafe-jev": jev_limit_usd,
            },
        )
        report = asyncio.run(
            reconcile_provider_budget(
                settings.artifacts_dir / "benchmarks",
                tuple(run_ids),
                ledger,
                safety_results_path=(
                    safety_results.expanduser().resolve() if safety_results is not None else None
                ),
            )
        )
    except Exception as error:
        _exit_with_error(error)
    typer.echo(report.model_dump_json(indent=2))


@app.command("ingest")
def ingest_command(
    source: Annotated[
        Path,
        typer.Option("--source", help="File hoặc directory JSON/JSONL/Parquet."),
    ],
    recreate: Annotated[
        bool,
        typer.Option(
            "--recreate",
            help="Xóa và tạo lại collection trước ingest; mặc định giữ dữ liệu cũ.",
        ),
    ] = False,
    batch_size: Annotated[
        int,
        typer.Option("--batch-size", min=1, help="Số passage mỗi batch embedding."),
    ] = 64,
) -> None:
    """Chunk tài liệu 512/64 và ingest bằng BGE-M3 vào Qdrant local."""

    try:
        passages = asyncio.run(execute_ingest(source, recreate=recreate, batch_size=batch_size))
    except Exception as error:
        _exit_with_error(error)
    typer.echo(json.dumps({"passages_ingested": passages}, ensure_ascii=False))


@app.command("prepare-data")
def prepare_data_command(
    ragrouter_source: Annotated[
        Path,
        typer.Option("--ragrouter-source", help="File/directory RAGRouter-Bench chính thức."),
    ],
    crag_source: Annotated[
        Path,
        typer.Option("--crag-source", help="CRAG JSON/JSONL/Parquet hoặc .jsonl.bz2."),
    ],
    output_root: Annotated[
        Path,
        typer.Option("--output-root", help="Thư mục data đích."),
    ] = Path("data"),
) -> None:
    """Chuẩn hóa dataset và frozen web fixture mà không gọi provider."""

    try:
        counts = prepare_benchmark_sources(ragrouter_source, crag_source, output_root)
    except Exception as error:
        _exit_with_error(error)
    typer.echo(json.dumps(asdict(counts), ensure_ascii=False, indent=2))


@app.command("prepare-safety")
def prepare_safety_command(
    config: Annotated[
        Path,
        typer.Option("--config", help="TOML formal dùng để lấy held-out split."),
    ] = Path("configs/benchmark.toml"),
    output: Annotated[
        Path,
        typer.Option("--output", help="File JSONL chứa safety variants."),
    ] = Path("artifacts/safety/variants.jsonl"),
    sample_size: Annotated[
        int,
        typer.Option("--sample-size", min=4, help="Số context variants cần tạo."),
    ] = 100,
) -> None:
    """Tạo offline safety variants cân bằng, không gọi provider."""

    try:
        plan = load_benchmark_execution_plan(config)
        variants = build_safety_variants(
            plan.samples,
            sample_size=sample_size,
            seed=plan.benchmark.seed,
        )
        output_path = write_safety_variants(output, variants)
    except Exception as error:
        _exit_with_error(error)
    typer.echo(
        json.dumps(
            {"variant_count": len(variants), "variants_jsonl": str(output_path)},
            ensure_ascii=False,
            indent=2,
        )
    )


@app.command("prepare-audit")
def prepare_audit_command(
    config: Annotated[
        Path,
        typer.Option("--config", help="TOML formal dùng để lấy held-out split."),
    ] = Path("configs/benchmark.toml"),
    run_id: Annotated[
        str | None,
        typer.Option("--run-id", help="Run formal; mặc định lấy run.name từ TOML."),
    ] = None,
    sample_size: Annotated[
        int,
        typer.Option("--sample-size", min=1, help="Số held-out query cần audit."),
    ] = 100,
) -> None:
    """Tạo packet human audit mù cho hai annotator từ formal records."""

    try:
        plan = load_benchmark_execution_plan(config)
        selected_run = run_id or plan.benchmark.run_id
        run_directory = plan.benchmark.output_dir / selected_run
        records = _load_benchmark_records(run_directory / "records.jsonl")
        routers_by_query: dict[str, set[RouterKind]] = {}
        for record in records:
            routers_by_query.setdefault(record.query_id, set()).add(record.router)
        complete_query_ids = {
            query_id for query_id, routers in routers_by_query.items() if routers == set(RouterKind)
        }
        auditable_samples = tuple(
            sample for sample in plan.samples if sample.query_id in complete_query_ids
        )
        if len(auditable_samples) < sample_size:
            raise ConfigurationError(
                f"Run {selected_run!r} chỉ có {len(auditable_samples)} query đủ ba router; "
                f"không thể tạo audit {sample_size} query."
            )
        bundle = build_human_audit_bundle(
            auditable_samples,
            records,
            sample_size=sample_size,
            seed=plan.benchmark.seed,
        )
        paths = write_human_audit_bundle(run_directory / "human-audit", bundle)
    except Exception as error:
        _exit_with_error(error)
    typer.echo(
        json.dumps(
            {
                "query_count": len(bundle.items),
                "packet": str(paths.packet),
                "blind_key": str(paths.blind_key),
                "annotator_one": str(paths.annotator_one),
                "annotator_two": str(paths.annotator_two),
                "adjudication": str(paths.adjudication),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


@app.command("benchmark-safety")
def benchmark_safety_command(
    config: Annotated[
        Path,
        typer.Option("--config", help="TOML formal cung cấp router/runtime settings."),
    ] = Path("configs/benchmark.toml"),
    variants_path: Annotated[
        Path,
        typer.Option("--variants", help="Safety variants JSONL đã tạo offline."),
    ] = Path("artifacts/safety/safety-100/variants.jsonl"),
    output: Annotated[
        Path,
        typer.Option("--output", help="File JSONL ghi kết quả exploratory."),
    ] = Path("artifacts/safety/safety-100/results.jsonl"),
    max_concurrency: Annotated[
        int,
        typer.Option("--max-concurrency", min=1, max=16),
    ] = 4,
) -> None:
    """Chạy context gate của ba router trên safety variants; có provider calls."""

    try:
        variants = _load_safety_variants(variants_path)
        results = asyncio.run(
            execute_safety_benchmark(
                config,
                variants,
                max_concurrency=max_concurrency,
            )
        )
        output_path = write_safety_results(output, results)
        report_path = write_safety_report(
            output_path.with_name("report.md"),
            build_safety_markdown_report(variants, results),
        )
    except Exception as error:
        _exit_with_error(error)
    typer.echo(
        json.dumps(
            {
                "result_count": len(results),
                "passed": sum(result.passed for result in results),
                "errors": sum(result.error is not None for result in results),
                "results_jsonl": str(output_path),
                "report_markdown": str(report_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


async def execute_safety_benchmark(
    config_path: Path,
    variants: tuple[SafetyVariant, ...],
    *,
    max_concurrency: int,
    settings: Settings | None = None,
) -> tuple[SafetyAssessmentResult, ...]:
    """Wiring và chạy safety context assessment trong cùng một event loop.

    Args:
        config_path: Formal TOML cung cấp router/runtime settings.
        variants: Safety variants đã được parse và validate.
        max_concurrency: Số provider decision calls đồng thời.
        settings: Settings inject tùy chọn cho test.

    Returns:
        Các kết quả safety assessment theo variant và router.
    """

    plan = load_benchmark_execution_plan(config_path)
    active_settings = _settings_for_plan(settings or get_settings(), plan)
    bundle = build_runtime(
        active_settings,
        engine_kinds=plan.benchmark.routers,
        web_provider="mock",
        mock_web_path=plan.mock_web_path,
    )
    try:
        return await run_safety_exploratory(
            variants,
            bundle.engines,
            max_concurrency=max_concurrency,
        )
    finally:
        await bundle.close()


async def execute_query(
    query_text: str,
    *,
    engine: RouterKind,
    live_web: bool,
    include_trace: bool,
    settings: Settings | None = None,
) -> AnswerResponse:
    """Wiring và xử lý một query cho CLI.

    Args:
        query_text: Truy vấn người dùng.
        engine: Decision engine cần chạy.
        live_web: Có dùng Tavily thật hay không.
        include_trace: Có trả trace trong JSON hay không.
        settings: Settings inject cho test.

    Returns:
        Answer response typed.
    """

    active_settings = _absolutize_settings(settings or get_settings(), Path.cwd())
    if live_web:
        # Hard cap toàn query gồm cả retry attempts, không chỉ retrieval rounds.
        active_settings = active_settings.model_copy(
            update={"tavily_credit_budget": min(3, active_settings.tavily_credit_budget)}
        )
    mock_path = Path.cwd() / "data" / "raw" / "crag" / "mock_web.jsonl"
    bundle = build_runtime(
        active_settings,
        engine_kinds=(engine,),
        web_provider="tavily" if live_web else "mock",
        mock_web_path=mock_path if mock_path.is_file() else None,
    )
    try:
        return await bundle.pipeline.answer(
            QueryRequest(query=query_text),
            engine,
            include_trace=include_trace,
        )
    finally:
        await bundle.close()


async def execute_confirmatory_benchmark(
    config_path: Path,
    *,
    settings: Settings | None = None,
) -> tuple[ConfirmatoryReport, tuple[Path, Path]]:
    """Chạy các confirmatory look tuần tự trên cùng manifest/checkpoint.

    Args:
        config_path: TOML có section ``[confirmatory]``.
        settings: Settings inject cho integration test.

    Returns:
        Báo cáo look cuối đã chạy và đường dẫn JSON/Markdown.

    Raises:
        ConfigurationError: Khi config không bật confirmatory mode hoặc look
            không tạo đủ paired records vì budget/provider failure.
    """

    plan = load_benchmark_execution_plan(config_path)
    if not plan.confirmatory_enabled:
        raise ConfigurationError("Config chưa bật [confirmatory].enabled=true")
    run_directory = plan.benchmark.output_dir / plan.benchmark.run_id
    _, manifest_hash = _prepare_confirmatory_manifest(plan)
    checkpoint_path = run_directory / "adaptive-records-checkpoint.jsonl"
    records = _load_benchmark_records(checkpoint_path) if checkpoint_path.is_file() else ()
    latest_report: ConfirmatoryReport | None = None
    latest_paths: tuple[Path, Path] | None = None
    for look_index, sample_limit in enumerate(plan.confirmatory_looks, 1):
        allowed_ids = {sample.query_id for sample in plan.samples[:sample_limit]}
        look_records = tuple(record for record in records if record.query_id in allowed_ids)
        paired_ids = {
            record.query_id for record in look_records if record.router is RouterKind.JEV
        } & {record.query_id for record in look_records if record.router is RouterKind.LLM}
        if len(paired_ids) < sample_limit:
            _, artifact_paths, _ = await execute_benchmark(
                config_path,
                live=False,
                settings=settings,
                sample_limit=sample_limit,
            )
            records = _load_benchmark_records(artifact_paths.records_jsonl)
            look_records = tuple(record for record in records if record.query_id in allowed_ids)
        latest_report = _analyze_confirmatory_records(
            plan,
            look_records,
            look_index=look_index,
            manifest_hash=manifest_hash,
        )
        if latest_report.pair_count < sample_limit:
            latest_report = latest_report.model_copy(
                update={"decision": ConfirmatoryDecision.INCONCLUSIVE}
            )
        latest_paths = write_confirmatory_report(
            run_directory / "confirmatory-analysis.json",
            latest_report,
        )
        write_confirmatory_report(
            run_directory / f"confirmatory-look-{look_index}.json",
            latest_report,
        )
        if latest_report.decision != ConfirmatoryDecision.CONTINUE:
            break
    if latest_report is None or latest_paths is None:
        raise ConfigurationError("Confirmatory config không có look để chạy")
    return latest_report, latest_paths


def refresh_confirmatory_reports(
    config_path: Path,
) -> tuple[ConfirmatoryReport, tuple[Path, Path]]:
    """Tái phân tích các look đầy đủ từ checkpoint hiện có, không gọi provider.

    Args:
        config_path: TOML confirmatory gốc.

    Returns:
        Look report mới nhất và đường dẫn canonical JSON/Markdown.

    Raises:
        ConfigurationError: Khi chưa có look nào đủ paired records.
    """

    plan = load_benchmark_execution_plan(config_path)
    if not plan.confirmatory_enabled:
        raise ConfigurationError("Config chưa bật [confirmatory].enabled=true")
    run_directory = plan.benchmark.output_dir / plan.benchmark.run_id
    records_path = run_directory / "adaptive-records-checkpoint.jsonl"
    records = _load_benchmark_records(records_path)
    _, manifest_hash = _prepare_confirmatory_manifest(plan)
    latest_report: ConfirmatoryReport | None = None
    latest_paths: tuple[Path, Path] | None = None
    for look_index, sample_limit in enumerate(plan.confirmatory_looks, 1):
        allowed_ids = {sample.query_id for sample in plan.samples[:sample_limit]}
        look_records = tuple(record for record in records if record.query_id in allowed_ids)
        paired_ids = {
            record.query_id for record in look_records if record.router is RouterKind.JEV
        } & {record.query_id for record in look_records if record.router is RouterKind.LLM}
        if len(paired_ids) < sample_limit:
            break
        latest_report = _analyze_confirmatory_records(
            plan,
            look_records,
            look_index=look_index,
            manifest_hash=manifest_hash,
        )
        write_confirmatory_report(
            run_directory / f"confirmatory-look-{look_index}.json",
            latest_report,
        )
        latest_paths = write_confirmatory_report(
            run_directory / "confirmatory-analysis.json",
            latest_report,
        )
        if latest_report.decision == ConfirmatoryDecision.SUPPORTED:
            break
    if latest_report is None or latest_paths is None:
        raise ConfigurationError("Chưa có confirmatory look nào đủ paired records")
    return latest_report, latest_paths


def _analyze_confirmatory_records(
    plan: BenchmarkExecutionPlan,
    records: tuple[BenchmarkRecord, ...],
    *,
    look_index: int,
    manifest_hash: str,
) -> ConfirmatoryReport:
    """Áp dụng đúng preregistered thresholds cho một tập look records."""

    return analyze_confirmatory_look(
        records,
        run_id=plan.benchmark.run_id,
        look_index=look_index,
        planned_looks=plan.confirmatory_looks,
        manifest_sha256=manifest_hash,
        resamples=plan.benchmark.bootstrap_resamples,
        seed=plan.benchmark.seed,
        quality_margin=plan.benchmark.success_thresholds.quality_non_inferiority_margin,
        minimum_cost_reduction=plan.benchmark.success_thresholds.minimum_cost_reduction,
        minimum_p50_reduction=(plan.benchmark.success_thresholds.minimum_p50_latency_reduction),
        maximum_p95_regression=(plan.benchmark.success_thresholds.maximum_p95_latency_regression),
        maximum_error_rate=plan.benchmark.success_thresholds.maximum_unhandled_error_rate,
        familywise_alpha=plan.confirmatory_familywise_alpha,
    )


def _prepare_confirmatory_manifest(plan: BenchmarkExecutionPlan) -> tuple[Path, str]:
    """Ghi manifest từ execution plan confirmatory đã validate."""

    if not plan.confirmatory_enabled:
        raise ConfigurationError("Config chưa bật [confirmatory].enabled=true")
    path = plan.benchmark.output_dir / plan.benchmark.run_id / "confirmatory-manifest.json"
    digest = write_confirmatory_manifest(
        path,
        plan.samples,
        excluded_run_ids=plan.confirmatory_excluded_run_ids,
    )
    return path, digest


async def execute_benchmark(
    config_path: Path,
    *,
    live: bool,
    tavily_credit_budget: int | None = None,
    settings: Settings | None = None,
    sample_limit: int | None = None,
) -> tuple[str, ArtifactPaths, int | None]:
    """Chạy benchmark từ TOML, ghi artifacts và trả tóm tắt.

    Args:
        config_path: Formal hoặc live TOML.
        live: Có bật Tavily live hay không.
        tavily_credit_budget: Override hard cap cho live run.
        settings: Settings inject cho test.
        sample_limit: Giới hạn look hiện tại; chỉ thu hẹp ``run.max_samples``.

    Returns:
        Run ID, đường dẫn artifacts và tổng credit ledger nếu là live run.

    Raises:
        ConfigurationError: Khi dataset, frozen fixture hoặc vector index chưa sẵn sàng.
    """

    plan = load_benchmark_execution_plan(
        config_path,
        live=live,
        tavily_credit_budget=tavily_credit_budget,
    )
    if sample_limit is not None:
        configured_limit = plan.benchmark.max_samples or len(plan.samples)
        if not 1 <= sample_limit <= configured_limit:
            raise ConfigurationError(
                f"sample_limit phải nằm trong 1..{configured_limit}, nhận {sample_limit}"
            )
        plan = replace(
            plan,
            benchmark=plan.benchmark.model_copy(update={"max_samples": sample_limit}),
        )
    active_settings = _settings_for_plan(settings or get_settings(), plan)
    if not active_settings.qdrant_path.is_dir():
        raise ConfigurationError(
            f"Chưa có Qdrant index tại {active_settings.qdrant_path}. "
            "Hãy chạy `rag-router ingest --source <path>` trước benchmark."
        )
    run_directory = plan.benchmark.output_dir / plan.benchmark.run_id
    tavily_checkpoint = run_directory / "tavily-credit-checkpoint.json" if live else None
    counterfactual_path = run_directory / "counterfactual.jsonl"
    replay_calibration_path = getattr(plan, "replay_calibration_path", None)
    replay_counterfactual_path = getattr(plan, "replay_counterfactual_path", None)
    replay_source_run_id = getattr(plan, "replay_source_run_id", None)
    confirmatory_enabled = getattr(plan, "confirmatory_enabled", False)
    confirmatory_calibration_path = getattr(plan, "confirmatory_calibration_path", None)
    cost_budget_usd = getattr(plan, "cost_budget_usd", None)
    run_cost_ledger = (
        FileUsdBudgetLedger(
            run_directory / "cost-budget-checkpoint.json",
            limit_usd=cost_budget_usd,
        )
        if cost_budget_usd is not None
        else None
    )
    provider_budget_path = getattr(plan, "provider_budget_path", None)
    deepseek_limit_usd = getattr(plan, "deepseek_limit_usd", None)
    jev_limit_usd = getattr(plan, "jev_limit_usd", None)
    provider_cost_ledger = (
        FileProviderBudgetLedger(
            provider_budget_path,
            limits_usd={
                "deepseek": deepseek_limit_usd,
                "typesafe-jev": jev_limit_usd,
            },
        )
        if provider_budget_path is not None
        and deepseek_limit_usd is not None
        and jev_limit_usd is not None
        else None
    )
    counterfactual_ledgers: list[UsdLedger] = []
    if run_cost_ledger is not None:
        counterfactual_ledgers.append(run_cost_ledger)
    if provider_cost_ledger is not None:
        counterfactual_ledgers.append(
            ProviderScopedUsdLedger(
                provider_cost_ledger,
                provider="deepseek",
                namespace=plan.benchmark.run_id,
            )
        )
    counterfactual_cost_ledger = (
        None
        if not counterfactual_ledgers
        else counterfactual_ledgers[0]
        if len(counterfactual_ledgers) == 1
        else CombinedUsdLedger(*counterfactual_ledgers)
    )
    samples = (
        plan.samples[: plan.benchmark.max_samples]
        if plan.benchmark.max_samples is not None
        else plan.samples
    )

    if live:
        if plan.mock_web_path is None:
            raise ConfigurationError(
                "Live benchmark cần frozen web fixture để tạo counterfactual gold"
            )
        gold_bundle = build_runtime(
            active_settings,
            engine_kinds=(RouterKind.RULE,),
            web_provider="mock",
            mock_web_path=plan.mock_web_path,
        )
        try:
            samples = await attach_counterfactual_gold(
                samples,
                StaticBranchEvaluator(
                    gold_bundle.retrievers,
                    gold_bundle.generator,
                    max_concurrency=plan.benchmark.max_concurrency,
                ),
                quality_floor=plan.counterfactual_quality_floor,
                artifact_path=counterfactual_path,
                budget_ledger=counterfactual_cost_ledger,
                budget_phase="main",
                reserve_usd=getattr(plan, "counterfactual_reserve_usd", 0.25),
            )
        finally:
            await gold_bundle.close()
    bundle = build_runtime(
        active_settings,
        engine_kinds=plan.benchmark.routers,
        web_provider=plan.web_provider,
        mock_web_path=plan.mock_web_path,
        tavily_checkpoint_path=tavily_checkpoint,
        pipeline_policy=getattr(plan, "pipeline_policy", None),
    )
    raw_adapter = PipelineBenchmarkAdapter(
        bundle.pipeline,
        credit_meter=bundle.credit_meter,
    )
    calibration_models: dict[RouterKind, TemperatureCalibrationModel] | None = None
    try:
        if live:
            if plan.calibration_artifact_path is None:
                raise ConfigurationError(
                    "Live benchmark thiếu đường dẫn calibration artifact formal"
                )
            calibration_models = load_calibration_models(
                plan.calibration_artifact_path,
                plan.benchmark.routers,
            )
        else:
            # Calibration phải hoàn tất và được khóa trước khi tạo bất kỳ
            # held-out artifact nào, kể cả counterfactual gold của test split.
            if plan.benchmark.split is DatasetSplit.TEST:
                if confirmatory_enabled:
                    if confirmatory_calibration_path is None:
                        raise ConfigurationError(
                            "Confirmatory track thiếu frozen calibration artifact"
                        )
                    calibration_models = load_calibration_models(
                        confirmatory_calibration_path,
                        plan.benchmark.routers,
                    )
                elif replay_calibration_path is not None:
                    calibration_models = load_calibration_models(
                        replay_calibration_path,
                        plan.benchmark.routers,
                    )
                else:
                    calibration_models = await _fit_formal_calibration(
                        plan,
                        bundle,
                        run_directory=run_directory,
                        adapter=raw_adapter,
                        run_cost_ledger=run_cost_ledger,
                        provider_cost_ledger=provider_cost_ledger,
                        counterfactual_cost_ledger=counterfactual_cost_ledger,
                    )
            if not confirmatory_enabled:
                samples = await attach_counterfactual_gold(
                    samples,
                    StaticBranchEvaluator(
                        bundle.retrievers,
                        bundle.generator,
                        max_concurrency=plan.benchmark.max_concurrency,
                    ),
                    quality_floor=plan.counterfactual_quality_floor,
                    artifact_path=replay_counterfactual_path or counterfactual_path,
                    budget_ledger=(
                        None
                        if replay_counterfactual_path is not None
                        else counterfactual_cost_ledger
                    ),
                    budget_phase="main",
                    reserve_usd=getattr(plan, "counterfactual_reserve_usd", 0.25),
                )
            else:
                samples = tuple(
                    sample.model_copy(
                        update={
                            "metadata": {
                                **sample.metadata,
                                "confirmatory_track": True,
                            }
                        }
                    )
                    for sample in samples
                )
            if replay_source_run_id is not None:
                samples = tuple(
                    sample.model_copy(
                        update={
                            "metadata": {
                                **sample.metadata,
                                "frozen_gold_run_id": replay_source_run_id,
                            }
                        }
                    )
                    for sample in samples
                )
    except (FileNotFoundError, ValueError) as error:
        await bundle.close()
        if live:
            raise ConfigurationError(
                f"Live benchmark chỉ được dùng calibration artifact từ formal run: {error}. "
                "Hãy chạy `rag-router benchmark` trước."
            ) from error
        raise
    except Exception:
        await bundle.close()
        raise

    benchmark_adapter = (
        CalibratedBenchmarkAdapter(raw_adapter, calibration_models)
        if calibration_models is not None
        else raw_adapter
    )
    checkpoint_hook = FileCheckpointHook(run_directory / "checkpoint.json")
    budget_hooks: list[BudgetHook] = []
    if live and bundle.tavily_ledger is not None:
        budget_hooks.append(TavilyBudgetHook(bundle.tavily_ledger))
    if run_cost_ledger is not None:
        budget_hooks.append(
            UsdBudgetHook(
                run_cost_ledger,
                run_id=plan.benchmark.run_id,
                reserve_usd=getattr(plan, "adaptive_reserve_usd", 0.50),
            )
        )
    if provider_cost_ledger is not None:
        budget_hooks.append(
            ProviderUsdBudgetHook(
                provider_cost_ledger,
                run_id=plan.benchmark.run_id,
                deepseek_reserve_usd=getattr(plan, "adaptive_reserve_usd", 0.50),
                jev_reserve_usd=getattr(plan, "jev_reserve_usd", 0.05),
            )
        )
    budget_hook = (
        None
        if not budget_hooks
        else budget_hooks[0]
        if len(budget_hooks) == 1
        else CombinedBudgetHook(*budget_hooks)
    )
    runner = BenchmarkRunner(
        benchmark_adapter,
        samples=samples,
        budget_hook=budget_hook,
        checkpoint_hook=checkpoint_hook,
        record_store=FileBenchmarkRecordStore(run_directory / "adaptive-records-checkpoint.jsonl"),
    )
    try:
        report = await runner.run(plan.benchmark)
        paths = BenchmarkArtifactWriter(plan.benchmark.output_dir).write_all(report)
        credits = bundle.tavily_ledger.consumed if bundle.tavily_ledger else None
        return report.run_id, paths, credits
    finally:
        await bundle.close()


async def _fit_formal_calibration(
    plan: BenchmarkExecutionPlan,
    bundle: RuntimeBundle,
    *,
    run_directory: Path,
    adapter: PipelineBenchmarkAdapter,
    run_cost_ledger: FileUsdBudgetLedger | None,
    provider_cost_ledger: FileProviderBudgetLedger | None,
    counterfactual_cost_ledger: UsdLedger | None,
) -> dict[RouterKind, TemperatureCalibrationModel]:
    """Chạy calibration split bằng frozen providers rồi khóa artifact theo router."""

    if not plan.calibration_samples:
        raise ConfigurationError("Formal test run thiếu calibration split")
    calibration_samples = await attach_counterfactual_gold(
        plan.calibration_samples,
        StaticBranchEvaluator(
            bundle.retrievers,
            bundle.generator,
            max_concurrency=plan.benchmark.max_concurrency,
        ),
        quality_floor=plan.counterfactual_quality_floor,
        artifact_path=run_directory / "calibration-counterfactual.jsonl",
        budget_ledger=counterfactual_cost_ledger,
        budget_phase="calibration",
        reserve_usd=plan.counterfactual_reserve_usd,
    )
    calibration_config = plan.benchmark.model_copy(
        update={
            "run_id": f"{plan.benchmark.run_id}-calibration",
            "split": DatasetSplit.CALIBRATION,
            "live": False,
            "max_samples": None,
        }
    )
    calibration_budget_hooks: list[BudgetHook] = []
    if run_cost_ledger is not None:
        calibration_budget_hooks.append(
            UsdBudgetHook(
                run_cost_ledger,
                run_id=calibration_config.run_id,
                reserve_usd=plan.adaptive_reserve_usd,
            )
        )
    if provider_cost_ledger is not None:
        calibration_budget_hooks.append(
            ProviderUsdBudgetHook(
                provider_cost_ledger,
                run_id=calibration_config.run_id,
                deepseek_reserve_usd=plan.adaptive_reserve_usd,
                jev_reserve_usd=plan.jev_reserve_usd,
            )
        )
    calibration_budget_hook = (
        None
        if not calibration_budget_hooks
        else calibration_budget_hooks[0]
        if len(calibration_budget_hooks) == 1
        else CombinedBudgetHook(*calibration_budget_hooks)
    )
    calibration_runner = BenchmarkRunner(
        adapter,
        samples=calibration_samples,
        budget_hook=calibration_budget_hook,
        checkpoint_hook=FileCheckpointHook(run_directory / "calibration-checkpoint.json"),
        record_store=FileBenchmarkRecordStore(
            run_directory / "calibration-records-checkpoint.jsonl"
        ),
    )
    calibration_report = await calibration_runner.run(calibration_config)
    BenchmarkArtifactWriter(calibration_config.output_dir).write_all(calibration_report)
    if calibration_report.stopped_early:
        raise ConfigurationError(
            "Calibration pilot dừng có kiểm soát vì cost budget không còn đủ reserve"
        )
    models = fit_router_calibrators(calibration_report.records, calibration_config.routers)
    write_calibration_models(run_directory / "calibration-models.json", models)
    return models


async def execute_ingest(
    source: Path,
    *,
    recreate: bool,
    batch_size: int,
    settings: Settings | None = None,
) -> int:
    """Ingest tài liệu mà không tạo LLM/router client.

    Args:
        source: File hoặc directory dữ liệu.
        recreate: Có tạo lại collection hay không.
        batch_size: Kích thước batch embedding.
        settings: Settings inject cho test.

    Returns:
        Tổng số passage đã upsert.
    """

    active_settings = _absolutize_settings(settings or get_settings(), Path.cwd())
    retriever = QdrantVectorRetriever(
        collection_name=active_settings.vector_collection,
        embedder=BgeM3EmbeddingAdapter(active_settings.embedding_model),
        path=active_settings.qdrant_path,
        top_k=active_settings.vector_top_k,
        evidence_limit=active_settings.evidence_limit,
    )
    return await ingest_source_path(
        source,
        retriever,
        recreate=recreate,
        batch_size=batch_size,
    )


def read_report(run_id: str, settings: Settings) -> str:
    """Đọc report theo run ID và chặn path traversal.

    Args:
        run_id: Định danh run chỉ gồm chữ, số, chấm, gạch ngang hoặc gạch dưới.
        settings: Cấu hình thư mục artifacts.

    Returns:
        Nội dung Markdown của report.

    Raises:
        ConfigurationError: Khi run ID không hợp lệ hoặc report chưa tồn tại.
    """

    if re.fullmatch(r"[A-Za-z0-9_.-]+", run_id) is None:
        raise ConfigurationError("run-id chỉ được chứa chữ, số, '.', '_' và '-'")
    root = settings.artifacts_dir.expanduser().resolve() / "benchmarks"
    report_path = (root / run_id / "report.md").resolve()
    if report_path.parent.parent != root:
        raise ConfigurationError("run-id không hợp lệ")
    if not report_path.is_file():
        raise ConfigurationError(f"Không tìm thấy report cho run {run_id!r}: {report_path}")
    return report_path.read_text(encoding="utf-8")


def _load_benchmark_records(path: Path) -> tuple[BenchmarkRecord, ...]:
    if not path.is_file():
        raise ConfigurationError(
            f"Không tìm thấy formal records tại {path}. Hãy chạy benchmark trước."
        )
    records: list[BenchmarkRecord] = []
    _line_number = 0
    try:
        for _line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            records.append(BenchmarkRecord.model_validate_json(line))
    except (OSError, ValueError) as error:
        raise ConfigurationError(
            f"Không thể đọc benchmark record tại dòng {_line_number}: {path}"
        ) from error
    return tuple(records)


def _load_safety_variants(path: Path) -> tuple[SafetyVariant, ...]:
    if not path.is_file():
        raise ConfigurationError(
            f"Không tìm thấy safety variants tại {path}. Hãy chạy prepare-safety trước."
        )
    variants: list[SafetyVariant] = []
    _line_number = 0
    try:
        for _line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.strip():
                variants.append(SafetyVariant.model_validate_json(line))
    except (OSError, ValueError) as error:
        raise ConfigurationError(
            f"Không thể đọc safety variant tại dòng {_line_number}: {path}"
        ) from error
    if not variants:
        raise ConfigurationError(f"Safety variants rỗng: {path}")
    return tuple(variants)


def _settings_for_plan(settings: Settings, plan: BenchmarkExecutionPlan) -> Settings:
    rooted = _absolutize_settings(settings, plan.project_root)
    return rooted.model_copy(
        update={
            "vector_top_k": plan.vector_top_k,
            "evidence_limit": plan.evidence_limit,
            "web_max_results": plan.web_max_results,
            "max_repair_rounds": plan.max_repair_rounds,
            "max_output_tokens": plan.max_output_tokens,
            "tavily_credit_budget": plan.tavily_credit_budget,
            "tavily_requests_per_minute": plan.tavily_requests_per_minute,
            "tavily_max_concurrency": plan.tavily_max_concurrency,
            "tavily_max_retries": plan.tavily_max_retries,
        }
    )


def _absolutize_settings(settings: Settings, project_root: Path) -> Settings:
    root = project_root.resolve()

    def rooted(path: Path) -> Path:
        return path.resolve() if path.is_absolute() else (root / path).resolve()

    return settings.model_copy(
        update={
            "qdrant_path": rooted(settings.qdrant_path),
            "cache_dir": rooted(settings.cache_dir),
            "artifacts_dir": rooted(settings.artifacts_dir),
        }
    )


def _echo_artifacts(
    run_id: str,
    paths: ArtifactPaths,
    *,
    tavily_credits_consumed: int | None = None,
) -> None:
    payload: dict[str, object] = {
        "run_id": run_id,
        "records_jsonl": str(paths.records_jsonl),
        "records_parquet": str(paths.records_parquet),
        "report_markdown": str(paths.report_markdown),
        "pareto_svg": str(paths.pareto_svg),
        "reliability_svg": str(paths.reliability_svg),
        "risk_coverage_svg": str(paths.risk_coverage_svg),
    }
    if tavily_credits_consumed is not None:
        payload["tavily_credits_consumed"] = tavily_credits_consumed
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))


def _exit_with_error(error: Exception) -> Never:
    if isinstance(error, (AdaptiveRAGError, FileNotFoundError, ValidationError, ValueError)):
        message = str(error)
    else:
        message = f"Lỗi runtime không dự kiến ({type(error).__name__})"
    typer.echo(f"Lỗi: {message}", err=True)
    raise typer.Exit(code=2)


if __name__ == "__main__":
    app()
