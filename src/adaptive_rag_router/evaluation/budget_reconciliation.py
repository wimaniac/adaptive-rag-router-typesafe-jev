"""Reconcile artifacts lịch sử vào hard-cap ledger project-wide theo provider."""

from __future__ import annotations

import asyncio
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from adaptive_rag_router.domain.enums import RouterKind
from adaptive_rag_router.domain.errors import ConfigurationError
from adaptive_rag_router.evaluation.cost_budget import (
    FileProviderBudgetLedger,
    ProviderScopedUsdLedger,
    ProviderUsdBudgetHook,
)
from adaptive_rag_router.evaluation.models import BenchmarkRecord, CounterfactualRunResult
from adaptive_rag_router.evaluation.safety import SafetyAssessmentResult


class ProviderBudgetReconciliation(BaseModel):
    """Kết quả migrate artifacts hiện có vào provider ledger."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_ids: tuple[str, ...]
    counterfactual_units: int = Field(ge=0)
    adaptive_units: int = Field(ge=0)
    safety_units: int = Field(ge=0)
    consumed_usd: dict[str, float]
    remaining_usd: dict[str, float]
    ledger_path: Path


async def reconcile_provider_budget(
    benchmark_root: Path,
    run_ids: tuple[str, ...],
    ledger: FileProviderBudgetLedger,
    *,
    safety_results_path: Path | None = None,
) -> ProviderBudgetReconciliation:
    """Nạp cost artifacts mà không gọi provider rồi commit idempotent vào ledger.

    Args:
        benchmark_root: Thư mục chứa các run benchmark.
        run_ids: Run IDs gốc; calibration artifacts được đọc trong cùng thư mục.
        ledger: Project-wide provider ledger đích.
        safety_results_path: Safety result JSONL tùy chọn.

    Returns:
        Số unit đã reconcile cùng cost consumed/remaining.

    Raises:
        ConfigurationError: Khi run ID/path hoặc artifact JSONL không hợp lệ.
    """

    if not run_ids:
        raise ValueError("cần ít nhất một run-id để reconcile")
    root = await asyncio.to_thread(lambda: benchmark_root.expanduser().resolve())
    counterfactual_units = 0
    adaptive_units = 0
    for run_id in run_ids:
        run_directory = await asyncio.to_thread(_resolve_run_directory, root, run_id)
        for filename, phase in (
            ("counterfactual.jsonl", "main"),
            ("calibration-counterfactual.jsonl", "calibration"),
        ):
            path = run_directory / filename
            if not await asyncio.to_thread(path.is_file):
                continue
            scoped = ProviderScopedUsdLedger(
                ledger,
                provider="deepseek",
                namespace=run_id,
            )
            results = await asyncio.to_thread(_read_jsonl, path, CounterfactualRunResult)
            for result in results:
                await scoped.commit(
                    f"counterfactual:{phase}:{result.query_id}",
                    sum(outcome.cost_usd for outcome in result.outcomes),
                )
                counterfactual_units += 1

        main_records = run_directory / "adaptive-records-checkpoint.jsonl"
        if not await asyncio.to_thread(main_records.is_file):
            main_records = run_directory / "records.jsonl"
        for path, adaptive_run_id in (
            (main_records, run_id),
            (run_directory / "calibration-records-checkpoint.jsonl", None),
        ):
            if not await asyncio.to_thread(path.is_file):
                continue
            records = await asyncio.to_thread(_read_jsonl, path, BenchmarkRecord)
            for record in records:
                hook = ProviderUsdBudgetHook(
                    ledger,
                    run_id=adaptive_run_id or record.run_id,
                    deepseek_reserve_usd=1.0,
                    jev_reserve_usd=1.0,
                )
                await hook.commit(record)
                adaptive_units += 1

    safety_units = 0
    if safety_results_path is not None:
        safety_results = await asyncio.to_thread(
            _read_jsonl,
            safety_results_path,
            SafetyAssessmentResult,
        )
        for safety_result in safety_results:
            costs = _safety_provider_costs(safety_result)
            await ledger.commit(
                f"safety:{safety_result.variant_id}:{safety_result.router.value}",
                costs,
            )
            safety_units += 1

    return ProviderBudgetReconciliation(
        run_ids=run_ids,
        counterfactual_units=counterfactual_units,
        adaptive_units=adaptive_units,
        safety_units=safety_units,
        consumed_usd=ledger.consumed_usd,
        remaining_usd=ledger.remaining_usd,
        ledger_path=ledger.path,
    )


def _resolve_run_directory(root: Path, run_id: str) -> Path:
    if not run_id or any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
        for character in run_id
    ):
        raise ConfigurationError(f"run-id không hợp lệ: {run_id!r}")
    path = (root / run_id).resolve()
    if path.parent != root or not path.is_dir():
        raise ConfigurationError(f"Không tìm thấy benchmark run: {path}")
    return path


def _read_jsonl[ModelT: BaseModel](path: Path, model: type[ModelT]) -> tuple[ModelT, ...]:
    records: list[ModelT] = []
    _line_number = 0
    try:
        for _line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.strip():
                records.append(model.model_validate_json(line))
    except (OSError, ValueError) as error:
        raise ConfigurationError(f"Artifact hỏng tại {path}, dòng {_line_number}") from error
    return tuple(records)


def _safety_provider_costs(result: SafetyAssessmentResult) -> dict[str, float]:
    if result.router is RouterKind.JEV:
        return {"typesafe-jev": result.cost_usd}
    if result.router is RouterKind.LLM:
        return {"deepseek": result.cost_usd}
    return {}
