"""Chạy và cache sáu static branches để tạo gold route cho benchmark."""

from __future__ import annotations

import asyncio
from pathlib import Path

from adaptive_rag_router.domain.enums import ModelTier, RetrievalSource
from adaptive_rag_router.domain.errors import ConfigurationError, CreditBudgetExceededError
from adaptive_rag_router.domain.models import QueryRequest, RetrievalResult
from adaptive_rag_router.evaluation import (
    BenchmarkSample,
    CounterfactualOutcome,
    CounterfactualRunner,
    calculate_answer_quality_metrics,
)
from adaptive_rag_router.evaluation.cost_budget import UsdLedger
from adaptive_rag_router.evaluation.models import CounterfactualRunResult
from adaptive_rag_router.generation import Generator
from adaptive_rag_router.retrieval import RetrieverRegistry


class StaticBranchEvaluator:
    """Đánh giá `{NONE,VECTOR,WEB} x {ECONOMY,STRONG}` không qua router.

    Retrieval được cache theo query/source để hai model tier dùng đúng cùng một
    context. Web provider của formal benchmark phải là mock/frozen ở runtime.
    """

    def __init__(
        self,
        retrievers: RetrieverRegistry,
        generator: Generator,
        *,
        max_concurrency: int = 4,
    ) -> None:
        """Khởi tạo evaluator với dependency đã được wiring.

        Args:
            retrievers: Registry vector và frozen-web retriever.
            generator: Generator dùng chung cho hai model tier.
            max_concurrency: Số generation call đồng thời tối đa.

        Raises:
            ValueError: Nếu concurrency không dương.
        """

        if max_concurrency < 1:
            raise ValueError("max_concurrency phải dương")
        self._retrievers = retrievers
        self._generator = generator
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._retrieval_lock = asyncio.Lock()
        self._retrieval_cache: dict[tuple[str, RetrievalSource], RetrievalResult] = {}

    async def __call__(
        self,
        sample: BenchmarkSample,
        source: RetrievalSource,
        tier: ModelTier,
    ) -> CounterfactualOutcome:
        """Chạy một branch cố định và chấm token-F1 với reference answer.

        Args:
            sample: Sample benchmark có reference answer.
            source: Nguồn retrieval cố định của branch.
            tier: Model tier cố định của branch.

        Returns:
            Outcome typed; provider failure được ghi `succeeded=false`.

        Raises:
            ConfigurationError: Khi sample không có reference answer.
        """

        if sample.reference_answer is None:
            raise ConfigurationError(
                f"Sample {sample.query_id!r} thiếu reference answer để tạo gold route"
            )
        request = QueryRequest(
            query=sample.query,
            query_id=sample.query_id,
            metadata=sample.metadata,
        )
        try:
            retrieval = await self._retrieve(sample, source)
            async with self._semaphore:
                generation = await self._generator.generate(
                    request,
                    retrieval.documents,
                    tier,
                )
        except Exception as error:
            return CounterfactualOutcome(
                query_id=sample.query_id,
                retrieval_source=source,
                model_tier=tier,
                quality_score=0.0,
                cost_usd=0.0,
                latency_ms=0.0,
                succeeded=False,
                metadata={"error": type(error).__name__},
            )

        quality = calculate_answer_quality_metrics(
            [generation.text],
            [sample.reference_answer],
        ).token_f1
        return CounterfactualOutcome(
            query_id=sample.query_id,
            retrieval_source=source,
            model_tier=tier,
            quality_score=quality,
            cost_usd=generation.cost_usd,
            latency_ms=retrieval.latency_ms + generation.latency_ms,
            succeeded=True,
            metadata={
                "model_id": generation.model_id,
                "retrieval_provider": retrieval.provider,
                "document_count": len(retrieval.documents),
                "answer": generation.text,
            },
        )

    async def _retrieve(
        self,
        sample: BenchmarkSample,
        source: RetrievalSource,
    ) -> RetrievalResult:
        if source is RetrievalSource.NONE:
            return RetrievalResult(
                source=source,
                query=sample.query,
                provider="none",
                cached=True,
            )
        key = (sample.query_id, source)
        async with self._retrieval_lock:
            cached = self._retrieval_cache.get(key)
            if cached is not None:
                return cached
            result = await self._retrievers.get(source).retrieve(
                sample.query,
                round_index=0,
            )
            self._retrieval_cache[key] = result
            return result


async def attach_counterfactual_gold(
    samples: tuple[BenchmarkSample, ...],
    evaluator: StaticBranchEvaluator,
    *,
    quality_floor: float,
    artifact_path: Path,
    budget_ledger: UsdLedger | None = None,
    budget_phase: str = "main",
    reserve_usd: float = 0.25,
) -> tuple[BenchmarkSample, ...]:
    """Tạo hoặc nạp gold action rồi gắn `expected_route` vào samples.

    Args:
        samples: Samples của đúng split sắp benchmark.
        evaluator: Static branch evaluator dùng frozen providers.
        quality_floor: Ngưỡng chọn branch rẻ nhất.
        artifact_path: JSONL cache chứa đủ outcomes và gold action.
        budget_ledger: Ledger USD dùng chung, nếu run có hard cap.
        budget_phase: Namespace để khóa idempotent giữa calibration/main.
        reserve_usd: Vùng đệm trước sáu branches của một query.

    Returns:
        Samples mới có expected route dạng `source:tier`.

    Raises:
        ConfigurationError: Khi cache cũ không khớp sample set.
    """

    expected_ids = {sample.query_id for sample in samples}
    if await asyncio.to_thread(artifact_path.is_file):
        results = await asyncio.to_thread(_load_results, artifact_path)
        unexpected_ids = set(results) - expected_ids
        if unexpected_ids:
            raise ConfigurationError(
                f"Counterfactual cache chứa query ngoài split hiện tại: "
                f"{sorted(unexpected_ids)} tại {artifact_path}. "
                "Hãy xóa cache hoặc dùng run-id khác."
            )
    else:
        results = {}

    runner = CounterfactualRunner(evaluator, max_concurrency=6)
    for sample in samples:
        if sample.query_id in results:
            if budget_ledger is not None:
                await budget_ledger.commit(
                    f"counterfactual:{budget_phase}:{sample.query_id}",
                    sum(outcome.cost_usd for outcome in results[sample.query_id].outcomes),
                )
            continue
        unit_id = f"counterfactual:{budget_phase}:{sample.query_id}"
        if budget_ledger is not None and not await budget_ledger.reserve(
            unit_id,
            reserve_usd=reserve_usd,
        ):
            raise CreditBudgetExceededError(
                f"Dừng trước query {sample.query_id!r}: reserve ${reserve_usd:.2f} "
                "sẽ vượt hard cost budget"
            )
        result = await runner.run(sample, quality_floor=quality_floor)
        results[result.query_id] = result
        ordered_checkpoint = [
            results[item.query_id] for item in samples if item.query_id in results
        ]
        # Ghi atomic sau từng query để formal run dài có thể resume mà không
        # lặp lại những provider calls đã hoàn tất.
        await asyncio.to_thread(_write_results, artifact_path, ordered_checkpoint)
        if budget_ledger is not None:
            await budget_ledger.commit(
                unit_id,
                sum(outcome.cost_usd for outcome in result.outcomes),
            )

    if set(results) != expected_ids:
        missing_ids = expected_ids - set(results)
        raise ConfigurationError(
            f"Counterfactual cache thiếu query sau resume: {sorted(missing_ids)}"
        )

    return tuple(
        sample.model_copy(
            update={
                "expected_route": results[sample.query_id].gold_action.route_label,
                "metadata": {
                    **sample.metadata,
                    "counterfactual_no_good_route": results[
                        sample.query_id
                    ].gold_action.no_good_route,
                },
            }
        )
        for sample in samples
    )


def _load_results(path: Path) -> dict[str, CounterfactualRunResult]:
    results: dict[str, CounterfactualRunResult] = {}
    try:
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            result = CounterfactualRunResult.model_validate_json(line)
            if result.query_id in results:
                raise ConfigurationError(f"Counterfactual cache trùng query tại dòng {line_number}")
            results[result.query_id] = result
    except (OSError, ValueError) as error:
        raise ConfigurationError(f"Không thể đọc counterfactual cache {path}: {error}") from error
    return results


def _write_results(
    path: Path,
    results: list[CounterfactualRunResult],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = "\n".join(result.model_dump_json() for result in results)
    temporary.write_text(payload + ("\n" if payload else ""), encoding="utf-8")
    temporary.replace(path)
