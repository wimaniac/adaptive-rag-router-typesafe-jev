"""Ghép decision engines, retrieval adapters và generation thành runtime dùng chung."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol, cast

from openai import AsyncOpenAI

from adaptive_rag_router.application import AdaptiveRAGRouter, PipelinePolicy
from adaptive_rag_router.config import Settings
from adaptive_rag_router.domain.enums import (
    FallbackAction,
    ModelTier,
    RetrievalSource,
    RouterKind,
)
from adaptive_rag_router.domain.errors import ConfigurationError, ProviderError
from adaptive_rag_router.domain.models import QueryRequest, RetrievedDocument
from adaptive_rag_router.evaluation import (
    BenchmarkSample,
    MulticlassTemperatureCalibrator,
    PipelineRunResult,
    TemperatureCalibrationModel,
    load_local_records,
)
from adaptive_rag_router.evaluation.models import BenchmarkCheckpoint, BenchmarkRecord
from adaptive_rag_router.generation import (
    DeepSeekGenerator,
    DeepSeekQueryRewriter,
    Generator,
    QueryRewriter,
)
from adaptive_rag_router.retrieval import (
    BgeM3EmbeddingAdapter,
    CreditLedger,
    FileSearchCache,
    HttpxTavilyTransport,
    JsonlMockWebRetriever,
    MockWebRetriever,
    QdrantVectorRetriever,
    Retriever,
    RetrieverRegistry,
    RollingWindowRateLimiter,
    TavilyRetriever,
    TavilySearchTransport,
)
from adaptive_rag_router.routing import DecisionEngine, JevClient, build_decision_engine

type EngineFactory = Callable[[RouterKind, Settings], DecisionEngine]


class AsyncCloseable(Protocol):
    """Bề mặt tối thiểu của resource bất đồng bộ cần đóng sau một CLI run."""

    def close(self) -> Awaitable[None]:
        """Đóng resource và giải phóng kết nối."""


class AsyncAcloseAdapter:
    """Đổi lifecycle `aclose()` của SDK thành contract `close()` nội bộ."""

    def __init__(self, resource: object) -> None:
        """Khởi tạo wrapper cho resource có coroutine `aclose`.

        Args:
            resource: SDK client do runtime sở hữu.
        """

        self._resource = resource

    async def close(self) -> None:
        """Gọi `aclose` nếu resource cung cấp method này."""

        aclose = getattr(self._resource, "aclose", None)
        if callable(aclose):
            await aclose()


class ExternalCreditMeter:
    """Đếm provider attempts theo asyncio task bằng context-local state."""

    def __init__(self, max_credits_per_scope: int = 3) -> None:
        """Khởi tạo meter và hard cap theo một query-router workflow.

        Args:
            max_credits_per_scope: Số Tavily attempts tối đa trong một scope.

        Raises:
            ValueError: Nếu hard cap không dương.
        """

        if max_credits_per_scope < 1:
            raise ValueError("max_credits_per_scope phải dương")
        self._max_credits_per_scope = max_credits_per_scope
        self._credits: ContextVar[int] = ContextVar("tavily_external_credits", default=0)

    def start(self) -> Token[int]:
        """Bắt đầu một scope đo mới và trả token để khôi phục context."""

        return self._credits.set(0)

    def consume(self, credits: int = 1) -> None:
        """Cộng credit của một provider attempt trong task hiện tại.

        Args:
            credits: Số credit cần cộng.
        """

        self._credits.set(self._credits.get() + credits)

    def ensure_available(self) -> None:
        """Từ chối attempt tiếp theo trước khi global ledger bị tiêu hao."""

        if self._credits.get() >= self._max_credits_per_scope:
            raise ProviderError(
                f"Tavily per-query hard cap đã đạt ({self._max_credits_per_scope} attempts)"
            )

    def finish(self, token: Token[int]) -> int:
        """Kết thúc scope, trả tổng credit và khôi phục context trước đó.

        Args:
            token: Token do ``start`` trả về.

        Returns:
            Số credit đã ghi trong scope.
        """

        consumed = self._credits.get()
        self._credits.reset(token)
        return consumed


class MeteredTavilyTransport:
    """Bọc Tavily transport để đếm cả request thành công lẫn retry 429."""

    def __init__(
        self,
        transport: TavilySearchTransport,
        meter: ExternalCreditMeter,
    ) -> None:
        """Khởi tạo wrapper từ transport và meter dùng chung.

        Args:
            transport: Tavily transport thật.
            meter: Meter context-local của benchmark adapter.
        """

        self._transport = transport
        self._meter = meter

    async def search(self, query: str, *, max_results: int) -> Mapping[str, Any]:
        """Ghi một Basic Search credit rồi chuyển tiếp provider call.

        Args:
            query: Truy vấn đã chuẩn hóa.
            max_results: Số kết quả tối đa.

        Returns:
            Provider response chưa chuẩn hóa.
        """

        self._meter.consume(1)
        return await self._transport.search(query, max_results=max_results)

    async def close(self) -> None:
        """Đóng transport gốc nếu nó cung cấp lifecycle method."""

        close = getattr(self._transport, "close", None)
        if callable(close):
            await close()


class TavilyBudgetHook:
    """Dừng cấp lượt benchmark mới sau khi hard credit ledger đã hết."""

    def __init__(self, ledger: CreditLedger) -> None:
        """Khởi tạo hook từ ledger live dùng chung.

        Args:
            ledger: Ledger được TavilyRetriever tiêu thụ trước mỗi attempt.
        """

        self._ledger = ledger

    async def reserve(self, sample: BenchmarkSample, router: RouterKind) -> bool:
        """Cho phép lượt mới khi ledger vẫn còn ít nhất một credit.

        Args:
            sample: Sample sắp chạy.
            router: Router sắp chạy.

        Returns:
            ``False`` khi live run cần dừng có kiểm soát.
        """

        del sample, router
        return self._ledger.remaining > 0

    async def commit(self, record: BenchmarkRecord) -> None:
        """Không làm gì vì retriever đã commit credit theo từng attempt.

        Args:
            record: Record vừa hoàn tất.
        """

        del record


@dataclass(slots=True)
class RuntimeBundle:
    """Tập service đã wiring cùng resource lifecycle.

    Args:
        pipeline: Adaptive RAG pipeline công khai.
        settings: Settings thực tế sau khi áp override từ benchmark config.
        engines: Mapping decision engines dùng chung với pipeline.
        generator: Generator dùng cho answer và counterfactual branches.
        retrievers: Registry vector/web dùng chung trong runtime.
        tavily_ledger: Ledger live, hoặc ``None`` cho mock web.
        credit_meter: Meter task-local để report đủ retry credits.
    """

    pipeline: AdaptiveRAGRouter
    settings: Settings
    engines: Mapping[RouterKind, DecisionEngine]
    generator: Generator
    retrievers: RetrieverRegistry
    tavily_ledger: CreditLedger | None = None
    credit_meter: ExternalCreditMeter | None = None
    _closeables: tuple[AsyncCloseable, ...] = field(default=(), repr=False)

    async def close(self) -> None:
        """Đóng các HTTP client do runtime tự tạo theo thứ tự ngược."""

        for closeable in reversed(self._closeables):
            try:
                await closeable.close()
            except Exception:
                # Cleanup không được che mất kết quả hoặc lỗi chính của CLI.
                continue


def build_runtime(
    settings: Settings,
    *,
    engine_kinds: Sequence[RouterKind],
    web_provider: str = "mock",
    mock_web_path: Path | None = None,
    tavily_checkpoint_path: Path | None = None,
    engine_factory: EngineFactory | None = None,
    generator: Generator | None = None,
    query_rewriter: QueryRewriter | None = None,
    deepseek_client: AsyncOpenAI | Any | None = None,
    jev_client: JevClient | None = None,
    pipeline_policy: PipelinePolicy | None = None,
) -> RuntimeBundle:
    """Khởi tạo runtime mà chưa thực hiện provider request hay tải embedding model.

    Args:
        settings: Cấu hình provider, model và guardrail.
        engine_kinds: Các decision engine cần đăng ký.
        web_provider: ``mock`` hoặc ``tavily``.
        mock_web_path: Frozen fixture tùy chọn cho mock provider.
        tavily_checkpoint_path: File ledger riêng của live run.
        engine_factory: Factory inject cho test; nếu có sẽ không tạo provider router.
        generator: Generator inject cho test.
        query_rewriter: Rewriter inject cho test.
        deepseek_client: Client dùng chung tùy chọn.
        jev_client: Jev client inject tùy chọn.
        pipeline_policy: Typed baseline/ablation switches.

    Returns:
        Runtime bundle sẵn sàng xử lý query.

    Raises:
        ConfigurationError: Khi provider/secret hoặc web fixture không hợp lệ.
    """

    kinds = tuple(dict.fromkeys(engine_kinds))
    if not kinds:
        raise ConfigurationError("Cần ít nhất một decision engine")
    if web_provider not in {"mock", "tavily"}:
        raise ConfigurationError("web_provider chỉ nhận 'mock' hoặc 'tavily'")

    owned_closeables: list[AsyncCloseable] = []
    shared_client = deepseek_client
    needs_deepseek = (
        generator is None
        or query_rewriter is None
        or (RouterKind.LLM in kinds and engine_factory is None)
    )
    if needs_deepseek and shared_client is None:
        shared_client = _build_deepseek_client(settings)
        owned_closeables.append(cast(AsyncCloseable, shared_client))

    if engine_factory is None:
        engines = {
            kind: build_decision_engine(
                kind,
                settings,
                llm_client=shared_client,
                jev_client=jev_client,
            )
            for kind in kinds
        }
        if RouterKind.JEV in kinds and jev_client is None:
            owned_closeables.append(
                AsyncAcloseAdapter(getattr(engines[RouterKind.JEV], "client", object()))
            )
    else:
        engines = {kind: engine_factory(kind, settings) for kind in kinds}

    active_generator = generator or DeepSeekGenerator(settings, client=shared_client)
    active_rewriter = query_rewriter or DeepSeekQueryRewriter(
        settings,
        client=shared_client,
    )
    vector = QdrantVectorRetriever(
        collection_name=settings.vector_collection,
        embedder=BgeM3EmbeddingAdapter(settings.embedding_model),
        path=settings.qdrant_path,
        top_k=settings.vector_top_k,
        evidence_limit=settings.evidence_limit,
    )
    owned_closeables.append(vector)

    ledger: CreditLedger | None = None
    credit_meter: ExternalCreditMeter | None = None
    web: Retriever
    if web_provider == "mock":
        if mock_web_path is not None and mock_web_path.suffix.casefold() in {
            ".jsonl",
            ".ndjson",
        }:
            web = JsonlMockWebRetriever(
                mock_web_path,
                max_results=min(settings.web_max_results, 5),
            )
        else:
            fixtures = load_mock_web_fixtures(mock_web_path) if mock_web_path else {}
            web = MockWebRetriever(fixtures, max_results=min(settings.web_max_results, 5))
    else:
        if settings.tavily_api_key is None:
            raise ConfigurationError("Thiếu TAVILY_API_KEY cho Tavily live")
        raw_transport = HttpxTavilyTransport(
            settings.tavily_api_key,
            timeout_seconds=settings.request_timeout_seconds,
        )
        credit_meter = ExternalCreditMeter()
        transport = MeteredTavilyTransport(raw_transport, credit_meter)
        owned_closeables.append(transport)
        ledger = CreditLedger(
            settings.tavily_credit_budget,
            checkpoint_path=tavily_checkpoint_path,
        )
        web = TavilyRetriever(
            transport,
            cache=FileSearchCache(settings.cache_dir / "tavily"),
            limiter=RollingWindowRateLimiter(
                max_requests=settings.tavily_requests_per_minute,
                window_seconds=60,
                max_concurrency=settings.tavily_max_concurrency,
            ),
            credit_ledger=ledger,
            max_results=min(settings.web_max_results, 5),
            max_retries=settings.tavily_max_retries,
            attempt_guard=credit_meter.ensure_available,
        )

    registry = RetrieverRegistry(
        {
            RetrievalSource.VECTOR: vector,
            RetrievalSource.WEB: web,
        }
    )
    pipeline = AdaptiveRAGRouter(
        settings,
        engines,
        registry,
        active_generator,
        active_rewriter,
        policy=pipeline_policy,
    )
    return RuntimeBundle(
        pipeline=pipeline,
        settings=settings,
        engines=engines,
        generator=active_generator,
        retrievers=registry,
        tavily_ledger=ledger,
        credit_meter=credit_meter,
        _closeables=tuple(owned_closeables),
    )


class PipelineBenchmarkAdapter:
    """Chuyển ``AnswerResponse`` của pipeline sang contract của BenchmarkRunner."""

    def __init__(
        self,
        pipeline: AdaptiveRAGRouter,
        *,
        credit_meter: ExternalCreditMeter | None = None,
    ) -> None:
        """Khởi tạo adapter từ pipeline đã wiring.

        Args:
            pipeline: Pipeline cần benchmark.
            credit_meter: Meter live để tính cả retry credits cho đúng task.
        """

        self._pipeline = pipeline
        self._credit_meter = credit_meter

    async def __call__(
        self,
        sample: BenchmarkSample,
        router: RouterKind,
    ) -> PipelineRunResult:
        """Chạy một sample-router và trích metric không chứa secret.

        Args:
            sample: Query benchmark cùng reference answer tùy chọn.
            router: Engine cần dùng.

        Returns:
            Kết quả chuẩn hóa cho evaluation runner.
        """

        meter_token = self._credit_meter.start() if self._credit_meter else None
        started = perf_counter()
        try:
            response = await self._pipeline.answer(
                QueryRequest(
                    query=sample.query,
                    query_id=sample.query_id,
                    metadata=sample.metadata,
                ),
                router,
                include_trace=True,
            )
        finally:
            measured_credits = (
                self._credit_meter.finish(meter_token)
                if self._credit_meter is not None and meter_token is not None
                else 0
            )
        elapsed_ms = (perf_counter() - started) * 1_000
        trace = response.trace
        predicted_route: str | None = None
        route_probabilities: dict[str, float] = {}
        cost_usd = 0.0
        metadata: dict[str, Any] = {
            "citations": list(response.citations),
            "reference_available": sample.reference_answer is not None,
        }
        supporting_ids = sample.metadata.get(
            "supporting_document_ids",
            sample.metadata.get("supporting_fact_ids"),
        )
        if isinstance(supporting_ids, list | tuple | set):
            metadata["supporting_document_ids"] = [str(item) for item in supporting_ids]
        expected_context_quality = sample.metadata.get("expected_context_quality")
        if isinstance(expected_context_quality, str):
            metadata["expected_context_quality"] = expected_context_quality
        frozen_gold_run_id = sample.metadata.get("frozen_gold_run_id")
        if isinstance(frozen_gold_run_id, str):
            metadata["frozen_gold_run_id"] = frozen_gold_run_id
        if trace is not None:
            selected_source = trace.pre_route.retrieval_source.selected
            selected_tier = trace.pre_route.initial_model_tier.selected
            predicted_route = f"{selected_source.value}:{selected_tier.value}"
            source_probabilities = trace.pre_route.retrieval_source.probabilities
            tier_probabilities = trace.pre_route.initial_model_tier.probabilities
            route_probabilities = {
                f"{source.value}:{tier.value}": (
                    source_probabilities.get(source.value, 0.0)
                    * tier_probabilities.get(tier.value, 0.0)
                )
                for source in RetrievalSource
                for tier in ModelTier
            }
            total_probability = sum(route_probabilities.values())
            if total_probability > 0:
                route_probabilities = {
                    label: probability / total_probability
                    for label, probability in route_probabilities.items()
                }
            cost_usd = trace.total_cost_usd
            retrieved_passage_ids = list(
                dict.fromkeys(
                    document.document_id
                    for retrieval in trace.retrieval_rounds
                    for document in retrieval.documents
                )
            )
            document_lineage = {
                document.document_id: str(
                    document.metadata.get("parent_document_id", document.document_id)
                )
                for retrieval in trace.retrieval_rounds
                for document in retrieval.documents
            }
            retrieved_document_ids = list(
                dict.fromkeys(
                    document_lineage.get(passage_id, passage_id)
                    for passage_id in retrieved_passage_ids
                )
            )
            metadata["citations"] = list(
                dict.fromkeys(
                    document_lineage.get(citation, citation) for citation in response.citations
                )
            )
            total_tokens = sum(
                int(stage.metadata.get("input_tokens", 0) or 0)
                + int(stage.metadata.get("output_tokens", 0) or 0)
                for stage in trace.stages
            )
            provider_costs: dict[str, float] = {}
            for stage in trace.stages:
                provider = stage.metadata.get("provider")
                if isinstance(provider, str) and provider:
                    provider_costs[provider] = provider_costs.get(provider, 0.0) + stage.cost_usd
                if stage.stage == "ablation_policy":
                    metadata["ablation_id"] = stage.metadata.get("ablation_id")
                    metadata["pipeline_policy"] = dict(stage.metadata)
            metadata.update(
                {
                    "trace_id": trace.trace_id,
                    "degraded": trace.degraded,
                    "initial_model_tier": trace.pre_route.initial_model_tier.selected.value,
                    "retrieval_rounds": len(trace.retrieval_rounds),
                    "retrieved_passage_ids": retrieved_passage_ids,
                    "retrieved_document_ids": retrieved_document_ids,
                    "used_retrieval": bool(trace.retrieval_rounds),
                    "used_strong_fallback": (
                        trace.fallback is not None
                        and trace.fallback.action.selected is FallbackAction.REGENERATE_STRONG
                    ),
                    "total_tokens": total_tokens,
                    "provider_costs_usd": provider_costs,
                }
            )
            if trace.context_assessments:
                metadata["predicted_context_quality"] = trace.context_assessments[
                    -1
                ].quality.selected.value
                metadata["accepted_document_ids"] = list(
                    trace.context_assessments[-1].accepted_document_ids
                )
                supporting_document_ids = metadata.get("supporting_document_ids")
                if isinstance(supporting_document_ids, list) and supporting_document_ids:
                    expected = {str(item) for item in supporting_document_ids}
                    retrieved = set(retrieved_document_ids)
                    metadata["expected_context_quality"] = (
                        "sufficient"
                        if expected <= retrieved
                        else "partial"
                        if expected & retrieved
                        else "insufficient"
                    )
        quality = (
            token_f1(response.answer, sample.reference_answer)
            if sample.reference_answer is not None
            else None
        )
        return PipelineRunResult(
            answer=response.answer,
            predicted_route=predicted_route,
            route_probabilities=route_probabilities,
            quality_score=quality,
            cost_usd=cost_usd,
            latency_ms=elapsed_ms,
            status=response.status,
            external_credits=measured_credits,
            metadata=metadata,
        )


class CalibratedBenchmarkAdapter:
    """Áp dụng calibration artifact đã khóa lên probability của pipeline adapter.

    Predicted route thực tế không bị sửa sau khi chạy. Raw distribution được giữ
    trong metadata để audit, còn ``route_probabilities`` dùng distribution đã
    calibration cho ECE, Brier, NLL và risk-coverage.
    """

    def __init__(
        self,
        pipeline: PipelineBenchmarkAdapter,
        models: Mapping[RouterKind, TemperatureCalibrationModel],
    ) -> None:
        """Khởi tạo wrapper với model riêng cho từng router.

        Args:
            pipeline: Adapter pipeline gốc trả raw probability.
            models: Temperature artifact đã fit trên calibration split.
        """

        self._pipeline = pipeline
        self._calibrators = {
            router: MulticlassTemperatureCalibrator(model) for router, model in models.items()
        }

    async def __call__(
        self,
        sample: BenchmarkSample,
        router: RouterKind,
    ) -> PipelineRunResult:
        """Chạy pipeline và calibration distribution nếu có.

        Args:
            sample: Query benchmark đang đánh giá.
            router: Router tương ứng với calibration model.

        Returns:
            Kết quả giữ raw probability trong metadata và dùng calibrated output.

        Raises:
            ConfigurationError: Khi thiếu calibration model cho router.
        """

        result = await self._pipeline(sample, router)
        if not result.route_probabilities:
            return result
        calibrator = self._calibrators.get(router)
        if calibrator is None:
            raise ConfigurationError(f"Thiếu calibration model cho router {router.value!r}")
        raw_probabilities = dict(result.route_probabilities)
        calibrated = calibrator.transform([raw_probabilities])[0]
        model = calibrator.model
        return result.model_copy(
            update={
                "route_probabilities": calibrated,
                "metadata": {
                    **result.metadata,
                    "raw_route_probabilities": raw_probabilities,
                    "calibration_version": model.calibrator_version,
                    "calibration_temperature": model.temperature,
                },
            }
        )


class FileCheckpointHook:
    """Ghi checkpoint benchmark dạng JSON bằng atomic replace."""

    def __init__(self, path: Path) -> None:
        """Khởi tạo hook với file đích.

        Args:
            path: File checkpoint JSON.
        """

        self._path = path

    async def save(self, checkpoint: BenchmarkCheckpoint) -> None:
        """Lưu snapshot tiến độ mà không ghi dữ liệu provider nhạy cảm.

        Args:
            checkpoint: Trạng thái do BenchmarkRunner cung cấp.
        """

        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._path.with_suffix(self._path.suffix + ".tmp")
        temporary.write_text(
            checkpoint.model_dump_json(indent=2),
            encoding="utf-8",
            newline="\n",
        )
        temporary.replace(self._path)


class FileBenchmarkRecordStore:
    """Lưu benchmark records JSONL atomically để resume mà không gọi lại provider."""

    def __init__(self, path: Path) -> None:
        """Khởi tạo record store tại một file thuộc run directory.

        Args:
            path: File JSONL checkpoint dành riêng cho adaptive records.
        """

        self._path = path

    async def load(self) -> tuple[BenchmarkRecord, ...]:
        """Đọc records đã checkpoint; file chưa tồn tại trả tuple rỗng.

        Returns:
            Records typed theo thứ tự lần ghi gần nhất.

        Raises:
            ConfigurationError: Khi JSONL checkpoint hỏng hoặc sai schema.
        """

        if not self._path.is_file():
            return ()
        records: list[BenchmarkRecord] = []
        _line_number = 0
        try:
            for _line_number, line in enumerate(
                self._path.read_text(encoding="utf-8").splitlines(),
                1,
            ):
                if line.strip():
                    records.append(BenchmarkRecord.model_validate_json(line))
        except (OSError, ValueError) as error:
            raise ConfigurationError(
                f"Record checkpoint hỏng tại dòng {_line_number}: {self._path}"
            ) from error
        return tuple(records)

    async def save(self, records: Sequence[BenchmarkRecord]) -> None:
        """Ghi atomic toàn bộ records hiện tại sau mỗi dispatch batch.

        Args:
            records: Records đã hoàn tất hoặc thất bại có kiểm soát.
        """

        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._path.with_suffix(self._path.suffix + ".tmp")
        payload = "\n".join(record.model_dump_json() for record in records)
        temporary.write_text(
            payload + ("\n" if payload else ""),
            encoding="utf-8",
            newline="\n",
        )
        temporary.replace(self._path)


def token_f1(answer: str, reference: str) -> float:
    """Tính token-F1 nhẹ, deterministic cho reference answer có sẵn.

    Args:
        answer: Câu trả lời hệ thống.
        reference: Câu trả lời tham chiếu.

    Returns:
        Điểm F1 trong [0, 1].
    """

    answer_tokens = re.findall(r"\w+", answer.casefold())
    reference_tokens = re.findall(r"\w+", reference.casefold())
    if not answer_tokens or not reference_tokens:
        return float(answer_tokens == reference_tokens)
    answer_counts: dict[str, int] = {}
    reference_counts: dict[str, int] = {}
    for token in answer_tokens:
        answer_counts[token] = answer_counts.get(token, 0) + 1
    for token in reference_tokens:
        reference_counts[token] = reference_counts.get(token, 0) + 1
    overlap = sum(
        min(count, reference_counts.get(token, 0)) for token, count in answer_counts.items()
    )
    if overlap == 0:
        return 0.0
    precision = overlap / len(answer_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def load_mock_web_fixtures(
    path: Path,
) -> dict[str, tuple[RetrievedDocument, ...]]:
    """Nạp frozen web JSON/JSONL/Parquet thành mapping dùng bởi MockWebRetriever.

    Chấp nhận object ``query -> [documents]`` hoặc record có ``query`` và một
    trong các trường ``results``, ``documents``, ``search_results``.

    Args:
        path: File snapshot local.

    Returns:
        Mapping query sang web documents chuẩn hóa.

    Raises:
        ConfigurationError: Khi file thiếu hoặc schema không hợp lệ.
    """

    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ConfigurationError(f"Không tìm thấy frozen web fixture: {resolved}")
    try:
        raw_payload: object
        if resolved.suffix.lower() == ".json":
            raw_payload = json.loads(resolved.read_text(encoding="utf-8"))
        else:
            raw_payload = load_local_records(resolved)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise ConfigurationError(f"Không thể đọc frozen web fixture {resolved}: {error}") from error

    grouped: dict[str, list[RetrievedDocument]] = {}
    if isinstance(raw_payload, Mapping) and "records" not in raw_payload:
        for query, raw_documents in raw_payload.items():
            if not isinstance(query, str) or not _is_document_sequence(raw_documents):
                raise ConfigurationError(
                    "Frozen web mapping phải có dạng query -> danh sách document"
                )
            grouped[query] = [
                _coerce_mock_document(item, query, index)
                for index, item in enumerate(cast(Sequence[object], raw_documents))
            ]
    else:
        records = raw_payload.get("records") if isinstance(raw_payload, Mapping) else raw_payload
        if not isinstance(records, Sequence) or isinstance(records, str | bytes):
            raise ConfigurationError("Frozen web fixture phải là mapping hoặc danh sách record")
        for row_index, row in enumerate(records):
            if not isinstance(row, Mapping):
                raise ConfigurationError(f"Frozen web record {row_index} không phải object")
            query = row.get("query") or row.get("question")
            if not isinstance(query, str) or not query.strip():
                raise ConfigurationError(f"Frozen web record {row_index} thiếu query")
            raw_documents = row.get("results") or row.get("documents") or row.get("search_results")
            if raw_documents is None and any(key in row for key in ("text", "content")):
                raw_documents = [row]
            if not _is_document_sequence(raw_documents):
                raise ConfigurationError(
                    f"Frozen web record {row_index} thiếu danh sách results/documents"
                )
            grouped.setdefault(query, []).extend(
                _coerce_mock_document(item, query, index)
                for index, item in enumerate(cast(Sequence[object], raw_documents))
            )
    return {query: tuple(documents) for query, documents in grouped.items()}


def _build_deepseek_client(settings: Settings) -> AsyncOpenAI:
    if settings.deepseek_api_key is None:
        raise ConfigurationError("Thiếu DEEPSEEK_API_KEY")
    return AsyncOpenAI(
        api_key=settings.deepseek_api_key.get_secret_value(),
        base_url=settings.deepseek_base_url,
        timeout=settings.request_timeout_seconds,
    )


def _is_document_sequence(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, str | bytes)


def _coerce_mock_document(item: object, query: str, index: int) -> RetrievedDocument:
    if isinstance(item, RetrievedDocument):
        return item.model_copy(update={"source": RetrievalSource.WEB})
    if not isinstance(item, Mapping):
        raise ConfigurationError(f"Web document thứ {index} của query {query!r} không phải object")
    payload = dict(cast(Mapping[str, Any], item))
    text = (
        payload.get("text")
        or payload.get("content")
        or payload.get("snippet")
        or payload.get("page_snippet")
    )
    if not isinstance(text, str) or not text.strip():
        raise ConfigurationError(f"Web document thứ {index} của query {query!r} thiếu text")
    raw_id = payload.get("document_id") or payload.get("id")
    if raw_id is None:
        identity = f"{query}\n{payload.get('url', '')}\n{text}"
        raw_id = f"mock-{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:24]}"
    score = payload.get("provider_score", payload.get("score"))
    return RetrievedDocument(
        document_id=str(raw_id),
        text=text.strip(),
        title=str(payload.get("title") or payload.get("page_name"))
        if payload.get("title") is not None or payload.get("page_name") is not None
        else None,
        url=str(payload.get("url") or payload.get("page_url"))
        if payload.get("url") is not None or payload.get("page_url") is not None
        else None,
        source=RetrievalSource.WEB,
        provider_score=float(score) if isinstance(score, int | float) else None,
        metadata=dict(payload.get("metadata", {}))
        if isinstance(payload.get("metadata"), Mapping)
        else {},
    )
