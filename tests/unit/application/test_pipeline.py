"""Kiểm thử offline end-to-end state machine của Adaptive RAG Router bằng fakes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import StrEnum

import pytest

from adaptive_rag_router.application import AdaptiveRAGRouter, PipelinePolicy
from adaptive_rag_router.config.settings import Settings
from adaptive_rag_router.domain.enums import (
    CompletionStatus,
    Complexity,
    ContextQuality,
    FallbackAction,
    ModelTier,
    ProbabilityProvenance,
    RepairAction,
    RetrievalSource,
    RouterKind,
)
from adaptive_rag_router.domain.errors import CreditBudgetExceededError, ProviderError
from adaptive_rag_router.domain.models import (
    ContextAssessment,
    DecisionEvidence,
    DecisionUsage,
    FallbackDecision,
    GenerationResult,
    PassageAssessment,
    PreRouteDecision,
    QueryRequest,
    RetrievalRepairDecision,
    RetrievalResult,
    RetrievedDocument,
)
from adaptive_rag_router.generation import QueryRewriteResult
from adaptive_rag_router.retrieval.registry import RetrieverRegistry


class _FakeEngine:
    def __init__(
        self,
        pre_route: PreRouteDecision | BaseException,
        *,
        contexts: Sequence[ContextAssessment | BaseException] = (),
        repairs: Sequence[RetrievalRepairDecision | BaseException] = (),
        fallback: FallbackDecision | BaseException | None = None,
    ) -> None:
        self._pre_route = pre_route
        self._contexts = list(contexts)
        self._repairs = list(repairs)
        self._fallback = fallback or _fallback(FallbackAction.ACCEPT)
        self.pre_route_calls: list[QueryRequest] = []
        self.context_calls: list[RetrievalResult] = []
        self.repair_calls: list[tuple[ContextAssessment, tuple[RetrievalResult, ...]]] = []
        self.fallback_calls: list[tuple[ContextAssessment | None, GenerationResult]] = []

    async def pre_route(self, request: QueryRequest) -> PreRouteDecision:
        self.pre_route_calls.append(request)
        return _return_or_raise(self._pre_route)

    async def assess_context(
        self,
        request: QueryRequest,
        retrieval: RetrievalResult,
    ) -> ContextAssessment:
        del request
        self.context_calls.append(retrieval)
        if not self._contexts:
            raise AssertionError("assess_context được gọi ngoài kịch bản test")
        return _return_or_raise(self._contexts.pop(0))

    async def decide_repair(
        self,
        request: QueryRequest,
        context: ContextAssessment,
        retrieval_rounds: tuple[RetrievalResult, ...],
    ) -> RetrievalRepairDecision:
        del request
        self.repair_calls.append((context, retrieval_rounds))
        if not self._repairs:
            raise AssertionError("decide_repair được gọi ngoài kịch bản test")
        return _return_or_raise(self._repairs.pop(0))

    async def decide_fallback(
        self,
        request: QueryRequest,
        context: ContextAssessment | None,
        draft: GenerationResult,
    ) -> FallbackDecision:
        del request
        self.fallback_calls.append((context, draft))
        return _return_or_raise(self._fallback)


class _FakeRetriever:
    def __init__(self, outcomes: Sequence[RetrievalResult | BaseException]) -> None:
        self._outcomes = list(outcomes)
        self.calls: list[tuple[str, int]] = []

    async def retrieve(self, query: str, *, round_index: int = 0) -> RetrievalResult:
        self.calls.append((query, round_index))
        if not self._outcomes:
            raise AssertionError("retrieve được gọi ngoài kịch bản test")
        return _return_or_raise(self._outcomes.pop(0))


class _FakeGenerator:
    def __init__(self, outcomes: Sequence[GenerationResult | BaseException]) -> None:
        self._outcomes = list(outcomes)
        self.calls: list[tuple[ModelTier, tuple[RetrievedDocument, ...]]] = []

    async def generate(
        self,
        request: QueryRequest,
        documents: tuple[RetrievedDocument, ...],
        model_tier: ModelTier,
    ) -> GenerationResult:
        del request
        self.calls.append((model_tier, documents))
        if not self._outcomes:
            raise AssertionError("generate được gọi ngoài kịch bản test")
        return _return_or_raise(self._outcomes.pop(0))


class _FakeRewriter:
    def __init__(self, responses: Sequence[str] = ("rewritten query",)) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[QueryRequest, tuple[str, ...], tuple[RetrievedDocument, ...]]] = []

    async def rewrite(
        self,
        request: QueryRequest,
        missing_information: tuple[str, ...],
        documents: tuple[RetrievedDocument, ...],
    ) -> QueryRewriteResult:
        self.calls.append((request, missing_information, documents))
        if not self._responses:
            raise AssertionError("rewrite được gọi ngoài kịch bản test")
        return QueryRewriteResult(query=self._responses.pop(0))


def _return_or_raise[T](outcome: T | BaseException) -> T:
    if isinstance(outcome, BaseException):
        raise outcome
    return outcome


def _evidence[DecisionT: StrEnum](selected: DecisionT) -> DecisionEvidence[DecisionT]:
    probabilities = {item.value: 1.0 if item is selected else 0.0 for item in type(selected)}
    return DecisionEvidence(
        selected=selected,
        raw_probabilities=probabilities,
        confidence=1.0,
        provenance=ProbabilityProvenance.HEURISTIC,
        engine=RouterKind.RULE,
        model_id="fake-router-v1",
        prompt_version="fake-v1",
    )


def _pre_route(
    source: RetrievalSource,
    tier: ModelTier = ModelTier.ECONOMY,
) -> PreRouteDecision:
    return PreRouteDecision(
        retrieval_source=_evidence(source),
        complexity=_evidence(Complexity.MEDIUM),
        initial_model_tier=_evidence(tier),
    )


def _context(
    quality: ContextQuality,
    *,
    accepted: Sequence[str] = (),
    conflicts: Sequence[str] = (),
    injections: Sequence[str] = (),
    passages: Sequence[PassageAssessment] = (),
) -> ContextAssessment:
    return ContextAssessment(
        quality=_evidence(quality),
        passages=tuple(passages),
        accepted_document_ids=tuple(accepted),
        conflicting_document_ids=tuple(conflicts),
        rejected_injection_ids=tuple(injections),
    )


def _repair(
    action: RepairAction,
    missing_information: Sequence[str] = (),
) -> RetrievalRepairDecision:
    return RetrievalRepairDecision(
        action=_evidence(action),
        missing_information=tuple(missing_information),
    )


def _fallback(action: FallbackAction) -> FallbackDecision:
    return FallbackDecision(action=_evidence(action))


def _document(
    document_id: str,
    source: RetrievalSource,
    *,
    text: str | None = None,
    url: str | None = None,
    score: float = 0.8,
) -> RetrievedDocument:
    return RetrievedDocument(
        document_id=document_id,
        text=text or f"Evidence from {document_id}",
        url=url,
        source=source,
        provider_score=score,
    )


def _retrieval(
    source: RetrievalSource,
    documents: Sequence[RetrievedDocument],
    *,
    query: str = "question",
    provider: str | None = None,
) -> RetrievalResult:
    return RetrievalResult(
        source=source,
        query=query,
        documents=tuple(documents),
        provider=provider or f"fake-{source.value}",
        cached=True,
    )


def _generation(
    tier: ModelTier,
    *,
    text: str | None = None,
    cost: float = 0.01,
) -> GenerationResult:
    return GenerationResult(
        text=text or f"Answer from {tier.value}",
        model_tier=tier,
        model_id=f"fake-{tier.value}",
        cost_usd=cost,
    )


def _build_pipeline(
    engine: _FakeEngine,
    generator: _FakeGenerator,
    *,
    retrievers: Mapping[RetrievalSource, _FakeRetriever] | None = None,
    rewriter: _FakeRewriter | None = None,
    max_repair_rounds: int = 2,
    policy: PipelinePolicy | None = None,
) -> AdaptiveRAGRouter:
    settings = Settings(
        _env_file=None,
        max_repair_rounds=max_repair_rounds,
        show_trace_by_default=True,
    )
    return AdaptiveRAGRouter(
        settings=settings,
        engines={RouterKind.RULE: engine},
        retrievers=RetrieverRegistry(retrievers),
        generator=generator,
        query_rewriter=rewriter or _FakeRewriter(),
        policy=policy,
    )


@pytest.mark.asyncio
async def test_no_retrieval_generates_directly_and_preserves_trace() -> None:
    engine = _FakeEngine(_pre_route(RetrievalSource.NONE))
    generator = _FakeGenerator([_generation(ModelTier.ECONOMY, text="Direct answer [memory]")])
    pipeline = _build_pipeline(engine, generator)

    response = await pipeline.answer(
        QueryRequest(query="Explain a stable concept", query_id="no-retrieval"),
        RouterKind.RULE,
    )

    assert response.status is CompletionStatus.COMPLETED
    assert response.citations == ()
    assert generator.calls == [(ModelTier.ECONOMY, ())]
    assert engine.context_calls == []
    assert response.trace is not None
    assert response.trace.retrieval_rounds == ()
    assert [stage.stage for stage in response.trace.stages] == [
        "pre_route",
        "generation_initial",
        "fallback_decision",
    ]


@pytest.mark.asyncio
async def test_context_gate_ablation_accepts_initial_evidence_without_router_assessment() -> None:
    """No-context-gate ablation bỏ assessment/repair và dùng evidence ban đầu."""

    document = _document("doc-1", RetrievalSource.VECTOR)
    engine = _FakeEngine(_pre_route(RetrievalSource.VECTOR))
    retriever = _FakeRetriever([_retrieval(RetrievalSource.VECTOR, [document])])
    generator = _FakeGenerator([_generation(ModelTier.ECONOMY)])
    pipeline = _build_pipeline(
        engine,
        generator,
        retrievers={RetrievalSource.VECTOR: retriever},
        policy=PipelinePolicy(context_gate_enabled=False),
    )

    response = await pipeline.answer(QueryRequest(query="question"), RouterKind.RULE)

    assert response.status is CompletionStatus.COMPLETED
    assert engine.context_calls == []
    assert engine.repair_calls == []
    assert generator.calls[0][1] == (document,)
    assert response.trace is not None
    assert response.trace.stages[0].stage == "ablation_policy"
    assessment = next(
        stage for stage in response.trace.stages if stage.stage == "context_assessment_0"
    )
    assert assessment.metadata["ablated"] is True


@pytest.mark.asyncio
async def test_repair_ablation_stops_after_initial_context_assessment() -> None:
    """No-repair ablation không gọi repair hoặc query rewriter."""

    document = _document("doc-1", RetrievalSource.VECTOR)
    engine = _FakeEngine(
        _pre_route(RetrievalSource.VECTOR),
        contexts=[_context(ContextQuality.PARTIAL, accepted=[document.document_id])],
    )
    retriever = _FakeRetriever([_retrieval(RetrievalSource.VECTOR, [document])])
    generator = _FakeGenerator([_generation(ModelTier.ECONOMY)])
    rewriter = _FakeRewriter()
    pipeline = _build_pipeline(
        engine,
        generator,
        retrievers={RetrievalSource.VECTOR: retriever},
        rewriter=rewriter,
        policy=PipelinePolicy(repair_enabled=False),
    )

    response = await pipeline.answer(QueryRequest(query="question"), RouterKind.RULE)

    assert response.status is CompletionStatus.COMPLETED
    assert len(engine.context_calls) == 1
    assert engine.repair_calls == []
    assert rewriter.calls == []
    assert response.trace is not None
    repair_stage = next(stage for stage in response.trace.stages if stage.stage == "repair_0")
    assert repair_stage.metadata["ablated"] is True


@pytest.mark.asyncio
async def test_strong_fallback_ablation_retains_economy_draft() -> None:
    """No-strong-fallback ablation ghi gate decision nhưng không gọi strong model."""

    engine = _FakeEngine(
        _pre_route(RetrievalSource.NONE),
        fallback=_fallback(FallbackAction.REGENERATE_STRONG),
    )
    generator = _FakeGenerator([_generation(ModelTier.ECONOMY, text="economy draft")])
    pipeline = _build_pipeline(
        engine,
        generator,
        policy=PipelinePolicy(strong_fallback_enabled=False),
    )

    response = await pipeline.answer(QueryRequest(query="question"), RouterKind.RULE)

    assert response.answer == "economy draft"
    assert len(generator.calls) == 1
    assert response.trace is not None
    fallback_stage = next(
        stage for stage in response.trace.stages if stage.stage == "generation_fallback"
    )
    assert fallback_stage.metadata["ablated"] is True


@pytest.mark.parametrize(
    ("source", "provider"),
    [
        (RetrievalSource.VECTOR, "fake-vector"),
        (RetrievalSource.WEB, "mock-web"),
    ],
)
@pytest.mark.asyncio
async def test_vector_and_mock_web_retrieval_feed_grounded_generation(
    source: RetrievalSource,
    provider: str,
) -> None:
    document = _document(f"{source.value}-1", source)
    retriever = _FakeRetriever([_retrieval(source, [document], provider=provider)])
    engine = _FakeEngine(
        _pre_route(source),
        contexts=[_context(ContextQuality.SUFFICIENT, accepted=[document.document_id])],
    )
    generator = _FakeGenerator([_generation(ModelTier.ECONOMY)])
    pipeline = _build_pipeline(engine, generator, retrievers={source: retriever})

    response = await pipeline.answer(QueryRequest(query="question"), RouterKind.RULE)

    assert response.status is CompletionStatus.COMPLETED
    assert retriever.calls == [("question", 0)]
    assert [item.document_id for item in generator.calls[0][1]] == [document.document_id]
    assert response.trace is not None
    assert response.trace.retrieval_rounds[0].provider == provider
    assert response.trace.context_assessments[0].quality.selected is ContextQuality.SUFFICIENT


@pytest.mark.asyncio
async def test_repair_switches_source_and_succeeds_with_rewritten_query() -> None:
    vector_document = _document("vector-1", RetrievalSource.VECTOR)
    web_document = _document("web-1", RetrievalSource.WEB)
    vector = _FakeRetriever(
        [_retrieval(RetrievalSource.VECTOR, [vector_document], query="question")]
    )
    web = _FakeRetriever(
        [_retrieval(RetrievalSource.WEB, [web_document], query="focused web query")]
    )
    engine = _FakeEngine(
        _pre_route(RetrievalSource.VECTOR),
        contexts=[
            _context(ContextQuality.PARTIAL, accepted=["vector-1"]),
            _context(ContextQuality.SUFFICIENT, accepted=["vector-1", "web-1"]),
        ],
        repairs=[_repair(RepairAction.SWITCH_WEB, ["current facts"])],
    )
    rewriter = _FakeRewriter(["focused web query"])
    generator = _FakeGenerator([_generation(ModelTier.ECONOMY)])
    pipeline = _build_pipeline(
        engine,
        generator,
        retrievers={RetrievalSource.VECTOR: vector, RetrievalSource.WEB: web},
        rewriter=rewriter,
    )

    response = await pipeline.answer(QueryRequest(query="question"), RouterKind.RULE)

    assert response.status is CompletionStatus.COMPLETED
    assert vector.calls == [("question", 0)]
    assert web.calls == [("focused web query", 1)]
    assert len(rewriter.calls) == 1
    assert rewriter.calls[0][1] == ("current facts",)
    assert [document.document_id for document in generator.calls[0][1]] == [
        "vector-1",
        "web-1",
    ]
    assert response.trace is not None
    assert len(response.trace.retrieval_rounds) == 2
    assert response.trace.repair_decisions[0].action.selected is RepairAction.SWITCH_WEB


@pytest.mark.asyncio
async def test_repair_stops_after_configured_round_budget() -> None:
    vector = _FakeRetriever(
        [
            _retrieval(
                RetrievalSource.VECTOR,
                [_document("v1", RetrievalSource.VECTOR)],
            ),
            _retrieval(
                RetrievalSource.VECTOR,
                [_document("v2", RetrievalSource.VECTOR)],
            ),
        ]
    )
    web = _FakeRetriever([_retrieval(RetrievalSource.WEB, [_document("w1", RetrievalSource.WEB)])])
    engine = _FakeEngine(
        _pre_route(RetrievalSource.VECTOR),
        contexts=[
            _context(ContextQuality.PARTIAL, accepted=["v1"]),
            _context(ContextQuality.PARTIAL, accepted=["v1", "v2"]),
            _context(ContextQuality.INSUFFICIENT),
        ],
        repairs=[
            _repair(RepairAction.REWRITE_VECTOR, ["detail one"]),
            _repair(RepairAction.SWITCH_WEB, ["detail two"]),
        ],
        fallback=_fallback(FallbackAction.ABSTAIN),
    )
    rewriter = _FakeRewriter(["vector rewrite", "web rewrite"])
    generator = _FakeGenerator([_generation(ModelTier.ECONOMY, text="Unsupported draft")])
    pipeline = _build_pipeline(
        engine,
        generator,
        retrievers={RetrievalSource.VECTOR: vector, RetrievalSource.WEB: web},
        rewriter=rewriter,
        max_repair_rounds=2,
    )

    response = await pipeline.answer(QueryRequest(query="question"), RouterKind.RULE)

    assert response.status is CompletionStatus.ABSTAINED
    assert vector.calls == [("question", 0), ("vector rewrite", 1)]
    assert web.calls == [("web rewrite", 2)]
    assert len(engine.context_calls) == 3
    assert len(engine.repair_calls) == 2
    assert len(rewriter.calls) == 2
    assert response.trace is not None
    assert len(response.trace.retrieval_rounds) == 3


@pytest.mark.asyncio
async def test_duplicate_evidence_is_merged_across_repair_rounds() -> None:
    original = _document(
        "original",
        RetrievalSource.VECTOR,
        text="Same supporting fact",
        url="https://example.test/fact?utm_source=first",
        score=0.4,
    )
    duplicate = _document(
        "duplicate",
        RetrievalSource.WEB,
        text="Same supporting fact",
        url="https://example.test/fact",
        score=0.9,
    )
    vector = _FakeRetriever([_retrieval(RetrievalSource.VECTOR, [original])])
    web = _FakeRetriever([_retrieval(RetrievalSource.WEB, [duplicate])])
    engine = _FakeEngine(
        _pre_route(RetrievalSource.VECTOR),
        contexts=[
            _context(ContextQuality.PARTIAL, accepted=["original"]),
            _context(ContextQuality.SUFFICIENT, accepted=["duplicate"]),
        ],
        repairs=[_repair(RepairAction.SWITCH_WEB)],
    )
    generator = _FakeGenerator([_generation(ModelTier.ECONOMY)])
    pipeline = _build_pipeline(
        engine,
        generator,
        retrievers={RetrievalSource.VECTOR: vector, RetrievalSource.WEB: web},
    )

    await pipeline.answer(QueryRequest(query="question"), RouterKind.RULE)

    final_context = engine.context_calls[-1]
    assert len(final_context.documents) == 1
    assert final_context.documents[0].document_id == "duplicate"
    assert [document.document_id for document in generator.calls[0][1]] == ["duplicate"]


@pytest.mark.asyncio
async def test_context_filter_marks_conflicts_and_excludes_prompt_injection() -> None:
    safe = _document("safe", RetrievalSource.WEB)
    conflict = _document("conflict", RetrievalSource.WEB)
    injected = _document("injected", RetrievalSource.WEB)
    retriever = _FakeRetriever([_retrieval(RetrievalSource.WEB, [safe, conflict, injected])])
    engine = _FakeEngine(
        _pre_route(RetrievalSource.WEB),
        contexts=[
            _context(
                ContextQuality.SUFFICIENT,
                accepted=["safe", "conflict", "injected"],
                conflicts=["conflict"],
                injections=["injected"],
            )
        ],
    )
    generator = _FakeGenerator([_generation(ModelTier.ECONOMY)])
    pipeline = _build_pipeline(
        engine,
        generator,
        retrievers={RetrievalSource.WEB: retriever},
    )

    await pipeline.answer(QueryRequest(query="question"), RouterKind.RULE)

    selected = generator.calls[0][1]
    assert [document.document_id for document in selected] == ["safe", "conflict"]
    assert selected[1].metadata["_context_conflict"] is True


@pytest.mark.asyncio
async def test_router_timeout_uses_deterministic_safe_pre_route_policy() -> None:
    document = _document("safe-vector", RetrievalSource.VECTOR)
    retriever = _FakeRetriever([_retrieval(RetrievalSource.VECTOR, [document])])
    engine = _FakeEngine(
        TimeoutError("router deadline"),
        contexts=[_context(ContextQuality.SUFFICIENT, accepted=["safe-vector"])],
    )
    generator = _FakeGenerator([_generation(ModelTier.STRONG)])
    pipeline = _build_pipeline(
        engine,
        generator,
        retrievers={RetrievalSource.VECTOR: retriever},
    )

    response = await pipeline.answer(
        QueryRequest(query="Explain a stable technical concept"),
        RouterKind.RULE,
    )

    assert response.status is CompletionStatus.DEGRADED
    assert retriever.calls == [("Explain a stable technical concept", 0)]
    assert generator.calls[0][0] is ModelTier.STRONG
    assert response.trace is not None
    assert response.trace.degraded is True
    assert response.trace.pre_route.retrieval_source.selected is RetrievalSource.VECTOR
    assert response.trace.stages[0].error == "TimeoutError: safe policy applied"


@pytest.mark.asyncio
async def test_retrieval_provider_error_switches_to_alternate_source() -> None:
    web_document = _document("web-after-error", RetrievalSource.WEB)
    vector = _FakeRetriever([ProviderError("vector unavailable")])
    web = _FakeRetriever([_retrieval(RetrievalSource.WEB, [web_document])])
    engine = _FakeEngine(
        _pre_route(RetrievalSource.VECTOR),
        contexts=[_context(ContextQuality.SUFFICIENT, accepted=["web-after-error"])],
    )
    generator = _FakeGenerator([_generation(ModelTier.ECONOMY)])
    pipeline = _build_pipeline(
        engine,
        generator,
        retrievers={RetrievalSource.VECTOR: vector, RetrievalSource.WEB: web},
    )

    response = await pipeline.answer(QueryRequest(query="question"), RouterKind.RULE)

    assert response.status is CompletionStatus.DEGRADED
    assert vector.calls == [("question", 0)]
    assert web.calls == [("question", 1)]
    assert [document.document_id for document in generator.calls[0][1]] == ["web-after-error"]
    assert response.trace is not None
    assert response.trace.stages[1].stage == "retrieval_0"
    assert response.trace.stages[1].error == "ProviderError"


@pytest.mark.asyncio
async def test_economy_provider_failure_retries_once_with_strong_model() -> None:
    engine = _FakeEngine(_pre_route(RetrievalSource.NONE))
    generator = _FakeGenerator(
        [
            ProviderError("economy unavailable"),
            _generation(ModelTier.STRONG, text="Recovered answer", cost=0.04),
        ]
    )
    pipeline = _build_pipeline(engine, generator)

    response = await pipeline.answer(QueryRequest(query="question"), RouterKind.RULE)

    assert response.answer == "Recovered answer"
    assert response.status is CompletionStatus.DEGRADED
    assert [tier for tier, _ in generator.calls] == [ModelTier.ECONOMY, ModelTier.STRONG]
    assert engine.fallback_calls[0][1].model_tier is ModelTier.STRONG
    assert response.trace is not None
    assert response.trace.total_cost_usd == pytest.approx(0.04)


@pytest.mark.asyncio
async def test_strong_initial_generation_failure_abstains_with_trace() -> None:
    """Strong provider failure không được biến thành unhandled benchmark error."""

    engine = _FakeEngine(_pre_route(RetrievalSource.NONE, tier=ModelTier.STRONG))
    pipeline = _build_pipeline(
        engine,
        _FakeGenerator([ProviderError("strong unavailable")]),
    )

    response = await pipeline.answer(QueryRequest(query="question"), RouterKind.RULE)

    assert response.status is CompletionStatus.ABSTAINED
    assert "provider failed" in response.answer
    assert response.trace is not None
    assert response.trace.degraded is True
    assert response.trace.stages[-1].stage == "generation_initial_error"
    assert response.trace.stages[-1].error == "ProviderError"


@pytest.mark.asyncio
async def test_strong_fallback_generation_failure_abstains_with_trace() -> None:
    """Fallback mạnh thất bại phải abstain thay vì trả weak draft hoặc ném lỗi."""

    engine = _FakeEngine(
        _pre_route(RetrievalSource.NONE),
        fallback=_fallback(FallbackAction.REGENERATE_STRONG),
    )
    pipeline = _build_pipeline(
        engine,
        _FakeGenerator(
            [
                _generation(ModelTier.ECONOMY, text="Weak answer"),
                ProviderError("strong unavailable"),
            ]
        ),
    )

    response = await pipeline.answer(QueryRequest(query="question"), RouterKind.RULE)

    assert response.status is CompletionStatus.ABSTAINED
    assert response.answer != "Weak answer"
    assert response.trace is not None
    assert response.trace.fallback is not None
    assert response.trace.stages[-1].stage == "generation_fallback"
    assert response.trace.stages[-1].error == "ProviderError"


@pytest.mark.asyncio
async def test_fallback_gate_regenerates_economy_draft_with_strong_model() -> None:
    engine = _FakeEngine(
        _pre_route(RetrievalSource.NONE),
        fallback=_fallback(FallbackAction.REGENERATE_STRONG),
    )
    generator = _FakeGenerator(
        [
            _generation(ModelTier.ECONOMY, text="Weak answer", cost=0.01),
            _generation(ModelTier.STRONG, text="Strong answer [source]", cost=0.05),
        ]
    )
    pipeline = _build_pipeline(engine, generator)

    response = await pipeline.answer(QueryRequest(query="question"), RouterKind.RULE)

    assert response.answer == "Strong answer [source]"
    assert response.citations == ()
    assert response.status is CompletionStatus.COMPLETED
    assert [tier for tier, _ in generator.calls] == [ModelTier.ECONOMY, ModelTier.STRONG]
    assert response.trace is not None
    assert response.trace.total_cost_usd == pytest.approx(0.06)
    assert [stage.stage for stage in response.trace.stages][-1] == "generation_fallback"


@pytest.mark.asyncio
async def test_repair_abstention_returns_without_calling_generator() -> None:
    document = _document("partial", RetrievalSource.VECTOR)
    retriever = _FakeRetriever([_retrieval(RetrievalSource.VECTOR, [document])])
    engine = _FakeEngine(
        _pre_route(RetrievalSource.VECTOR),
        contexts=[_context(ContextQuality.INSUFFICIENT, accepted=["partial"])],
        repairs=[_repair(RepairAction.ABSTAIN, ["supporting fact"])],
    )
    generator = _FakeGenerator([])
    pipeline = _build_pipeline(
        engine,
        generator,
        retrievers={RetrievalSource.VECTOR: retriever},
    )

    response = await pipeline.answer(QueryRequest(query="question"), RouterKind.RULE)

    assert response.status is CompletionStatus.ABSTAINED
    assert "sufficient evidence" in response.answer
    assert generator.calls == []
    assert engine.fallback_calls == []
    assert response.trace is not None
    assert response.trace.fallback is None
    assert response.trace.repair_decisions[0].action.selected is RepairAction.ABSTAIN


@pytest.mark.asyncio
async def test_fallback_abstention_does_not_expose_unsupported_draft() -> None:
    engine = _FakeEngine(
        _pre_route(RetrievalSource.NONE),
        fallback=_fallback(FallbackAction.ABSTAIN),
    )
    generator = _FakeGenerator(
        [_generation(ModelTier.ECONOMY, text="Unsupported hallucinated draft")]
    )
    pipeline = _build_pipeline(engine, generator)

    response = await pipeline.answer(QueryRequest(query="question"), RouterKind.RULE)

    assert response.status is CompletionStatus.ABSTAINED
    assert response.answer != "Unsupported hallucinated draft"
    assert "reliably" in response.answer.lower()
    assert response.trace is not None
    assert response.trace.fallback is not None
    assert response.trace.fallback.action.selected is FallbackAction.ABSTAIN


@pytest.mark.asyncio
async def test_credit_budget_exhaustion_propagates_to_benchmark_runner() -> None:
    """Credit hard cap phải dừng live run thay vì bị che thành retrieval fallback."""

    engine = _FakeEngine(_pre_route(RetrievalSource.VECTOR))
    generator = _FakeGenerator([])
    retriever = _FakeRetriever([CreditBudgetExceededError("budget exhausted")])
    pipeline = _build_pipeline(
        engine,
        generator,
        retrievers={RetrievalSource.VECTOR: retriever},
    )

    with pytest.raises(CreditBudgetExceededError, match="budget exhausted"):
        await pipeline.answer(QueryRequest(query="Question"), RouterKind.RULE)

    assert generator.calls == []


@pytest.mark.asyncio
async def test_decision_provider_cost_is_included_once_in_total_cost() -> None:
    """Một call pre-route đã batch không bị bỏ sót hoặc cộng lặp theo số decision."""

    pre_route = _pre_route(RetrievalSource.NONE).model_copy(
        update={
            "usage": DecisionUsage(
                provider="typesafe-jev",
                model_id="jev-1.13.0",
                input_tokens=1_000,
                cost_usd=0.000042,
            )
        }
    )
    pipeline = _build_pipeline(
        _FakeEngine(pre_route),
        _FakeGenerator([_generation(ModelTier.ECONOMY, cost=0.01)]),
    )

    response = await pipeline.answer(QueryRequest(query="Question"), RouterKind.RULE)

    assert response.trace is not None
    assert response.trace.total_cost_usd == pytest.approx(0.010042)
    assert response.trace.stages[0].metadata["input_tokens"] == 1_000


@pytest.mark.asyncio
async def test_repair_evidence_can_enter_full_top_eight_context() -> None:
    """Evidence mới điểm cao không bị cắt chỉ vì vòng đầu đã đủ tám passage."""

    initial_documents = [
        _document(
            f"old-{index}",
            RetrievalSource.VECTOR,
            score=0.10 + index / 100,
        )
        for index in range(8)
    ]
    repaired = _document("repair-best", RetrievalSource.WEB, score=0.99)
    vector = _FakeRetriever([_retrieval(RetrievalSource.VECTOR, initial_documents)])
    web = _FakeRetriever([_retrieval(RetrievalSource.WEB, [repaired])])
    engine = _FakeEngine(
        _pre_route(RetrievalSource.VECTOR),
        contexts=[
            _context(
                ContextQuality.PARTIAL,
                accepted=[document.document_id for document in initial_documents],
            ),
            _context(ContextQuality.SUFFICIENT, accepted=["repair-best"]),
        ],
        repairs=[_repair(RepairAction.SWITCH_WEB)],
    )
    generator = _FakeGenerator([_generation(ModelTier.ECONOMY)])
    pipeline = _build_pipeline(
        engine,
        generator,
        retrievers={RetrievalSource.VECTOR: vector, RetrievalSource.WEB: web},
    )

    await pipeline.answer(QueryRequest(query="Question"), RouterKind.RULE)

    second_context_ids = {document.document_id for document in engine.context_calls[1].documents}
    assert "repair-best" in second_context_ids
    assert len(second_context_ids) == 8
