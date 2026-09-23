"""Điều phối routing, retrieval, generation và fallback theo một state machine hữu hạn."""

from __future__ import annotations

import re
from collections.abc import Mapping
from time import perf_counter

from adaptive_rag_router.application.policy import PipelinePolicy
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
    AnswerResponse,
    ContextAssessment,
    DecisionEvidence,
    DecisionTrace,
    DecisionUsage,
    FallbackDecision,
    GenerationResult,
    PreRouteDecision,
    QueryRequest,
    RetrievalRepairDecision,
    RetrievalResult,
    RetrievedDocument,
    StageTrace,
)
from adaptive_rag_router.generation.deepseek import DeepSeekGenerator
from adaptive_rag_router.generation.ports import Generator
from adaptive_rag_router.generation.rewriter import QueryRewriter
from adaptive_rag_router.retrieval.base import deduplicate_documents
from adaptive_rag_router.retrieval.registry import RetrieverRegistry
from adaptive_rag_router.routing.ports import DecisionEngine


class AdaptiveRAGRouter:
    """Pipeline công khai xử lý một query bằng decision engine được chọn.

    Pipeline có số vòng hữu hạn: tối đa một retrieval ban đầu, một số repair
    rounds đã cấu hình và tối đa một strong-model regeneration.
    """

    def __init__(
        self,
        settings: Settings,
        engines: Mapping[RouterKind, DecisionEngine],
        retrievers: RetrieverRegistry,
        generator: Generator,
        query_rewriter: QueryRewriter,
        policy: PipelinePolicy | None = None,
    ) -> None:
        """Khởi tạo pipeline từ các port đã được inject.

        Args:
            settings: Giới hạn retrieval và runtime.
            engines: Decision engine theo router kind.
            retrievers: Registry của vector/web retriever.
            generator: Answer generator dùng chung giữa các router.
            query_rewriter: Rewriter dùng chung để bảo đảm benchmark công bằng.
            policy: Typed switches cho baseline hoặc component ablation.
        """

        self._settings = settings
        self._engines = dict(engines)
        self._retrievers = retrievers
        self._generator = generator
        self._query_rewriter = query_rewriter
        self._policy = policy or PipelinePolicy()

    async def answer(
        self,
        request: QueryRequest,
        engine: RouterKind,
        *,
        include_trace: bool | None = None,
    ) -> AnswerResponse:
        """Chạy toàn bộ Adaptive RAG pipeline cho một query.

        Args:
            request: Query và metadata được router quan sát.
            engine: Jev, rule hoặc LLM router.
            include_trace: Có đính kèm audit trace trong response hay không.

        Returns:
            Answer, citations, trạng thái và optional trace.
        """

        if engine not in self._engines:
            raise ValueError(f"Decision engine chưa được đăng ký: {engine}")
        decision_engine = self._engines[engine]
        degraded = False
        stages: list[StageTrace] = []
        if not self._policy.is_baseline:
            stages.append(
                StageTrace(
                    stage="ablation_policy",
                    metadata={
                        "ablation_id": self._policy.ablation_id,
                        **self._policy.model_dump(),
                    },
                )
            )

        pre_started = perf_counter()
        try:
            pre_route = await decision_engine.pre_route(request)
            stages.append(
                StageTrace(
                    stage="pre_route",
                    latency_ms=self._elapsed_ms(pre_started),
                    cost_usd=self._usage_cost(pre_route.usage),
                    metadata=self._usage_metadata(pre_route.usage),
                )
            )
        except Exception as exc:
            degraded = True
            pre_route = self._safe_pre_route(request, engine)
            stages.append(
                StageTrace(
                    stage="pre_route",
                    latency_ms=self._elapsed_ms(pre_started),
                    error=f"{type(exc).__name__}: safe policy applied",
                )
            )

        retrieval_rounds: list[RetrievalResult] = []
        context_assessments: list[ContextAssessment] = []
        repair_decisions: list[RetrievalRepairDecision] = []
        candidate_documents: tuple[RetrievedDocument, ...] = ()
        all_documents: tuple[RetrievedDocument, ...] = ()
        context: ContextAssessment | None = None
        abstain_requested = False

        source = pre_route.retrieval_source.selected
        current_query = request.query
        if source is not RetrievalSource.NONE:
            for round_index in range(self._settings.max_repair_rounds + 1):
                retrieval_started = perf_counter()
                try:
                    result = await self._retrievers.get(source).retrieve(
                        current_query, round_index=round_index
                    )
                except CreditBudgetExceededError:
                    # Live benchmark runner cần thấy tín hiệu này để dừng, ghi
                    # checkpoint và giữ nguyên các record đã hoàn tất.
                    raise
                except Exception as exc:
                    degraded = True
                    stages.append(
                        StageTrace(
                            stage=f"retrieval_{round_index}",
                            latency_ms=self._elapsed_ms(retrieval_started),
                            error=type(exc).__name__,
                        )
                    )
                    alternate = self._alternate_source(source)
                    if round_index < self._settings.max_repair_rounds:
                        source = alternate
                        continue
                    break
                retrieval_rounds.append(result)
                previous_candidate_count = len(candidate_documents)
                candidate_documents = self._merge_documents(
                    candidate_documents,
                    result.documents,
                    limit=None,
                )
                new_document_count = len(candidate_documents) - previous_candidate_count
                all_documents = self._rank_documents(
                    candidate_documents,
                    self._settings.evidence_limit,
                )
                stages.append(
                    StageTrace(
                        stage=f"retrieval_{round_index}",
                        latency_ms=self._elapsed_ms(retrieval_started),
                        metadata={
                            "source": source.value,
                            "document_count": len(result.documents),
                            "new_document_count": new_document_count,
                            "cached": result.cached,
                        },
                    )
                )

                combined = RetrievalResult(
                    source=source,
                    query=current_query,
                    documents=all_documents,
                    latency_ms=sum(item.latency_ms for item in retrieval_rounds),
                    cached=all(item.cached for item in retrieval_rounds),
                    provider="merged",
                    round_index=round_index,
                    errors=tuple(error for item in retrieval_rounds for error in item.errors),
                )
                assessment_started = perf_counter()
                if not self._policy.context_gate_enabled:
                    context = self._ablated_context_assessment(engine, all_documents)
                    context_assessments.append(context)
                    stages.append(
                        StageTrace(
                            stage=f"context_assessment_{round_index}",
                            latency_ms=self._elapsed_ms(assessment_started),
                            metadata={"ablated": True, "accepted_all_evidence": True},
                        )
                    )
                else:
                    try:
                        context = await decision_engine.assess_context(request, combined)
                        context_assessments.append(context)
                        stages.append(
                            StageTrace(
                                stage=f"context_assessment_{round_index}",
                                latency_ms=self._elapsed_ms(assessment_started),
                                cost_usd=self._usage_cost(context.usage),
                                metadata=self._usage_metadata(context.usage),
                            )
                        )
                    except Exception as exc:
                        degraded = True
                        context = self._safe_context_assessment(engine, all_documents)
                        context_assessments.append(context)
                        stages.append(
                            StageTrace(
                                stage=f"context_assessment_{round_index}",
                                latency_ms=self._elapsed_ms(assessment_started),
                                error=f"{type(exc).__name__}: safe assessment applied",
                            )
                        )

                if context.quality.selected is ContextQuality.SUFFICIENT:
                    break
                if not self._policy.repair_enabled:
                    stages.append(
                        StageTrace(
                            stage=f"repair_{round_index}",
                            metadata={"ablated": True, "reason": "repair_disabled"},
                        )
                    )
                    break
                if round_index >= self._settings.max_repair_rounds:
                    break
                # Cho phép repair sau lần retrieval đầu tiên ngay cả khi nguồn đầu
                # không trả passage. Từ vòng repair trở đi, không có evidence mới
                # là điều kiện dừng để tránh lặp vô ích.
                if round_index > 0 and new_document_count == 0:
                    break

                repair_started = perf_counter()
                try:
                    repair = await decision_engine.decide_repair(
                        request, context, tuple(retrieval_rounds)
                    )
                except Exception as exc:
                    degraded = True
                    repair = self._safe_repair(engine, source)
                    stages.append(
                        StageTrace(
                            stage=f"repair_{round_index}",
                            latency_ms=self._elapsed_ms(repair_started),
                            error=f"{type(exc).__name__}: safe repair applied",
                        )
                    )
                else:
                    stages.append(
                        StageTrace(
                            stage=f"repair_{round_index}",
                            latency_ms=self._elapsed_ms(repair_started),
                            cost_usd=self._usage_cost(repair.usage),
                            metadata=self._usage_metadata(repair.usage),
                        )
                    )
                repair_decisions.append(repair)
                action = repair.action.selected
                if action is RepairAction.STOP:
                    break
                if action is RepairAction.ABSTAIN:
                    abstain_requested = True
                    break
                next_source = (
                    RetrievalSource.VECTOR
                    if action is RepairAction.REWRITE_VECTOR
                    else RetrievalSource.WEB
                )
                rewrite_started = perf_counter()
                try:
                    rewrite = await self._query_rewriter.rewrite(
                        request,
                        repair.missing_information,
                        all_documents,
                    )
                    current_query = rewrite.query
                except Exception as exc:
                    degraded = True
                    stages.append(
                        StageTrace(
                            stage=f"query_rewrite_{round_index}",
                            latency_ms=self._elapsed_ms(rewrite_started),
                            error=f"{type(exc).__name__}: original query retained",
                        )
                    )
                    # Khi vẫn ở cùng nguồn, lặp lại nguyên query chỉ tốn thêm
                    # budget. Nếu đổi nguồn, query gốc vẫn là fallback hữu ích.
                    if next_source is source:
                        break
                    current_query = request.query
                else:
                    stages.append(
                        StageTrace(
                            stage=f"query_rewrite_{round_index}",
                            latency_ms=self._elapsed_ms(rewrite_started),
                            cost_usd=rewrite.cost_usd,
                            metadata={
                                "provider": "deepseek",
                                "input_tokens": rewrite.usage.input_tokens,
                                "output_tokens": rewrite.usage.output_tokens,
                                "cached_input_tokens": rewrite.usage.cached_input_tokens,
                            },
                        )
                    )
                source = next_source

        selected_documents = self._select_documents(all_documents, context)
        retrieval_was_required = pre_route.retrieval_source.selected is not RetrievalSource.NONE
        if abstain_requested or (retrieval_was_required and not selected_documents):
            stages.append(
                StageTrace(
                    stage="evidence_gate",
                    metadata={
                        "reason": ("repair_abstain" if abstain_requested else "no_usable_evidence")
                    },
                )
            )
            trace = self._build_trace(
                engine,
                pre_route,
                retrieval_rounds,
                context_assessments,
                repair_decisions,
                None,
                stages,
                sum(stage.cost_usd for stage in stages),
                degraded,
            )
            return AnswerResponse(
                query_id=request.query_id,
                answer="Unable to answer reliably because sufficient evidence was not found.",
                status=CompletionStatus.ABSTAINED,
                trace=trace if self._should_include_trace(include_trace) else None,
            )

        initial_tier = pre_route.initial_model_tier.selected
        generation_started = perf_counter()
        try:
            draft = await self._generator.generate(request, selected_documents, initial_tier)
        except ProviderError as error:
            degraded = True
            stages.append(
                StageTrace(
                    stage="generation_initial_error",
                    latency_ms=self._elapsed_ms(generation_started),
                    error=type(error).__name__,
                )
            )
            if initial_tier is ModelTier.STRONG:
                return self._generation_failure_response(
                    request,
                    engine,
                    pre_route,
                    retrieval_rounds,
                    context_assessments,
                    repair_decisions,
                    None,
                    stages,
                    include_trace=include_trace,
                )
            recovery_started = perf_counter()
            try:
                draft = await self._generator.generate(
                    request,
                    selected_documents,
                    ModelTier.STRONG,
                )
            except ProviderError as recovery_error:
                stages.append(
                    StageTrace(
                        stage="generation_recovery_error",
                        latency_ms=self._elapsed_ms(recovery_started),
                        error=type(recovery_error).__name__,
                    )
                )
                return self._generation_failure_response(
                    request,
                    engine,
                    pre_route,
                    retrieval_rounds,
                    context_assessments,
                    repair_decisions,
                    None,
                    stages,
                    include_trace=include_trace,
                )
        stages.append(
            StageTrace(
                stage="generation_initial",
                latency_ms=self._elapsed_ms(generation_started),
                cost_usd=draft.cost_usd,
                metadata={
                    "provider": "deepseek",
                    "model_id": draft.model_id,
                    "tier": draft.model_tier.value,
                    "input_tokens": draft.usage.input_tokens,
                    "output_tokens": draft.usage.output_tokens,
                    "cached_input_tokens": draft.usage.cached_input_tokens,
                },
            )
        )

        fallback_started = perf_counter()
        fallback, fallback_degraded = await self._decide_fallback_safely(
            decision_engine, engine, request, context, draft
        )
        degraded = degraded or fallback_degraded
        stages.append(
            StageTrace(
                stage="fallback_decision",
                latency_ms=self._elapsed_ms(fallback_started),
                cost_usd=self._usage_cost(fallback.usage),
                metadata=self._usage_metadata(fallback.usage),
                error="safe fallback applied" if fallback_degraded else None,
            )
        )
        final = draft
        status = CompletionStatus.DEGRADED if degraded else CompletionStatus.COMPLETED
        if fallback.action.selected is FallbackAction.ABSTAIN:
            status = CompletionStatus.ABSTAINED
        elif (
            fallback.action.selected is FallbackAction.REGENERATE_STRONG
            and draft.model_tier is ModelTier.ECONOMY
        ):
            if not self._policy.strong_fallback_enabled:
                stages.append(
                    StageTrace(
                        stage="generation_fallback",
                        metadata={
                            "ablated": True,
                            "reason": "strong_fallback_disabled",
                            "retained_tier": draft.model_tier.value,
                        },
                    )
                )
            else:
                strong_result = await self._regenerate_strong(
                    request,
                    engine,
                    pre_route,
                    retrieval_rounds,
                    context_assessments,
                    repair_decisions,
                    fallback,
                    selected_documents,
                    stages,
                    include_trace=include_trace,
                )
                if isinstance(strong_result, AnswerResponse):
                    return strong_result
                final = strong_result

        total_cost = sum(stage.cost_usd for stage in stages)
        trace = self._build_trace(
            engine,
            pre_route,
            retrieval_rounds,
            context_assessments,
            repair_decisions,
            fallback,
            stages,
            total_cost,
            degraded,
        )
        valid_document_ids = {document.document_id for document in selected_documents}
        citations = tuple(
            citation
            for citation in DeepSeekGenerator.extract_citations(final.text)
            if citation in valid_document_ids
        )
        answer = final.text
        if status is CompletionStatus.ABSTAINED:
            answer = "Unable to answer reliably because sufficient evidence was not found."
            citations = ()
        return AnswerResponse(
            query_id=request.query_id,
            answer=answer,
            citations=citations,
            status=status,
            trace=trace if self._should_include_trace(include_trace) else None,
        )

    async def _regenerate_strong(
        self,
        request: QueryRequest,
        engine: RouterKind,
        pre_route: PreRouteDecision,
        retrieval_rounds: list[RetrievalResult],
        context_assessments: list[ContextAssessment],
        repair_decisions: list[RetrievalRepairDecision],
        fallback: FallbackDecision,
        selected_documents: tuple[RetrievedDocument, ...],
        stages: list[StageTrace],
        *,
        include_trace: bool | None,
    ) -> GenerationResult | AnswerResponse:
        """Thực thi strong fallback và chuyển provider failure thành abstention."""

        strong_started = perf_counter()
        try:
            final = await self._generator.generate(
                request,
                selected_documents,
                ModelTier.STRONG,
            )
        except ProviderError as error:
            stages.append(
                StageTrace(
                    stage="generation_fallback",
                    latency_ms=self._elapsed_ms(strong_started),
                    error=type(error).__name__,
                )
            )
            return self._generation_failure_response(
                request,
                engine,
                pre_route,
                retrieval_rounds,
                context_assessments,
                repair_decisions,
                fallback,
                stages,
                include_trace=include_trace,
            )
        stages.append(
            StageTrace(
                stage="generation_fallback",
                latency_ms=self._elapsed_ms(strong_started),
                cost_usd=final.cost_usd,
                metadata={
                    "provider": "deepseek",
                    "model_id": final.model_id,
                    "tier": final.model_tier.value,
                    "input_tokens": final.usage.input_tokens,
                    "output_tokens": final.usage.output_tokens,
                    "cached_input_tokens": final.usage.cached_input_tokens,
                },
            )
        )
        return final

    def _generation_failure_response(
        self,
        request: QueryRequest,
        engine: RouterKind,
        pre_route: PreRouteDecision,
        retrieval_rounds: list[RetrievalResult],
        context_assessments: list[ContextAssessment],
        repair_decisions: list[RetrievalRepairDecision],
        fallback: FallbackDecision | None,
        stages: list[StageTrace],
        *,
        include_trace: bool | None,
    ) -> AnswerResponse:
        """Chuyển strong-generation failure thành abstention có audit trace."""

        trace = self._build_trace(
            engine,
            pre_route,
            retrieval_rounds,
            context_assessments,
            repair_decisions,
            fallback,
            stages,
            sum(stage.cost_usd for stage in stages),
            True,
        )
        return AnswerResponse(
            query_id=request.query_id,
            answer="Unable to answer reliably because the answer provider failed.",
            status=CompletionStatus.ABSTAINED,
            trace=trace if self._should_include_trace(include_trace) else None,
        )

    async def _decide_fallback_safely(
        self,
        decision_engine: DecisionEngine,
        engine: RouterKind,
        request: QueryRequest,
        context: ContextAssessment | None,
        draft: GenerationResult,
    ) -> tuple[FallbackDecision, bool]:
        try:
            return await decision_engine.decide_fallback(request, context, draft), False
        except Exception:
            return self._safe_fallback(engine, context, draft), True

    def _should_include_trace(self, value: bool | None) -> bool:
        return self._settings.show_trace_by_default if value is None else value

    @staticmethod
    def _elapsed_ms(started: float) -> float:
        return (perf_counter() - started) * 1000

    @staticmethod
    def _usage_cost(usage: DecisionUsage | None) -> float:
        return 0.0 if usage is None else usage.cost_usd

    @staticmethod
    def _usage_metadata(usage: DecisionUsage | None) -> dict[str, object]:
        if usage is None:
            return {}
        return {
            "provider": usage.provider,
            "model_id": usage.model_id,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cached_input_tokens": usage.cached_input_tokens,
        }

    @staticmethod
    def _alternate_source(source: RetrievalSource) -> RetrievalSource:
        return RetrievalSource.WEB if source is RetrievalSource.VECTOR else RetrievalSource.VECTOR

    @staticmethod
    def _merge_documents(
        existing: tuple[RetrievedDocument, ...],
        incoming: tuple[RetrievedDocument, ...],
        limit: int | None,
    ) -> tuple[RetrievedDocument, ...]:
        return deduplicate_documents((*existing, *incoming), limit=limit)

    @staticmethod
    def _rank_documents(
        documents: tuple[RetrievedDocument, ...],
        limit: int,
    ) -> tuple[RetrievedDocument, ...]:
        """Xếp hạng candidate toàn cục để evidence repair có thể vào top-k."""

        ranked = sorted(
            enumerate(documents),
            key=lambda item: (
                -(item[1].provider_score if item[1].provider_score is not None else 0.0),
                item[0],
            ),
        )
        return tuple(document for _, document in ranked[:limit])

    @staticmethod
    def _select_documents(
        documents: tuple[RetrievedDocument, ...], context: ContextAssessment | None
    ) -> tuple[RetrievedDocument, ...]:
        if context is None:
            return ()
        accepted = set(context.accepted_document_ids)
        rejected = set(context.rejected_injection_ids)
        conflicting = set(context.conflicting_document_ids)
        return tuple(
            document.model_copy(
                update={"metadata": {**document.metadata, "_context_conflict": True}}
            )
            if document.document_id in conflicting
            else document
            for document in documents
            if document.document_id in accepted and document.document_id not in rejected
        )

    @staticmethod
    def _evidence[DecisionT](
        selected: DecisionT,
        probabilities: dict[str, float],
        engine: RouterKind,
        reason: str,
    ) -> DecisionEvidence[DecisionT]:
        return DecisionEvidence(
            selected=selected,
            raw_probabilities=probabilities,
            confidence=max(probabilities.values()),
            provenance=ProbabilityProvenance.HEURISTIC,
            engine=engine,
            model_id="safe-policy-v1",
            prompt_version="safe-v1",
            reasons=(reason,),
        )

    def _safe_pre_route(self, request: QueryRequest, engine: RouterKind) -> PreRouteDecision:
        temporal = bool(
            re.search(
                r"\b(latest|current|today|yesterday|recent|now|this year)\b",
                request.query,
                re.IGNORECASE,
            )
        )
        source = RetrievalSource.WEB if temporal else RetrievalSource.VECTOR
        source_probs = {
            RetrievalSource.NONE.value: 0.02,
            RetrievalSource.VECTOR.value: 0.08 if temporal else 0.90,
            RetrievalSource.WEB.value: 0.90 if temporal else 0.08,
        }
        return PreRouteDecision(
            retrieval_source=self._evidence(source, source_probs, engine, "router_failure"),
            complexity=self._evidence(
                Complexity.HIGH,
                {"low": 0.02, "medium": 0.08, "high": 0.90},
                engine,
                "router_failure",
            ),
            initial_model_tier=self._evidence(
                ModelTier.STRONG,
                {"economy": 0.05, "strong": 0.95},
                engine,
                "router_failure",
            ),
        )

    def _safe_context_assessment(
        self, engine: RouterKind, documents: tuple[RetrievedDocument, ...]
    ) -> ContextAssessment:
        accepted = tuple(document.document_id for document in documents)
        return ContextAssessment(
            quality=self._evidence(
                ContextQuality.PARTIAL,
                {"insufficient": 0.35, "partial": 0.60, "sufficient": 0.05},
                engine,
                "assessment_failure",
            ),
            accepted_document_ids=accepted,
        )

    def _ablated_context_assessment(
        self,
        engine: RouterKind,
        documents: tuple[RetrievedDocument, ...],
    ) -> ContextAssessment:
        """Chấp nhận toàn bộ evidence ban đầu cho ablation bỏ context gate."""

        accepted = tuple(document.document_id for document in documents)
        return ContextAssessment(
            quality=self._evidence(
                ContextQuality.SUFFICIENT,
                {"insufficient": 0.0, "partial": 0.0, "sufficient": 1.0},
                engine,
                "context_gate_ablation",
            ),
            accepted_document_ids=accepted,
        )

    def _safe_repair(self, engine: RouterKind, source: RetrievalSource) -> RetrievalRepairDecision:
        action = (
            RepairAction.SWITCH_WEB if source is RetrievalSource.VECTOR else RepairAction.REFINE_WEB
        )
        probabilities = {item.value: 0.0 for item in RepairAction}
        probabilities[action.value] = 1.0
        return RetrievalRepairDecision(
            action=self._evidence(action, probabilities, engine, "repair_failure")
        )

    def _safe_fallback(
        self,
        engine: RouterKind,
        context: ContextAssessment | None,
        draft: GenerationResult,
    ) -> FallbackDecision:
        if context is not None and context.quality.selected is ContextQuality.INSUFFICIENT:
            action = FallbackAction.ABSTAIN
        elif draft.model_tier is ModelTier.ECONOMY:
            action = FallbackAction.REGENERATE_STRONG
        else:
            action = FallbackAction.ACCEPT
        probabilities = {item.value: 0.0 for item in FallbackAction}
        probabilities[action.value] = 1.0
        return FallbackDecision(
            action=self._evidence(action, probabilities, engine, "fallback_failure")
        )

    @staticmethod
    def _build_trace(
        engine: RouterKind,
        pre_route: PreRouteDecision,
        retrieval_rounds: list[RetrievalResult],
        context_assessments: list[ContextAssessment],
        repair_decisions: list[RetrievalRepairDecision],
        fallback: FallbackDecision | None,
        stages: list[StageTrace],
        total_cost: float,
        degraded: bool,
    ) -> DecisionTrace:
        return DecisionTrace(
            router=engine,
            pre_route=pre_route,
            retrieval_rounds=tuple(retrieval_rounds),
            context_assessments=tuple(context_assessments),
            repair_decisions=tuple(repair_decisions),
            fallback=fallback,
            stages=tuple(stages),
            total_cost_usd=total_cost,
            degraded=degraded,
        )
