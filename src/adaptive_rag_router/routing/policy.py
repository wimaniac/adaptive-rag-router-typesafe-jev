"""Cung cấp deterministic safety policy khi decision provider không khả dụng."""

from enum import StrEnum

from adaptive_rag_router.domain import (
    Complexity,
    ContextAssessment,
    ContextQuality,
    DecisionEvidence,
    FallbackAction,
    FallbackDecision,
    GenerationResult,
    ModelTier,
    PassageAssessment,
    PreRouteDecision,
    ProbabilityProvenance,
    QueryRequest,
    RepairAction,
    RetrievalRepairDecision,
    RetrievalResult,
    RetrievalSource,
    RouterKind,
)
from adaptive_rag_router.routing.helpers import build_evidence, peaked_distribution


class SafeDecisionPolicy:
    """Fail-closed policy dùng khi orchestration bắt lỗi từ router chính."""

    model_id = "safe-policy-v1"
    prompt_version = "safe-policy-v1"

    def _evidence[EnumT: StrEnum](
        self,
        enum_type: type[EnumT],
        selected: EnumT,
        reason: str,
    ) -> DecisionEvidence[EnumT]:
        return build_evidence(
            enum_type,
            selected,
            peaked_distribution(selected, enum_type, peak=1.0),
            provenance=ProbabilityProvenance.HEURISTIC,
            engine=RouterKind.RULE,
            model_id=self.model_id,
            prompt_version=self.prompt_version,
            reasons=(reason,),
        )

    async def pre_route(self, request: QueryRequest) -> PreRouteDecision:
        """Chọn vector và strong model để giảm rủi ro khi pre-router lỗi."""

        del request
        return PreRouteDecision(
            retrieval_source=self._evidence(
                RetrievalSource, RetrievalSource.VECTOR, "provider_failure"
            ),
            complexity=self._evidence(Complexity, Complexity.MEDIUM, "unknown_complexity"),
            initial_model_tier=self._evidence(
                ModelTier, ModelTier.STRONG, "conservative_model_tier"
            ),
        )

    async def assess_context(
        self,
        request: QueryRequest,
        retrieval: RetrievalResult,
    ) -> ContextAssessment:
        """Không tự tuyên bố context đủ khi provider đánh giá context bị lỗi."""

        del request
        passages = tuple(
            PassageAssessment(
                document_id=document.document_id,
                relevance_probability=0.0,
                evidence_probability=0.0,
                contradiction_probability=0.0,
                injection_probability=0.0,
            )
            for document in retrieval.documents
        )
        return ContextAssessment(
            quality=self._evidence(
                ContextQuality, ContextQuality.INSUFFICIENT, "assessment_unavailable"
            ),
            passages=passages,
        )

    async def decide_repair(
        self,
        request: QueryRequest,
        context: ContextAssessment,
        retrieval_rounds: tuple[RetrievalResult, ...],
    ) -> RetrievalRepairDecision:
        """Abstain thay vì tiếp tục gọi provider không kiểm soát khi policy lỗi."""

        del request, context, retrieval_rounds
        return RetrievalRepairDecision(
            action=self._evidence(RepairAction, RepairAction.ABSTAIN, "router_unavailable"),
            missing_information=("routing_decision",),
        )

    async def decide_fallback(
        self,
        request: QueryRequest,
        context: ContextAssessment | None,
        draft: GenerationResult,
    ) -> FallbackDecision:
        """Abstain khi không thể xác nhận draft an toàn."""

        del request, context, draft
        return FallbackDecision(
            action=self._evidence(FallbackAction, FallbackAction.ABSTAIN, "review_unavailable")
        )
