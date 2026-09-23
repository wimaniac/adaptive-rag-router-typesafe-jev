"""Cài đặt deterministic baseline bằng heuristic có version và probability giả lập."""

from __future__ import annotations

import re
from collections.abc import Iterable
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

_WORD_RE = re.compile(r"[\w'-]+", flags=re.UNICODE)
_CURRENT_TERMS = frozenset(
    {
        "current",
        "currently",
        "latest",
        "live",
        "news",
        "today",
        "tonight",
        "weather",
        "price",
        "score",
        "president",
        "ceo",
        "2025",
        "2026",
        "hiện",
        "nay",
        "mới",
    }
)
_CORPUS_TERMS = frozenset(
    {
        "document",
        "documents",
        "policy",
        "manual",
        "contract",
        "report",
        "knowledge",
        "corpus",
        "tài",
        "liệu",
        "chính",
        "sách",
    }
)
_HIGH_COMPLEXITY_TERMS = frozenset(
    {
        "analyze",
        "compare",
        "evaluate",
        "derive",
        "prove",
        "tradeoff",
        "multi-hop",
        "phân",
        "tích",
        "so",
        "sánh",
        "chứng",
        "minh",
    }
)
_NO_RETRIEVAL_PREFIXES = (
    "hello",
    "hi ",
    "hey",
    "thanks",
    "thank you",
    "translate ",
    "rewrite ",
    "summarize this",
    "xin chào",
    "cảm ơn",
)
_INJECTION_MARKERS = (
    "ignore previous instructions",
    "ignore all instructions",
    "system prompt",
    "developer message",
    "reveal your prompt",
    "bỏ qua chỉ dẫn",
)


def _tokens(text: str) -> set[str]:
    return {match.group(0).lower() for match in _WORD_RE.finditer(text)}


def _bounded(value: float) -> float:
    return max(0.0, min(1.0, value))


class RuleBasedRouter:
    """Baseline xác định dùng heuristic công khai và không gọi provider ngoài."""

    def __init__(
        self,
        *,
        version: str = "rule-v1",
        prompt_version: str = "rules-2026-09-21",
        max_repair_rounds: int = 2,
    ) -> None:
        """Khởi tạo baseline.

        Args:
            version: ID phiên bản rule để audit benchmark.
            prompt_version: ID bộ tiêu chí tương đương prompt.
            max_repair_rounds: Số vòng repair tối đa trước khi dừng hoặc abstain.
        """

        self.model_id = version
        self.prompt_version = prompt_version
        self.max_repair_rounds = max_repair_rounds

    def _evidence[EnumT: StrEnum](
        self,
        enum_type: type[EnumT],
        selected: EnumT,
        peak: float,
        *reasons: str,
    ) -> DecisionEvidence[EnumT]:
        # Helper này giữ mọi heuristic distribution cùng provenance; calibration được làm ở layer khác.
        return build_evidence(
            enum_type,
            selected,
            peaked_distribution(selected, enum_type, peak=peak),
            provenance=ProbabilityProvenance.HEURISTIC,
            engine=RouterKind.RULE,
            model_id=self.model_id,
            prompt_version=self.prompt_version,
            reasons=reasons,
        )

    async def pre_route(self, request: QueryRequest) -> PreRouteDecision:
        """Phân tuyến ban đầu bằng tín hiệu thời sự, corpus và độ dài truy vấn."""

        lowered = request.query.lower()
        tokens = _tokens(lowered)
        metadata = request.metadata

        if bool(metadata.get("requires_current_information")) or tokens & _CURRENT_TERMS:
            source = RetrievalSource.WEB
            source_reason = "time_sensitive"
            source_peak = 0.88
        elif bool(metadata.get("corpus_available")) or tokens & _CORPUS_TERMS:
            source = RetrievalSource.VECTOR
            source_reason = "corpus_signal"
            source_peak = 0.84
        elif lowered.startswith(_NO_RETRIEVAL_PREFIXES):
            source = RetrievalSource.NONE
            source_reason = "direct_task"
            source_peak = 0.92
        elif len(tokens) >= 18 or "?" in request.query:
            source = RetrievalSource.VECTOR
            source_reason = "knowledge_question"
            source_peak = 0.68
        else:
            source = RetrievalSource.NONE
            source_reason = "low_information_need"
            source_peak = 0.72

        conjunctions = sum(lowered.count(term) for term in (" and ", " versus ", " vs ", " và "))
        if len(tokens) >= 32 or tokens & _HIGH_COMPLEXITY_TERMS or conjunctions >= 2:
            complexity = Complexity.HIGH
            complexity_peak = 0.82
            complexity_reason = "multi_step_reasoning"
        elif len(tokens) >= 13 or conjunctions == 1:
            complexity = Complexity.MEDIUM
            complexity_peak = 0.74
            complexity_reason = "moderate_scope"
        else:
            complexity = Complexity.LOW
            complexity_peak = 0.84
            complexity_reason = "short_direct_query"

        tier = ModelTier.STRONG if complexity is Complexity.HIGH else ModelTier.ECONOMY
        tier_reason = "high_complexity" if tier is ModelTier.STRONG else "cost_first"
        return PreRouteDecision(
            retrieval_source=self._evidence(RetrievalSource, source, source_peak, source_reason),
            complexity=self._evidence(Complexity, complexity, complexity_peak, complexity_reason),
            initial_model_tier=self._evidence(ModelTier, tier, 0.86, tier_reason),
        )

    async def assess_context(
        self,
        request: QueryRequest,
        retrieval: RetrievalResult,
    ) -> ContextAssessment:
        """Chấm passage bằng overlap, provider score và dấu hiệu prompt injection."""

        query_tokens = _tokens(request.query)
        assessments: list[PassageAssessment] = []
        accepted: list[str] = []
        conflicts: list[str] = []
        injections: list[str] = []

        for document in retrieval.documents:
            document_tokens = _tokens(document.text)
            overlap = len(query_tokens & document_tokens) / max(1, len(query_tokens))
            provider_score = _bounded(document.provider_score or 0.0)
            relevance = _bounded(0.7 * overlap + 0.3 * provider_score)
            # Evidence cần cả liên quan lẫn lượng nội dung tối thiểu; provider score chỉ là tín hiệu phụ.
            length_signal = min(1.0, len(document_tokens) / 40.0)
            evidence = _bounded(0.65 * relevance + 0.35 * length_signal)
            lowered = document.text.lower()
            injection = 0.98 if any(marker in lowered for marker in _INJECTION_MARKERS) else 0.02
            if bool(document.metadata.get("prompt_injection")):
                injection = 0.99
            contradiction = 0.9 if bool(document.metadata.get("contradictory")) else 0.05

            assessment = PassageAssessment(
                document_id=document.document_id,
                relevance_probability=relevance,
                evidence_probability=evidence,
                contradiction_probability=contradiction,
                injection_probability=injection,
            )
            assessments.append(assessment)
            if injection >= 0.5:
                injections.append(document.document_id)
            if contradiction >= 0.5:
                conflicts.append(document.document_id)
            if relevance >= 0.30 and evidence >= 0.30 and injection < 0.5:
                accepted.append(document.document_id)

        strong_passages = sum(
            item.relevance_probability >= 0.55 and item.evidence_probability >= 0.50
            for item in assessments
            if item.document_id in accepted
        )
        if strong_passages >= 2 or (strong_passages == 1 and len(accepted) >= 2 and not conflicts):
            quality = ContextQuality.SUFFICIENT
            peak = 0.82
            reason = "enough_supporting_passages"
        elif accepted:
            quality = ContextQuality.PARTIAL
            peak = 0.72
            reason = "limited_support"
        else:
            quality = ContextQuality.INSUFFICIENT
            peak = 0.9
            reason = "no_usable_evidence"

        return ContextAssessment(
            quality=self._evidence(ContextQuality, quality, peak, reason),
            passages=tuple(assessments),
            accepted_document_ids=tuple(accepted),
            conflicting_document_ids=tuple(conflicts),
            rejected_injection_ids=tuple(injections),
        )

    async def decide_repair(
        self,
        request: QueryRequest,
        context: ContextAssessment,
        retrieval_rounds: tuple[RetrievalResult, ...],
    ) -> RetrievalRepairDecision:
        """Dừng khi đủ context, nếu chưa đủ thì đổi hoặc tinh chỉnh nguồn retrieval."""

        del request
        quality = context.quality.selected
        if quality is ContextQuality.SUFFICIENT:
            action = RepairAction.STOP
            reason = "context_sufficient"
        elif len(retrieval_rounds) >= self.max_repair_rounds + 1:
            action = (
                RepairAction.ABSTAIN
                if quality is ContextQuality.INSUFFICIENT
                else RepairAction.STOP
            )
            reason = "repair_budget_exhausted"
        elif not retrieval_rounds:
            action = RepairAction.REWRITE_VECTOR
            reason = "start_vector_repair"
        elif retrieval_rounds[-1].source is RetrievalSource.VECTOR:
            action = RepairAction.SWITCH_WEB
            reason = "vector_context_inadequate"
        elif retrieval_rounds[-1].source is RetrievalSource.WEB:
            action = RepairAction.REFINE_WEB
            reason = "web_context_inadequate"
        else:
            action = RepairAction.REWRITE_VECTOR
            reason = "no_retrieval_context"

        missing = () if action is RepairAction.STOP else ("supporting_evidence",)
        return RetrievalRepairDecision(
            action=self._evidence(RepairAction, action, 0.88, reason),
            missing_information=missing,
        )

    async def decide_fallback(
        self,
        request: QueryRequest,
        context: ContextAssessment | None,
        draft: GenerationResult,
    ) -> FallbackDecision:
        """Áp dụng guardrail evidence trước khi chấp nhận hoặc nâng cấp draft."""

        del request
        if context is not None and context.quality.selected is ContextQuality.INSUFFICIENT:
            action = FallbackAction.ABSTAIN
            reason = "insufficient_evidence"
        elif not draft.text.strip() or draft.finish_reason in {"length", "content_filter", "error"}:
            action = (
                FallbackAction.REGENERATE_STRONG
                if draft.model_tier is ModelTier.ECONOMY
                else FallbackAction.ABSTAIN
            )
            reason = "incomplete_draft"
        elif draft.model_tier is ModelTier.ECONOMY and any(
            marker in draft.text.lower()
            for marker in ("i don't know", "not sure", "cannot determine", "không chắc")
        ):
            action = FallbackAction.REGENERATE_STRONG
            reason = "uncertain_draft"
        else:
            action = FallbackAction.ACCEPT
            reason = "draft_acceptable"

        return FallbackDecision(action=self._evidence(FallbackAction, action, 0.9, reason))


def count_accepted(assessments: Iterable[PassageAssessment]) -> int:
    """Đếm passage thỏa ngưỡng cơ bản; hữu ích cho audit rule baseline."""

    return sum(
        item.relevance_probability >= 0.30
        and item.evidence_probability >= 0.30
        and item.injection_probability < 0.5
        for item in assessments
    )
