"""Khai báo contract bất đồng bộ chung cho mọi decision engine của hệ thống."""

from typing import Protocol, runtime_checkable

from adaptive_rag_router.domain import (
    ContextAssessment,
    FallbackDecision,
    GenerationResult,
    PreRouteDecision,
    QueryRequest,
    RetrievalRepairDecision,
    RetrievalResult,
)


@runtime_checkable
class DecisionEngine(Protocol):
    """Contract mà Jev, LLM và rule-based router cùng phải triển khai."""

    async def pre_route(self, request: QueryRequest) -> PreRouteDecision:
        """Quyết định nguồn retrieval, độ phức tạp và model tier ban đầu."""

        ...

    async def assess_context(
        self,
        request: QueryRequest,
        retrieval: RetrievalResult,
    ) -> ContextAssessment:
        """Đánh giá mức hữu ích và an toàn của context vừa retrieval."""

        ...

    async def decide_repair(
        self,
        request: QueryRequest,
        context: ContextAssessment,
        retrieval_rounds: tuple[RetrievalResult, ...],
    ) -> RetrievalRepairDecision:
        """Chọn hành động tiếp theo cho adaptive retrieval loop."""

        ...

    async def decide_fallback(
        self,
        request: QueryRequest,
        context: ContextAssessment | None,
        draft: GenerationResult,
    ) -> FallbackDecision:
        """Chọn chấp nhận, nâng model hoặc abstain sau khi có draft."""

        ...
