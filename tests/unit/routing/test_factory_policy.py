"""Kiểm thử factory dependency injection và deterministic safety policy."""

from types import SimpleNamespace
from typing import Any

from adaptive_rag_router.config import Settings
from adaptive_rag_router.domain import (
    ContextQuality,
    FallbackAction,
    GenerationResult,
    ModelTier,
    QueryRequest,
    RepairAction,
    RetrievalResult,
    RetrievalSource,
    RouterKind,
)
from adaptive_rag_router.routing import (
    JevRouter,
    LLMRouter,
    RuleBasedRouter,
    SafeDecisionPolicy,
    build_decision_engine,
)


class _UnusedJevClient:
    async def system_one(self, **kwargs: Any) -> Any:
        raise AssertionError("factory không được gọi API")


def test_factory_builds_all_engines_with_injected_clients() -> None:
    """Factory chỉ khởi tạo object và không cần key khi dependency đã inject."""

    settings = Settings(_env_file=None)
    llm_client = SimpleNamespace()
    jev_client = _UnusedJevClient()

    assert isinstance(build_decision_engine(RouterKind.RULE, settings), RuleBasedRouter)
    assert isinstance(
        build_decision_engine(RouterKind.LLM, settings, llm_client=llm_client), LLMRouter
    )
    assert isinstance(
        build_decision_engine(RouterKind.JEV, settings, jev_client=jev_client), JevRouter
    )


async def test_safe_policy_fails_closed() -> None:
    """Safety policy không chấp nhận evidence/draft khi assessor không khả dụng."""

    policy = SafeDecisionPolicy()
    request = QueryRequest(query="Question")
    retrieval = RetrievalResult(
        source=RetrievalSource.VECTOR,
        query=request.query,
        provider="fixture",
    )
    context = await policy.assess_context(request, retrieval)
    repair = await policy.decide_repair(request, context, (retrieval,))
    fallback = await policy.decide_fallback(
        request,
        context,
        GenerationResult(
            text="Draft",
            model_tier=ModelTier.STRONG,
            model_id="strong",
        ),
    )

    assert context.quality.selected is ContextQuality.INSUFFICIENT
    assert repair.action.selected is RepairAction.ABSTAIN
    assert fallback.action.selected is FallbackAction.ABSTAIN
