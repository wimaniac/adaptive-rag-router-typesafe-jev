"""Gói triển khai Adaptive RAG Router với các decision engine có thể thay thế."""

from typing import TYPE_CHECKING, Any

from adaptive_rag_router.domain.enums import RouterKind
from adaptive_rag_router.domain.models import AnswerResponse, QueryRequest

if TYPE_CHECKING:
    from adaptive_rag_router.application.pipeline import AdaptiveRAGRouter

__all__ = ["AdaptiveRAGRouter", "AnswerResponse", "QueryRequest", "RouterKind"]


def __getattr__(name: str) -> Any:
    """Nạp pipeline theo nhu cầu để domain/config dùng được độc lập.

    Args:
        name: Tên public attribute được yêu cầu.

    Returns:
        Public class tương ứng.

    Raises:
        AttributeError: Khi tên không thuộc public API.
    """

    if name == "AdaptiveRAGRouter":
        from adaptive_rag_router.application.pipeline import AdaptiveRAGRouter

        return AdaptiveRAGRouter
    raise AttributeError(name)
