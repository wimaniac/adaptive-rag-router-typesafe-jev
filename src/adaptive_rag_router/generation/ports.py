"""Khai báo port sinh câu trả lời để pipeline không phụ thuộc provider cụ thể."""

from typing import Protocol

from adaptive_rag_router.domain.enums import ModelTier
from adaptive_rag_router.domain.models import GenerationResult, QueryRequest, RetrievedDocument


class Generator(Protocol):
    """Giao diện bất đồng bộ cho một answer generator."""

    async def generate(
        self,
        request: QueryRequest,
        documents: tuple[RetrievedDocument, ...],
        model_tier: ModelTier,
    ) -> GenerationResult:
        """Sinh câu trả lời từ truy vấn và evidence đã được lọc."""
