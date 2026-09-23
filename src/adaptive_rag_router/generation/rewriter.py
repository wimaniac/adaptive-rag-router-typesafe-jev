"""Rewrite truy vấn retrieval bằng một service chung để benchmark công bằng."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from openai import AsyncOpenAI

from adaptive_rag_router.config.settings import Settings
from adaptive_rag_router.domain.enums import ModelTier
from adaptive_rag_router.domain.errors import ConfigurationError, ProviderError
from adaptive_rag_router.domain.models import QueryRequest, RetrievedDocument, TokenUsage
from adaptive_rag_router.generation.catalog import ModelCatalog


@dataclass(frozen=True, slots=True)
class QueryRewriteResult:
    """Kết quả rewrite cùng usage/cost để trace và budget không bỏ sót.

    Args:
        query: Query mới cho retrieval round tiếp theo.
        usage: Token usage do DeepSeek trả về.
        cost_usd: Variable cost theo pricing snapshot.
    """

    query: str
    usage: TokenUsage = field(default_factory=TokenUsage)
    cost_usd: float = 0.0


class QueryRewriter(Protocol):
    """Giao diện rewrite truy vấn khi context còn thiếu."""

    async def rewrite(
        self,
        request: QueryRequest,
        missing_information: tuple[str, ...],
        documents: tuple[RetrievedDocument, ...],
    ) -> QueryRewriteResult:
        """Trả query mới cùng usage/cost cho retrieval round tiếp theo."""


class DeepSeekQueryRewriter:
    """Dùng DeepSeek Flash non-thinking để tạo search query dưới 400 ký tự."""

    def __init__(self, settings: Settings, *, client: AsyncOpenAI | Any | None = None) -> None:
        """Khởi tạo rewriter với client có thể inject cho unit test."""

        if client is None:
            if settings.deepseek_api_key is None:
                raise ConfigurationError("Thiếu DEEPSEEK_API_KEY")
            client = AsyncOpenAI(
                api_key=settings.deepseek_api_key.get_secret_value(),
                base_url=settings.deepseek_base_url,
                timeout=settings.request_timeout_seconds,
            )
        self._client = client
        self._model = settings.deepseek_router_model
        self._pricing = ModelCatalog(settings).get(ModelTier.ECONOMY)

    async def rewrite(
        self,
        request: QueryRequest,
        missing_information: tuple[str, ...],
        documents: tuple[RetrievedDocument, ...],
    ) -> QueryRewriteResult:
        """Tạo một query mới dựa trên khoảng trống evidence, không tự trả lời câu hỏi."""

        known_titles = ", ".join(
            document.title or document.document_id for document in documents[:8]
        )
        gaps = "; ".join(missing_information) or "Evidence is incomplete or irrelevant"
        prompt = (
            "Rewrite the original question as one focused web/vector search query. "
            "Do not answer it. Output only the query, no quotes, at most 350 characters.\n"
            f"Original: {request.query}\nMissing information: {gaps}\n"
            f"Already seen sources: {known_titles or '(none)'}"
        )
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=128,
                extra_body={"thinking": {"type": "disabled"}},
            )
        except Exception as exc:
            raise ProviderError(f"DeepSeek query rewrite thất bại: {type(exc).__name__}") from exc
        content = (response.choices[0].message.content or "").strip() if response.choices else ""
        if not content:
            raise ProviderError("DeepSeek không trả rewritten query")
        raw_usage = getattr(response, "usage", None)
        details = getattr(raw_usage, "prompt_tokens_details", None)
        usage = TokenUsage(
            input_tokens=int(getattr(raw_usage, "prompt_tokens", 0) or 0),
            output_tokens=int(getattr(raw_usage, "completion_tokens", 0) or 0),
            cached_input_tokens=int(getattr(details, "cached_tokens", 0) or 0),
        )
        return QueryRewriteResult(
            query=content[:400],
            usage=usage,
            cost_usd=self._pricing.calculate_cost(usage),
        )
