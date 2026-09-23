"""Kiểm thử DeepSeek adapter bằng fake OpenAI-compatible client, không gọi mạng."""

from types import SimpleNamespace

import pytest

from adaptive_rag_router.config.settings import Settings
from adaptive_rag_router.domain.enums import ModelTier, RetrievalSource
from adaptive_rag_router.domain.models import QueryRequest, RetrievedDocument
from adaptive_rag_router.generation.deepseek import DeepSeekGenerator
from adaptive_rag_router.generation.rewriter import DeepSeekQueryRewriter


class FakeCompletions:
    """Ghi request và trả response tối thiểu giống OpenAI SDK."""

    def __init__(self) -> None:
        """Khởi tạo fake chưa nhận request."""

        self.last_kwargs: dict[str, object] = {}

    async def create(self, **kwargs: object) -> SimpleNamespace:
        """Trả answer có citation và token usage xác định."""

        self.last_kwargs = kwargs
        return SimpleNamespace(
            model="deepseek-flash-resolved",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="RAG uses retrieval [doc-1]."),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=100,
                completion_tokens=20,
                prompt_tokens_details=SimpleNamespace(cached_tokens=10),
            ),
        )


class FakeClient:
    """Bọc fake completions theo shape `client.chat.completions`."""

    def __init__(self) -> None:
        """Khởi tạo các namespace cần cho adapter."""

        self.completions = FakeCompletions()
        self.chat = SimpleNamespace(completions=self.completions)


@pytest.mark.asyncio
async def test_generate_disables_thinking_for_economy() -> None:
    """Economy generation phải tắt thinking và giữ usage/model metadata."""

    client = FakeClient()
    generator = DeepSeekGenerator(Settings(_env_file=None), client=client)
    request = QueryRequest(query="What is RAG?")
    document = RetrievedDocument(
        document_id="doc-1",
        text="RAG retrieves documents before generation.",
        source=RetrievalSource.VECTOR,
    )

    result = await generator.generate(request, (document,), ModelTier.ECONOMY)

    assert result.model_id == "deepseek-flash-resolved"
    assert result.usage.input_tokens == 100
    assert result.cost_usd > 0
    assert client.completions.last_kwargs["extra_body"] == {"thinking": {"type": "disabled"}}
    assert client.completions.last_kwargs["max_tokens"] == 1024
    assert DeepSeekGenerator.extract_citations(result.text) == ("doc-1",)


@pytest.mark.asyncio
async def test_generate_enables_high_thinking_for_strong() -> None:
    """Strong generation phải bật thinking high theo plan."""

    client = FakeClient()
    generator = DeepSeekGenerator(Settings(_env_file=None), client=client)

    await generator.generate(QueryRequest(query="Compare two systems"), (), ModelTier.STRONG)

    assert client.completions.last_kwargs["extra_body"] == {"thinking": {"type": "enabled"}}
    assert client.completions.last_kwargs["reasoning_effort"] == "high"
    assert client.completions.last_kwargs["max_tokens"] == 4096


@pytest.mark.asyncio
async def test_strong_budget_reserves_reasoning_tokens_for_small_answer_cap() -> None:
    """Strong tier phải có tối thiểu 2.048 token gồm cả hidden reasoning."""

    client = FakeClient()
    generator = DeepSeekGenerator(
        Settings(_env_file=None),
        client=client,
        max_output_tokens=128,
    )

    await generator.generate(QueryRequest(query="Reason carefully"), (), ModelTier.STRONG)

    assert client.completions.last_kwargs["max_tokens"] == 2048


@pytest.mark.asyncio
async def test_query_rewriter_returns_typed_usage_and_cost() -> None:
    """Rewrite phải đưa DeepSeek usage vào trace/budget thay vì chỉ trả chuỗi."""

    client = FakeClient()
    rewriter = DeepSeekQueryRewriter(Settings(_env_file=None), client=client)

    result = await rewriter.rewrite(QueryRequest(query="original"), ("missing fact",), ())

    assert result.query == "RAG uses retrieval [doc-1]."
    assert result.usage.input_tokens == 100
    assert result.usage.output_tokens == 20
    assert result.cost_usd > 0
    assert client.completions.last_kwargs["max_tokens"] == 128
