"""Smoke test có opt-in cho ba provider thật với ngân sách Tavily một credit."""

from __future__ import annotations

import os

import pytest

from adaptive_rag_router.config import Settings
from adaptive_rag_router.domain import ModelTier, QueryRequest, RouterKind
from adaptive_rag_router.generation import DeepSeekGenerator
from adaptive_rag_router.retrieval import (
    CreditLedger,
    HttpxTavilyTransport,
    InMemorySearchCache,
    RollingWindowRateLimiter,
    TavilyRetriever,
)
from adaptive_rag_router.routing import build_decision_engine

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("RUN_LIVE_INTEGRATION") != "1",
        reason="Chỉ gọi provider thật khi RUN_LIVE_INTEGRATION=1",
    ),
]


@pytest.mark.asyncio
async def test_typesafe_jev_smoke() -> None:
    """Gửi đúng một System One request đã batch ba quyết định pre-route."""

    settings = Settings()
    router = build_decision_engine(RouterKind.JEV, settings)

    result = await router.pre_route(QueryRequest(query="What is retrieval-augmented generation?"))

    assert result.retrieval_source.engine is RouterKind.JEV
    assert result.retrieval_source.model_id


@pytest.mark.asyncio
async def test_deepseek_generation_smoke() -> None:
    """Gửi đúng một request economy không bật thinking tới DeepSeek."""

    settings = Settings()
    generator = DeepSeekGenerator(settings, max_output_tokens=64)

    result = await generator.generate(
        QueryRequest(query="Reply with exactly: adaptive router"),
        (),
        ModelTier.ECONOMY,
    )

    assert result.text.strip()
    assert result.model_tier is ModelTier.ECONOMY


@pytest.mark.asyncio
async def test_tavily_basic_search_smoke() -> None:
    """Gửi tối đa một Basic Search và xác nhận ledger không vượt một credit."""

    settings = Settings()
    if settings.tavily_api_key is None:
        pytest.fail("Thiếu TAVILY_API_KEY")
    transport = HttpxTavilyTransport(
        settings.tavily_api_key,
        timeout_seconds=settings.request_timeout_seconds,
    )
    ledger = CreditLedger(1)
    retriever = TavilyRetriever(
        transport,
        cache=InMemorySearchCache(),
        limiter=RollingWindowRateLimiter(max_requests=1, max_concurrency=1),
        credit_ledger=ledger,
        max_results=1,
        max_retries=0,
    )
    try:
        result = await retriever.retrieve(
            "official Python programming language website",
            round_index=0,
        )
    finally:
        await transport.close()

    assert result.provider == "tavily"
    assert ledger.consumed <= 1
