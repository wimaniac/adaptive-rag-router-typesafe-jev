"""Kiểm thử offline các guardrail cache, limiter, retry và Tavily credit."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from adaptive_rag_router.domain.errors import (
    CreditBudgetExceededError,
    ProviderError,
    ProviderRateLimitError,
)
from adaptive_rag_router.retrieval import (
    CreditLedger,
    InMemorySearchCache,
    RollingWindowRateLimiter,
    TavilyRateLimitError,
    TavilyRetriever,
)


class FakeClock:
    """Đồng hồ monotonic có sleep tức thời cho limiter test."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def read(self) -> float:
        """Trả thời gian giả hiện tại."""

        return self.now

    async def sleep(self, seconds: float) -> None:
        """Tiến đồng hồ thay vì chờ thật."""

        self.sleeps.append(seconds)
        self.now += seconds


class FakeTransport:
    """Transport ghi nhận calls và trả chuỗi outcome đã cấu hình."""

    def __init__(self, outcomes: list[Mapping[str, Any] | Exception] | None = None) -> None:
        self.calls: list[tuple[str, int]] = []
        self._outcomes = outcomes or [_response()]

    async def search(self, query: str, *, max_results: int) -> Mapping[str, Any]:
        """Trả outcome tiếp theo hoặc ném exception fixture."""

        self.calls.append((query, max_results))
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _response(text: str = "Evidence") -> dict[str, Any]:
    return {
        "results": [
            {
                "title": "Result",
                "url": "https://example.com/item",
                "content": text,
                "score": 0.8,
            }
        ]
    }


@pytest.mark.asyncio
async def test_rolling_window_waits_after_limit() -> None:
    clock = FakeClock()
    limiter = RollingWindowRateLimiter(
        max_requests=2,
        window_seconds=60,
        max_concurrency=1,
        clock=clock.read,
        sleep=clock.sleep,
    )

    for _ in range(3):
        await limiter.acquire()
        limiter.release()

    assert clock.sleeps == [60.0]


@pytest.mark.asyncio
async def test_credit_ledger_blocks_before_exceeding_and_restores_checkpoint(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "credits.json"
    ledger = CreditLedger(2, checkpoint_path=checkpoint)

    await ledger.consume()
    await ledger.consume()
    with pytest.raises(CreditBudgetExceededError):
        await ledger.consume()

    restored = CreditLedger(2, checkpoint_path=checkpoint)
    assert restored.consumed == 2
    assert restored.remaining == 0


@pytest.mark.asyncio
async def test_cache_hit_skips_transport_limiter_and_credit() -> None:
    transport = FakeTransport()
    ledger = CreditLedger(3)
    cache = InMemorySearchCache()
    retriever = TavilyRetriever(transport, cache=cache, credit_ledger=ledger)

    first = await retriever.retrieve("Alpha query")
    second = await retriever.retrieve("  alpha   QUERY ")

    assert first.cached is False
    assert second.cached is True
    assert first.external_credits == 1
    assert second.external_credits == 0
    assert len(transport.calls) == 1
    assert ledger.consumed == 1
    assert second.documents[0].text == "Evidence"


@pytest.mark.asyncio
async def test_retriever_stops_at_credit_hard_cap() -> None:
    transport = FakeTransport([_response("one"), _response("two")])
    ledger = CreditLedger(1)
    retriever = TavilyRetriever(transport, credit_ledger=ledger)

    await retriever.retrieve("first")
    with pytest.raises(CreditBudgetExceededError):
        await retriever.retrieve("second")

    assert len(transport.calls) == 1


@pytest.mark.asyncio
async def test_429_uses_retry_after_and_retry_consumes_credit() -> None:
    transport = FakeTransport([TavilyRateLimitError(3.0), _response()])
    ledger = CreditLedger(4)
    delays: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        delays.append(seconds)

    retriever = TavilyRetriever(
        transport,
        credit_ledger=ledger,
        retry_base_seconds=1,
        sleep=fake_sleep,
        random_value=lambda: 0,
    )

    result = await retriever.retrieve("retry me")

    assert result.cached is False
    assert result.external_credits == 2
    assert delays == [3.0]
    assert ledger.consumed == 2
    assert len(transport.calls) == 2


@pytest.mark.asyncio
async def test_429_stops_after_four_retries() -> None:
    transport = FakeTransport([TavilyRateLimitError(0)] * 5)
    ledger = CreditLedger(10)

    async def no_sleep(seconds: float) -> None:
        assert seconds >= 0

    retriever = TavilyRetriever(
        transport,
        credit_ledger=ledger,
        max_retries=4,
        retry_base_seconds=0,
        sleep=no_sleep,
        random_value=lambda: 0,
    )

    with pytest.raises(ProviderRateLimitError, match="5 attempts"):
        await retriever.retrieve("always limited")

    assert ledger.consumed == 5


@pytest.mark.asyncio
async def test_per_query_guard_runs_before_global_credit_is_consumed() -> None:
    """Attempt bị query cap từ chối không làm hao ledger hoặc gọi transport."""

    transport = FakeTransport()
    ledger = CreditLedger(3)

    def deny() -> None:
        raise ProviderError("per-query cap")

    retriever = TavilyRetriever(
        transport,
        cache=InMemorySearchCache(),
        credit_ledger=ledger,
        attempt_guard=deny,
    )

    with pytest.raises(ProviderError, match="per-query cap"):
        await retriever.retrieve("query")

    assert ledger.consumed == 0
    assert transport.calls == []
