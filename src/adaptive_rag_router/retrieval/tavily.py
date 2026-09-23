"""Triển khai Tavily Basic Search với cache, rate limit, retry và credit hard cap."""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import time
from collections import deque
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from time import perf_counter
from types import TracebackType
from typing import Any, Protocol, cast

import httpx
from pydantic import SecretStr

from adaptive_rag_router.config.settings import Settings
from adaptive_rag_router.domain.enums import RetrievalSource
from adaptive_rag_router.domain.errors import (
    ConfigurationError,
    CreditBudgetExceededError,
    ProviderError,
    ProviderRateLimitError,
)
from adaptive_rag_router.domain.models import RetrievalResult, RetrievedDocument
from adaptive_rag_router.retrieval.base import deduplicate_documents, normalize_query

AsyncSleeper = Callable[[float], Awaitable[None]]
AttemptGuard = Callable[[], None]


class SearchCache(Protocol):
    """Contract cache đồng bộ nhỏ cho raw Tavily response."""

    def get(self, key: str) -> Mapping[str, Any] | None:
        """Đọc response theo cache key hoặc trả `None` khi miss."""

    def set(self, key: str, value: Mapping[str, Any]) -> None:
        """Lưu raw response JSON-safe theo cache key."""


class TavilySearchTransport(Protocol):
    """Contract transport live tối thiểu để inject fake trong unit test."""

    async def search(self, query: str, *, max_results: int) -> Mapping[str, Any]:
        """Gọi Tavily Basic Search và trả response đã parse."""


class InMemorySearchCache:
    """Cache in-memory deterministic dành cho test và process ngắn."""

    def __init__(self) -> None:
        """Khởi tạo cache rỗng."""

        self._values: dict[str, dict[str, Any]] = {}

    def get(self, key: str) -> Mapping[str, Any] | None:
        """Đọc bản sao nông của response đã cache.

        Args:
            key: Cache key SHA-256.

        Returns:
            Response đã lưu hoặc `None`.
        """

        value = self._values.get(key)
        return None if value is None else dict(value)

    def set(self, key: str, value: Mapping[str, Any]) -> None:
        """Lưu bản sao response.

        Args:
            key: Cache key SHA-256.
            value: Raw Tavily response JSON-safe.
        """

        self._values[key] = dict(value)


class FileSearchCache:
    """Cache response Tavily theo file JSON để replay benchmark."""

    def __init__(self, directory: Path | str) -> None:
        """Khởi tạo cache tại thư mục chỉ định.

        Args:
            directory: Thư mục chứa mỗi response dưới một file JSON.
        """

        self._directory = Path(directory)

    def get(self, key: str) -> Mapping[str, Any] | None:
        """Đọc response JSON nếu cache hit.

        Args:
            key: Cache key SHA-256.

        Returns:
            Mapping response hoặc `None` nếu thiếu/hỏng cache.
        """

        path = self._path_for(key)
        if not path.exists():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return cast(dict[str, Any], value) if isinstance(value, dict) else None

    def set(self, key: str, value: Mapping[str, Any]) -> None:
        """Ghi response JSON theo kiểu atomic replace.

        Args:
            key: Cache key SHA-256.
            value: Raw response JSON-safe.
        """

        self._directory.mkdir(parents=True, exist_ok=True)
        target = self._path_for(key)
        temporary = target.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(dict(value), ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(target)

    def _path_for(self, key: str) -> Path:
        safe_key = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self._directory / f"{safe_key}.json"


class RollingWindowRateLimiter:
    """Rate limiter cửa sổ trượt kết hợp giới hạn concurrency.

    Một timestamp chỉ được ghi khi request thực sự được phép bắt đầu. Semaphore
    được giữ trong thời gian provider call và được nhả trước thời gian backoff.
    """

    def __init__(
        self,
        *,
        max_requests: int = 90,
        window_seconds: float = 60.0,
        max_concurrency: int = 4,
        clock: Callable[[], float] = time.monotonic,
        sleep: AsyncSleeper = asyncio.sleep,
    ) -> None:
        """Khởi tạo limiter.

        Args:
            max_requests: Số request tối đa trong một cửa sổ.
            window_seconds: Độ dài cửa sổ trượt tính bằng giây.
            max_concurrency: Số provider call đồng thời tối đa.
            clock: Đồng hồ monotonic inject cho test.
            sleep: Coroutine sleep inject cho test.

        Raises:
            ValueError: Nếu cấu hình không dương hoặc vượt guardrail 90/60s.
        """

        if max_requests <= 0 or max_requests > 90:
            raise ValueError("max_requests phải nằm trong khoảng 1..90")
        if window_seconds <= 0:
            raise ValueError("window_seconds phải lớn hơn 0")
        if max_concurrency <= 0 or max_concurrency > 4:
            raise ValueError("max_concurrency phải nằm trong khoảng 1..4")
        self._max_requests = max_requests
        self._window_seconds = window_seconds
        self._clock = clock
        self._sleep = sleep
        self._timestamps: deque[float] = deque()
        self._window_lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def acquire(self) -> None:
        """Chờ đến khi còn cả concurrency slot và rate-window slot."""

        await self._semaphore.acquire()
        try:
            await self._acquire_window_slot()
        except BaseException:
            self._semaphore.release()
            raise

    def release(self) -> None:
        """Nhả concurrency slot sau khi provider call hoàn tất."""

        self._semaphore.release()

    async def __aenter__(self) -> RollingWindowRateLimiter:
        """Giữ một request slot trong async context."""

        await self.acquire()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Nhả concurrency slot dù provider thành công hay thất bại."""

        self.release()

    async def _acquire_window_slot(self) -> None:
        while True:
            async with self._window_lock:
                now = self._clock()
                cutoff = now - self._window_seconds
                while self._timestamps and self._timestamps[0] <= cutoff:
                    self._timestamps.popleft()
                if len(self._timestamps) < self._max_requests:
                    self._timestamps.append(now)
                    return
                wait_seconds = max(
                    0.0,
                    self._window_seconds - (now - self._timestamps[0]),
                )
            await self._sleep(wait_seconds)


class CreditLedger:
    """Sổ credit thread-safe chặn request trước khi vượt hard cap."""

    def __init__(self, budget: int, *, checkpoint_path: Path | str | None = None) -> None:
        """Khởi tạo ledger và phục hồi checkpoint nếu có.

        Args:
            budget: Hard cap credits của một live run.
            checkpoint_path: File JSON lưu số credit đã tiêu.

        Raises:
            ValueError: Nếu budget không dương.
            ConfigurationError: Nếu checkpoint có budget khác hoặc không hợp lệ.
        """

        if budget <= 0:
            raise ValueError("credit budget phải lớn hơn 0")
        self._budget = budget
        self._checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self._consumed = 0
        self._lock = asyncio.Lock()
        self._load_checkpoint()

    @property
    def budget(self) -> int:
        """Trả hard cap credit của run."""

        return self._budget

    @property
    def consumed(self) -> int:
        """Trả số credit đã dành cho provider attempts."""

        return self._consumed

    @property
    def remaining(self) -> int:
        """Trả số credit còn lại trước hard cap."""

        return self._budget - self._consumed

    async def consume(self, credits: int = 1) -> None:
        """Dành credit trước khi bắt đầu provider call.

        Args:
            credits: Số credit của attempt, Basic Search mặc định là 1.

        Raises:
            ValueError: Nếu credits không dương.
            CreditBudgetExceededError: Nếu attempt sẽ vượt hard cap.
        """

        if credits <= 0:
            raise ValueError("credits phải lớn hơn 0")
        async with self._lock:
            if self._consumed + credits > self._budget:
                raise CreditBudgetExceededError(
                    f"Tavily credit budget đã hết ({self._consumed}/{self._budget})"
                )
            self._consumed += credits
            self._write_checkpoint()

    def _load_checkpoint(self) -> None:
        if self._checkpoint_path is None or not self._checkpoint_path.exists():
            return
        try:
            payload = json.loads(self._checkpoint_path.read_text(encoding="utf-8"))
            checkpoint_budget = int(payload["budget"])
            checkpoint_consumed = int(payload["consumed"])
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
            raise ConfigurationError("Tavily credit checkpoint không hợp lệ") from error
        if checkpoint_budget != self._budget:
            raise ConfigurationError("Tavily credit checkpoint có budget khác cấu hình hiện tại")
        if not 0 <= checkpoint_consumed <= self._budget:
            raise ConfigurationError("Số credit trong checkpoint không hợp lệ")
        self._consumed = checkpoint_consumed

    def _write_checkpoint(self) -> None:
        if self._checkpoint_path is None:
            return
        self._checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._checkpoint_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(
                {
                    "budget": self._budget,
                    "consumed": self._consumed,
                    "updated_at": datetime.now(UTC).isoformat(),
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        temporary.replace(self._checkpoint_path)


class TavilyRateLimitError(ProviderRateLimitError):
    """Lỗi 429 nội bộ có kèm thời gian Retry-After đã parse."""

    def __init__(self, retry_after_seconds: float | None = None) -> None:
        """Khởi tạo lỗi rate limit.

        Args:
            retry_after_seconds: Số giây provider yêu cầu chờ, nếu có.
        """

        super().__init__("Tavily trả HTTP 429")
        self.retry_after_seconds = retry_after_seconds


class HttpxTavilyTransport:
    """Transport HTTP gọi duy nhất endpoint Tavily Basic Search."""

    def __init__(
        self,
        api_key: SecretStr | str,
        *,
        timeout_seconds: float = 60.0,
        client: httpx.AsyncClient | None = None,
        base_url: str = "https://api.tavily.com",
    ) -> None:
        """Khởi tạo transport live.

        Args:
            api_key: Tavily API key, được giữ ngoài payload/log công khai.
            timeout_seconds: Timeout cho mỗi HTTP attempt.
            client: HTTP client inject; transport chỉ đóng client do nó tự tạo.
            base_url: Tavily API base URL.

        Raises:
            ConfigurationError: Nếu API key rỗng.
        """

        secret = api_key.get_secret_value() if isinstance(api_key, SecretStr) else api_key
        if not secret.strip():
            raise ConfigurationError("Thiếu TAVILY_API_KEY")
        self._api_key = secret
        self._base_url = base_url.rstrip("/")
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=timeout_seconds)

    async def search(self, query: str, *, max_results: int) -> Mapping[str, Any]:
        """Gọi Tavily Search ở chế độ `basic`.

        Args:
            query: Truy vấn đã chuẩn hóa.
            max_results: Số kết quả tối đa, không vượt 5.

        Returns:
            Raw JSON response.

        Raises:
            TavilyRateLimitError: Khi provider trả HTTP 429.
            ProviderError: Với lỗi network, HTTP khác hoặc JSON sai dạng.
        """

        try:
            response = await self._client.post(
                f"{self._base_url}/search",
                json={
                    "api_key": self._api_key,
                    "query": query,
                    "search_depth": "basic",
                    "max_results": max_results,
                    "include_answer": False,
                    "include_raw_content": False,
                },
            )
        except httpx.HTTPError as error:
            raise ProviderError(f"Tavily network lỗi ({type(error).__name__})") from error
        if response.status_code == 429:
            raise TavilyRateLimitError(_parse_retry_after(response.headers.get("Retry-After")))
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            raise ProviderError(f"Tavily trả HTTP {response.status_code}") from error
        try:
            payload = response.json()
        except ValueError as error:
            raise ProviderError("Tavily trả JSON không hợp lệ") from error
        if not isinstance(payload, dict):
            raise ProviderError("Tavily response phải là JSON object")
        return cast(dict[str, Any], payload)

    async def close(self) -> None:
        """Đóng HTTP client nếu transport là bên tạo client."""

        if self._owns_client:
            await self._client.aclose()


class TavilyRetriever:
    """Live web retriever tuân thủ guardrail của Tavily free tier."""

    def __init__(
        self,
        transport: TavilySearchTransport,
        *,
        cache: SearchCache | None = None,
        limiter: RollingWindowRateLimiter | None = None,
        credit_ledger: CreditLedger | None = None,
        max_results: int = 5,
        max_retries: int = 4,
        retry_base_seconds: float = 1.0,
        sleep: AsyncSleeper = asyncio.sleep,
        random_value: Callable[[], float] = random.random,
        attempt_guard: AttemptGuard | None = None,
    ) -> None:
        """Khởi tạo retriever live từ các dependency có thể thay thế.

        Args:
            transport: Transport Tavily thật hoặc fake test.
            cache: Cache được kiểm tra trước mọi provider attempt.
            limiter: Limiter dùng chung giữa các worker.
            credit_ledger: Ledger dùng chung của live benchmark run.
            max_results: Số kết quả tối đa, hard cap là 5.
            max_retries: Số lần retry sau HTTP 429, hard cap là 4.
            retry_base_seconds: Backoff cơ sở.
            sleep: Coroutine sleep inject cho test.
            random_value: Nguồn jitter trong khoảng [0, 1].
            attempt_guard: Guard đồng bộ chạy trước khi dành global credit,
                dùng để áp hard cap theo từng query/workflow.

        Raises:
            ValueError: Nếu cấu hình vượt guardrail MVP.
        """

        if not 1 <= max_results <= 5:
            raise ValueError("max_results phải nằm trong khoảng 1..5")
        if not 0 <= max_retries <= 4:
            raise ValueError("max_retries phải nằm trong khoảng 0..4")
        if retry_base_seconds < 0:
            raise ValueError("retry_base_seconds không được âm")
        self._transport = transport
        self._cache = cache or InMemorySearchCache()
        self._limiter = limiter or RollingWindowRateLimiter()
        self._credit_ledger = credit_ledger or CreditLedger(350)
        self._max_results = max_results
        self._max_retries = max_retries
        self._retry_base_seconds = retry_base_seconds
        self._sleep = sleep
        self._random_value = random_value
        self._attempt_guard = attempt_guard

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        checkpoint_path: Path | str | None = None,
        transport: TavilySearchTransport | None = None,
        cache: SearchCache | None = None,
    ) -> TavilyRetriever:
        """Tạo live retriever từ application settings.

        Args:
            settings: Cấu hình ứng dụng đã validate.
            checkpoint_path: Checkpoint credit riêng cho run.
            transport: Transport inject; nếu thiếu sẽ cần Tavily API key.
            cache: Cache inject; mặc định dùng cache file.

        Returns:
            Tavily retriever đã áp guardrail 90 request/phút, concurrency 4.

        Raises:
            ConfigurationError: Nếu không inject transport và thiếu API key.
        """

        selected_transport = transport
        if selected_transport is None:
            if settings.tavily_api_key is None:
                raise ConfigurationError("Thiếu TAVILY_API_KEY cho live retrieval")
            selected_transport = HttpxTavilyTransport(
                settings.tavily_api_key,
                timeout_seconds=settings.request_timeout_seconds,
            )
        selected_cache = cache or FileSearchCache(settings.cache_dir / "tavily")
        selected_checkpoint = checkpoint_path or (
            settings.artifacts_dir / "tavily-credit-checkpoint.json"
        )
        return cls(
            selected_transport,
            cache=selected_cache,
            limiter=RollingWindowRateLimiter(
                max_requests=min(settings.tavily_requests_per_minute, 90),
                window_seconds=60,
                max_concurrency=min(settings.tavily_max_concurrency, 4),
            ),
            credit_ledger=CreditLedger(
                settings.tavily_credit_budget,
                checkpoint_path=selected_checkpoint,
            ),
            max_results=min(settings.web_max_results, 5),
            max_retries=min(settings.tavily_max_retries, 4),
        )

    async def retrieve(self, query: str, *, round_index: int = 0) -> RetrievalResult:
        """Tìm web evidence với cache-before-call và retry có kiểm soát.

        Mỗi provider attempt, kể cả attempt nhận HTTP 429, tiêu một đơn vị trong
        ledger. Cache hit không chiếm request slot và không tiêu credit.

        Args:
            query: Truy vấn web.
            round_index: Chỉ số retrieval round.

        Returns:
            Tối đa năm kết quả web đã deduplicate.

        Raises:
            CreditBudgetExceededError: Nếu attempt tiếp theo vượt hard cap.
            ProviderRateLimitError: Nếu vẫn bị 429 sau tối đa bốn retries.
            ProviderError: Nếu provider hoặc payload gặp lỗi khác.
        """

        started = perf_counter()
        normalized_query = normalize_query(query)
        cache_key = build_tavily_cache_key(normalized_query, self._max_results)
        cached_response = self._cache.get(cache_key)
        if cached_response is not None:
            return self._to_result(
                cached_response,
                normalized_query,
                round_index,
                cached=True,
                started=started,
                external_credits=0,
            )

        response: Mapping[str, Any] | None = None
        attempts_used = 0
        for attempt in range(self._max_retries + 1):
            if self._attempt_guard is not None:
                self._attempt_guard()
            await self._credit_ledger.consume(1)
            attempts_used += 1
            try:
                async with self._limiter:
                    response = await self._transport.search(
                        normalized_query,
                        max_results=self._max_results,
                    )
                break
            except TavilyRateLimitError as error:
                if attempt >= self._max_retries:
                    raise ProviderRateLimitError(
                        f"Tavily vẫn rate-limit sau {attempt + 1} attempts"
                    ) from error
                exponential = self._retry_base_seconds * (2**attempt)
                jitter = self._retry_base_seconds * self._random_value()
                retry_after = error.retry_after_seconds or 0.0
                await self._sleep(max(retry_after, exponential + jitter))
            except (CreditBudgetExceededError, ProviderError):
                raise
            except Exception as error:
                raise ProviderError(
                    f"Tavily retrieval thất bại ({type(error).__name__})"
                ) from error

        if response is None:
            raise ProviderError("Tavily không trả response")
        self._cache.set(cache_key, response)
        return self._to_result(
            response,
            normalized_query,
            round_index,
            cached=False,
            started=started,
            external_credits=attempts_used,
        )

    def _to_result(
        self,
        response: Mapping[str, Any],
        query: str,
        round_index: int,
        *,
        cached: bool,
        started: float,
        external_credits: int,
    ) -> RetrievalResult:
        raw_results = response.get("results", ())
        if not isinstance(raw_results, Sequence) or isinstance(raw_results, str | bytes):
            raise ProviderError("Tavily response.results phải là danh sách")
        documents = tuple(
            document
            for rank, raw in enumerate(raw_results)
            if isinstance(raw, Mapping)
            if (document := _tavily_item_to_document(cast(Mapping[str, Any], raw), rank))
            is not None
        )
        return RetrievalResult(
            source=RetrievalSource.WEB,
            query=query,
            documents=deduplicate_documents(documents, limit=self._max_results),
            latency_ms=(perf_counter() - started) * 1_000,
            cached=cached,
            provider="tavily",
            round_index=round_index,
            external_credits=external_credits,
        )


def build_tavily_cache_key(query: str, max_results: int) -> str:
    """Tạo cache key từ normalized query và tham số Basic Search.

    Args:
        query: Truy vấn đầu vào.
        max_results: Số kết quả yêu cầu.

    Returns:
        SHA-256 hex key ổn định.
    """

    payload = json.dumps(
        {
            "query": normalize_query(query).casefold(),
            "search_depth": "basic",
            "max_results": max_results,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _tavily_item_to_document(
    item: Mapping[str, Any],
    rank: int,
) -> RetrievedDocument | None:
    content = item.get("content")
    if not isinstance(content, str) or not content.strip():
        return None
    url = item.get("url") if isinstance(item.get("url"), str) else None
    title = item.get("title") if isinstance(item.get("title"), str) else None
    identity = url or f"{title or ''}\n{content}"
    score = item.get("score")
    metadata: dict[str, Any] = {"rank": rank}
    if isinstance(item.get("published_date"), str):
        metadata["published_date"] = item["published_date"]
    return RetrievedDocument(
        document_id=f"tavily-{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:24]}",
        text=content.strip(),
        title=title,
        url=url,
        source=RetrievalSource.WEB,
        provider_score=float(score) if isinstance(score, int | float) else None,
        metadata=metadata,
    )


def _parse_retry_after(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=UTC)
        return max(0.0, (retry_at - datetime.now(UTC)).total_seconds())
