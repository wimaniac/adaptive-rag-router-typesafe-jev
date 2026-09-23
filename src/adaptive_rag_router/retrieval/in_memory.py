"""Cung cấp retriever in-memory và mock web deterministic cho test, replay và demo."""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from time import perf_counter
from typing import Any

from adaptive_rag_router.domain.enums import RetrievalSource
from adaptive_rag_router.domain.models import RetrievalResult, RetrievedDocument
from adaptive_rag_router.retrieval.base import deduplicate_documents, normalize_query

DocumentScorer = Callable[[str, RetrievedDocument], float]
DocumentFixture = RetrievedDocument | Mapping[str, Any]


class InMemoryRetriever:
    """Retriever lexical nhỏ gọn dùng cho unit test và corpus demo.

    Adapter này không thay thế vector database trong benchmark chính. Scorer có
    thể được inject để test orchestration mà không tải embedding model.
    """

    def __init__(
        self,
        documents: Iterable[RetrievedDocument],
        *,
        source: RetrievalSource = RetrievalSource.VECTOR,
        result_limit: int = 8,
        scorer: DocumentScorer | None = None,
        provider: str = "in-memory",
    ) -> None:
        """Khởi tạo retriever từ tập passage có sẵn.

        Args:
            documents: Passage được giữ trong bộ nhớ.
            source: Nguồn logic gắn vào kết quả.
            result_limit: Số passage tối đa cho mỗi truy vấn.
            scorer: Hàm chấm điểm tùy chọn.
            provider: Tên provider ghi vào trace.

        Raises:
            ValueError: Nếu nguồn là `NONE` hoặc limit không dương.
        """

        if source is RetrievalSource.NONE:
            raise ValueError("InMemoryRetriever không hỗ trợ nguồn NONE")
        if result_limit <= 0:
            raise ValueError("result_limit phải lớn hơn 0")
        self._source = source
        self._result_limit = result_limit
        self._scorer = scorer or _lexical_score
        self._provider = provider
        self._documents = tuple(
            document.model_copy(update={"source": source}) for document in documents
        )

    async def retrieve(self, query: str, *, round_index: int = 0) -> RetrievalResult:
        """Xếp hạng passage bằng scorer in-memory.

        Args:
            query: Truy vấn tìm kiếm.
            round_index: Chỉ số retrieval round.

        Returns:
            Kết quả deterministic, không tiêu network hoặc credit.
        """

        started = perf_counter()
        normalized_query = normalize_query(query)
        scored_documents = [
            (index, document, self._scorer(normalized_query, document))
            for index, document in enumerate(self._documents)
        ]
        ranked = sorted(
            scored_documents,
            key=lambda item: (-item[2], item[0]),
        )
        scored = tuple(
            document.model_copy(update={"provider_score": score})
            for _, document, score in ranked[: self._result_limit]
        )
        return RetrievalResult(
            source=self._source,
            query=normalized_query,
            documents=deduplicate_documents(scored, limit=self._result_limit),
            latency_ms=(perf_counter() - started) * 1_000,
            cached=True,
            provider=self._provider,
            round_index=round_index,
        )


class MockWebRetriever:
    """Web retriever phát lại fixture theo normalized query, không gọi mạng."""

    def __init__(
        self,
        fixtures: Mapping[str, Sequence[DocumentFixture]],
        *,
        max_results: int = 5,
        provider: str = "mock-web",
    ) -> None:
        """Khởi tạo mock web từ mapping query sang danh sách passage.

        Key `"*"` có thể được dùng làm fixture mặc định khi query không khớp.

        Args:
            fixtures: Mapping query sang documents hoặc dictionaries hợp lệ.
            max_results: Số kết quả tối đa, không vượt giới hạn web MVP là 5.
            provider: Tên provider ghi vào trace.

        Raises:
            ValueError: Nếu `max_results` nằm ngoài khoảng 1..5.
        """

        if not 1 <= max_results <= 5:
            raise ValueError("max_results phải nằm trong khoảng 1..5")
        self._max_results = max_results
        self._provider = provider
        self._fixtures = {
            _fixture_key(query): tuple(_coerce_web_document(document) for document in documents)
            for query, documents in fixtures.items()
        }

    async def retrieve(self, query: str, *, round_index: int = 0) -> RetrievalResult:
        """Trả fixture của truy vấn mà không gọi provider live.

        Args:
            query: Truy vấn cần lookup.
            round_index: Chỉ số retrieval round.

        Returns:
            Kết quả mock luôn được đánh dấu `cached=True`.
        """

        started = perf_counter()
        normalized_query = normalize_query(query)
        documents = self._fixtures.get(normalized_query.casefold(), self._fixtures.get("*", ()))
        return RetrievalResult(
            source=RetrievalSource.WEB,
            query=normalized_query,
            documents=deduplicate_documents(documents, limit=self._max_results),
            latency_ms=(perf_counter() - started) * 1_000,
            cached=True,
            provider=self._provider,
            round_index=round_index,
        )


def _fixture_key(query: str) -> str:
    return "*" if query == "*" else normalize_query(query).casefold()


def _coerce_web_document(document: DocumentFixture) -> RetrievedDocument:
    if isinstance(document, RetrievedDocument):
        return document.model_copy(update={"source": RetrievalSource.WEB})
    payload = dict(document)
    payload["source"] = RetrievalSource.WEB
    return RetrievedDocument.model_validate(payload)


def _lexical_score(query: str, document: RetrievedDocument) -> float:
    query_terms = set(re.findall(r"\w+", query.casefold()))
    document_terms = set(re.findall(r"\w+", document.text.casefold()))
    if not query_terms or not document_terms:
        return 0.0
    overlap = len(query_terms & document_terms)
    return overlap / math.sqrt(len(query_terms) * len(document_terms))
