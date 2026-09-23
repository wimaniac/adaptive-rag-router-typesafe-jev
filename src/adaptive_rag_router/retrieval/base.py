"""Khai báo contract retrieval, registry và các hàm hợp nhất evidence dùng chung."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Mapping
from typing import Protocol, runtime_checkable
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from adaptive_rag_router.domain.enums import RetrievalSource
from adaptive_rag_router.domain.errors import EvidenceUnavailableError
from adaptive_rag_router.domain.models import RetrievalResult, RetrievedDocument

_TRACKING_QUERY_PREFIXES = ("utm_",)
_TRACKING_QUERY_KEYS = {"fbclid", "gclid", "mc_cid", "mc_eid"}


@runtime_checkable
class Retriever(Protocol):
    """Contract bất đồng bộ mà mọi nguồn retrieval phải triển khai."""

    async def retrieve(self, query: str, *, round_index: int = 0) -> RetrievalResult:
        """Tìm evidence cho truy vấn ở một vòng adaptive retrieval.

        Args:
            query: Truy vấn đã được chuẩn hóa hoặc rewrite.
            round_index: Chỉ số vòng retrieval, bắt đầu từ 0.

        Returns:
            Kết quả retrieval đã chuẩn hóa về contract miền.
        """


class RetrieverRegistry:
    """Registry ánh xạ nguồn retrieval sang adapter tương ứng.

    `RetrievalSource.NONE` không được đăng ký vì pipeline tự xử lý nhánh không
    retrieval. Registry không tự fallback để orchestration giữ quyền quyết định.
    """

    def __init__(self, retrievers: Mapping[RetrievalSource, Retriever] | None = None) -> None:
        """Khởi tạo registry từ các adapter tùy chọn.

        Args:
            retrievers: Ánh xạ nguồn sang adapter ban đầu.
        """

        self._retrievers: dict[RetrievalSource, Retriever] = {}
        for source, retriever in (retrievers or {}).items():
            self.register(source, retriever)

    def register(self, source: RetrievalSource, retriever: Retriever) -> None:
        """Đăng ký hoặc thay thế adapter của một nguồn retrieval.

        Args:
            source: Nguồn `VECTOR` hoặc `WEB`.
            retriever: Adapter thỏa `Retriever` protocol.

        Raises:
            ValueError: Nếu cố đăng ký nguồn `NONE`.
            TypeError: Nếu adapter không thỏa protocol tại runtime.
        """

        if source is RetrievalSource.NONE:
            raise ValueError("RetrievalSource.NONE do pipeline xử lý, không được đăng ký")
        if not isinstance(retriever, Retriever):
            raise TypeError("retriever phải triển khai async retrieve")
        self._retrievers[source] = retriever

    def get(self, source: RetrievalSource) -> Retriever:
        """Lấy adapter đã đăng ký cho một nguồn.

        Args:
            source: Nguồn retrieval cần dùng.

        Returns:
            Adapter đã đăng ký.

        Raises:
            EvidenceUnavailableError: Nếu nguồn là `NONE` hoặc chưa có adapter.
        """

        if source is RetrievalSource.NONE:
            raise EvidenceUnavailableError("Nguồn NONE không có retriever")
        try:
            return self._retrievers[source]
        except KeyError as error:
            raise EvidenceUnavailableError(
                f"Chưa cấu hình retriever cho nguồn {source.value}"
            ) from error


def normalize_query(query: str) -> str:
    """Chuẩn hóa khoảng trắng để tạo cache key và lookup fixture ổn định.

    Args:
        query: Truy vấn đầu vào.

    Returns:
        Truy vấn đã trim và gộp các khoảng trắng liên tiếp.

    Raises:
        ValueError: Nếu truy vấn chỉ chứa khoảng trắng.
    """

    normalized = " ".join(query.split())
    if not normalized:
        raise ValueError("query không được rỗng")
    return normalized


def canonicalize_url(url: str) -> str:
    """Chuẩn hóa URL và loại tracking parameters để phát hiện evidence trùng.

    Args:
        url: URL nguồn của passage.

    Returns:
        URL canonical; chuỗi rỗng nếu đầu vào rỗng.
    """

    stripped = url.strip()
    if not stripped:
        return ""
    parts = urlsplit(stripped)
    host = (parts.hostname or "").lower()
    if parts.port is not None:
        host = f"{host}:{parts.port}"
    filtered_query = [
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if key.lower() not in _TRACKING_QUERY_KEYS
        and not key.lower().startswith(_TRACKING_QUERY_PREFIXES)
    ]
    path = re.sub(r"/{2,}", "/", parts.path or "/")
    if path != "/":
        path = path.rstrip("/")
    return urlunsplit((parts.scheme.lower(), host, path, urlencode(filtered_query), ""))


def deduplicate_documents(
    documents: Iterable[RetrievedDocument],
    limit: int | None = None,
) -> tuple[RetrievedDocument, ...]:
    """Loại passage trùng theo URL, document ID hoặc nội dung.

    Passage có `provider_score` cao hơn được ưu tiên, nhưng vẫn giữ vị trí đầu
    tiên để thứ tự giữa các retrieval round có tính quyết định và tái lập.

    Args:
        documents: Chuỗi passage theo thứ tự ưu tiên.
        limit: Số passage tối đa sau deduplicate; `None` nghĩa là không giới hạn.

    Returns:
        Tuple passage duy nhất, đã hợp nhất metadata của các bản trùng.

    Raises:
        ValueError: Nếu `limit` không dương.
    """

    if limit is not None and limit <= 0:
        raise ValueError("limit phải lớn hơn 0")

    merged: list[RetrievedDocument] = []
    key_sets: list[set[str]] = []
    for candidate in documents:
        candidate_keys = _document_keys(candidate)
        matching = [
            index for index, existing_keys in enumerate(key_sets) if existing_keys & candidate_keys
        ]
        if not matching:
            merged.append(candidate)
            key_sets.append(candidate_keys)
            continue

        target_index = matching[0]
        combined = merged[target_index]
        combined_keys = set(key_sets[target_index])
        for index in matching[1:]:
            combined = _merge_document_pair(combined, merged[index])
            combined_keys.update(key_sets[index])
        combined = _merge_document_pair(combined, candidate)
        combined_keys.update(candidate_keys)
        merged[target_index] = combined
        key_sets[target_index] = combined_keys

        for index in reversed(matching[1:]):
            del merged[index]
            del key_sets[index]

    return tuple(merged if limit is None else merged[:limit])


def merge_documents(
    results: Iterable[RetrievalResult],
    limit: int,
) -> tuple[RetrievedDocument, ...]:
    """Gộp evidence từ nhiều retrieval round rồi deduplicate.

    Args:
        results: Kết quả retrieval theo thứ tự round.
        limit: Số passage tối đa trả về.

    Returns:
        Tuple passage duy nhất theo thứ tự ưu tiên ổn định.
    """

    return deduplicate_documents(
        (document for result in results for document in result.documents),
        limit=limit,
    )


def _document_keys(document: RetrievedDocument) -> set[str]:
    normalized_text = " ".join(document.text.casefold().split())
    keys = {
        f"id:{document.document_id.casefold()}",
        f"text:{hashlib.sha256(normalized_text.encode('utf-8')).hexdigest()}",
    }
    if document.url:
        canonical_url = canonicalize_url(document.url)
        if canonical_url:
            keys.add(f"url:{canonical_url}")
    return keys


def _merge_document_pair(
    first: RetrievedDocument,
    second: RetrievedDocument,
) -> RetrievedDocument:
    first_score = first.provider_score if first.provider_score is not None else float("-inf")
    second_score = second.provider_score if second.provider_score is not None else float("-inf")
    preferred, other = (second, first) if second_score > first_score else (first, second)
    metadata = {**other.metadata, **preferred.metadata}
    return preferred.model_copy(
        update={
            "title": preferred.title or other.title,
            "url": preferred.url or other.url,
            "metadata": metadata,
        }
    )
