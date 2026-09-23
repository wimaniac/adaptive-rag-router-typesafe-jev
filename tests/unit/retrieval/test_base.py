"""Kiểm thử registry và quy tắc deduplicate/merge evidence."""

from __future__ import annotations

import pytest

from adaptive_rag_router.domain.enums import RetrievalSource
from adaptive_rag_router.domain.errors import EvidenceUnavailableError
from adaptive_rag_router.domain.models import RetrievalResult, RetrievedDocument
from adaptive_rag_router.retrieval import (
    RetrieverRegistry,
    canonicalize_url,
    deduplicate_documents,
    merge_documents,
)


class StubRetriever:
    """Retriever tối thiểu phục vụ registry test."""

    async def retrieve(self, query: str, *, round_index: int = 0) -> RetrievalResult:
        """Trả một kết quả rỗng."""

        return RetrievalResult(
            source=RetrievalSource.VECTOR,
            query=query,
            provider="stub",
            round_index=round_index,
        )


def _document(
    document_id: str,
    text: str,
    *,
    url: str | None = None,
    score: float | None = None,
    metadata: dict[str, object] | None = None,
) -> RetrievedDocument:
    return RetrievedDocument(
        document_id=document_id,
        text=text,
        url=url,
        source=RetrievalSource.WEB,
        provider_score=score,
        metadata=metadata or {},
    )


def test_canonicalize_url_removes_fragment_tracking_and_trailing_slash() -> None:
    assert canonicalize_url("HTTPS://Example.COM/a/?utm_source=x&b=2#frag") == (
        "https://example.com/a?b=2"
    )


def test_deduplicate_prefers_higher_score_and_merges_metadata() -> None:
    first = _document(
        "first",
        "Nội dung cũ",
        url="https://example.com/page?utm_source=test",
        score=0.2,
        metadata={"round": 0, "shared": "old"},
    )
    stronger = _document(
        "second",
        "Nội dung cập nhật",
        url="https://example.com/page",
        score=0.9,
        metadata={"round": 1, "shared": "new"},
    )

    merged = deduplicate_documents((first, stronger))

    assert len(merged) == 1
    assert merged[0].document_id == "second"
    assert merged[0].provider_score == 0.9
    assert merged[0].metadata == {"round": 1, "shared": "new"}


def test_deduplicate_matches_identical_text_with_different_ids() -> None:
    documents = (
        _document("one", "  Same   evidence ", url="https://one.example"),
        _document("two", "same evidence", url="https://two.example"),
    )

    assert len(deduplicate_documents(documents)) == 1


def test_merge_documents_preserves_round_order_and_limit() -> None:
    first = RetrievalResult(
        source=RetrievalSource.WEB,
        query="q",
        documents=(_document("a", "A"), _document("b", "B")),
        provider="mock",
    )
    second = RetrievalResult(
        source=RetrievalSource.WEB,
        query="q2",
        documents=(_document("a2", "A"), _document("c", "C")),
        provider="mock",
        round_index=1,
    )

    assert [item.document_id for item in merge_documents((first, second), 2)] == ["a", "b"]


def test_registry_returns_registered_adapter_and_rejects_none() -> None:
    retriever = StubRetriever()
    registry = RetrieverRegistry({RetrievalSource.VECTOR: retriever})

    assert registry.get(RetrievalSource.VECTOR) is retriever
    with pytest.raises(EvidenceUnavailableError):
        registry.get(RetrievalSource.NONE)
