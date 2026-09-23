"""Kiểm thử retriever in-memory và mock web fixture."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from adaptive_rag_router.domain.enums import RetrievalSource
from adaptive_rag_router.domain.models import RetrievedDocument
from adaptive_rag_router.retrieval import (
    InMemoryRetriever,
    JsonlMockWebRetriever,
    MockWebRetriever,
)


def _document(document_id: str, text: str) -> RetrievedDocument:
    return RetrievedDocument(
        document_id=document_id,
        text=text,
        source=RetrievalSource.VECTOR,
    )


@pytest.mark.asyncio
async def test_in_memory_retriever_ranks_lexical_match_first() -> None:
    retriever = InMemoryRetriever(
        (
            _document("irrelevant", "weather tomorrow"),
            _document("relevant", "adaptive retrieval router"),
        )
    )

    result = await retriever.retrieve("adaptive router")

    assert result.cached is True
    assert result.documents[0].document_id == "relevant"
    assert result.documents[0].provider_score is not None


@pytest.mark.asyncio
async def test_mock_web_normalizes_query_and_forces_web_source() -> None:
    retriever = MockWebRetriever(
        {
            "What is RAG?": [
                {
                    "document_id": "rag",
                    "text": "Retrieval augmented generation.",
                    "source": "vector",
                }
            ]
        }
    )

    result = await retriever.retrieve("  what   IS rag? ", round_index=2)

    assert result.provider == "mock-web"
    assert result.cached is True
    assert result.round_index == 2
    assert result.documents[0].source is RetrievalSource.WEB


@pytest.mark.asyncio
async def test_mock_web_uses_wildcard_and_deduplicates() -> None:
    shared = {
        "document_id": "fallback",
        "text": "Fallback evidence",
        "source": "web",
    }
    retriever = MockWebRetriever({"*": [shared, {**shared, "document_id": "duplicate"}]})

    result = await retriever.retrieve("unknown")

    assert len(result.documents) == 1
    assert result.documents[0].document_id == "fallback"


def test_mock_web_rejects_more_than_five_results() -> None:
    with pytest.raises(ValueError, match=r"1\.\.5"):
        MockWebRetriever({}, max_results=6)


@pytest.mark.asyncio
async def test_jsonl_mock_web_uses_persisted_offset_index(tmp_path: Path) -> None:
    fixture = tmp_path / "mock.jsonl"
    fixture.write_text(
        "\n".join(
            json.dumps(record)
            for record in (
                {
                    "query": "First question",
                    "results": [{"document_id": "first", "text": "First evidence"}],
                },
                {
                    "query": "Second question",
                    "results": [{"document_id": "second", "text": "Second evidence"}],
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )

    retriever = JsonlMockWebRetriever(fixture)
    result = await retriever.retrieve("  SECOND   question ")
    reloaded = JsonlMockWebRetriever(fixture)
    repeated = await reloaded.retrieve("Second question")

    assert result.documents[0].document_id == "second"
    assert repeated.documents[0].text == "Second evidence"
    assert fixture.with_suffix(".jsonl.index.json").is_file()
