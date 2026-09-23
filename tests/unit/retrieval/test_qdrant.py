"""Kiểm thử Qdrant adapter bằng fake client và embedding, không mở database thật."""

from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace

import pytest

from adaptive_rag_router.domain.enums import RetrievalSource
from adaptive_rag_router.retrieval import QdrantVectorRetriever, SourceDocument, TextChunker


class FakeEmbedder:
    """Embedding adapter deterministic cho Qdrant unit test."""

    async def embed_documents(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        """Biểu diễn mỗi text bằng độ dài và một hằng số."""

        return tuple((float(len(text)), 1.0) for text in texts)

    async def embed_query(self, query: str) -> tuple[float, ...]:
        """Biểu diễn query bằng vector hai chiều."""

        return (float(len(query)), 1.0)


class FakeQdrantClient:
    """Fake client ghi nhận collection và upsert points."""

    def __init__(self) -> None:
        self.exists = False
        self.created: object | None = None
        self.upserted: list[object] = []
        self.query: list[float] | None = None

    def collection_exists(self, *, collection_name: str) -> bool:
        """Trả trạng thái collection giả."""

        assert collection_name == "docs"
        return self.exists

    def create_collection(self, *, collection_name: str, vectors_config: object) -> None:
        """Ghi nhận cấu hình collection."""

        assert collection_name == "docs"
        self.exists = True
        self.created = vectors_config

    def upsert(self, *, collection_name: str, points: list[object], wait: bool) -> None:
        """Ghi nhận points đã upsert."""

        assert collection_name == "docs"
        assert wait is True
        self.upserted.extend(points)

    def query_points(
        self,
        *,
        collection_name: str,
        query: list[float],
        limit: int,
        with_payload: bool,
    ) -> object:
        """Trả hai point trùng URL để kiểm tra deduplicate."""

        assert collection_name == "docs"
        assert limit == 12
        assert with_payload is True
        self.query = query
        return SimpleNamespace(
            points=[
                SimpleNamespace(
                    id="one",
                    score=0.4,
                    payload={
                        "document_id": "one",
                        "text": "lower",
                        "url": "https://example.com/a?utm_source=x",
                    },
                ),
                SimpleNamespace(
                    id="two",
                    score=0.9,
                    payload={
                        "document_id": "two",
                        "text": "higher",
                        "url": "https://example.com/a",
                    },
                ),
            ]
        )


@pytest.mark.asyncio
async def test_qdrant_retrieve_maps_points_and_deduplicates() -> None:
    client = FakeQdrantClient()
    retriever = QdrantVectorRetriever(
        collection_name="docs",
        embedder=FakeEmbedder(),
        client=client,
    )

    result = await retriever.retrieve("query")

    assert client.query == [5.0, 1.0]
    assert result.source is RetrievalSource.VECTOR
    assert len(result.documents) == 1
    assert result.documents[0].document_id == "two"


@pytest.mark.asyncio
async def test_qdrant_ingest_chunks_and_upserts_without_real_dependencies() -> None:
    client = FakeQdrantClient()
    retriever = QdrantVectorRetriever(
        collection_name="docs",
        embedder=FakeEmbedder(),
        client=client,
        vector_config_factory=lambda size: {"size": size, "distance": "cosine"},
        point_factory=lambda point_id, vector, payload: {
            "id": point_id,
            "vector": list(vector),
            "payload": dict(payload),
        },
    )

    count = await retriever.ingest_documents(
        (SourceDocument(document_id="parent", text="a b c d e f"),),
        chunker=TextChunker(chunk_size=4, overlap=1),
        batch_size=1,
    )

    assert count == 2
    assert client.created == {"size": 2, "distance": "cosine"}
    assert len(client.upserted) == 2
