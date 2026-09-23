"""Kiểm thử adapter embedding và việc truyền batch xuống model thực tế."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from adaptive_rag_router.retrieval.embeddings import BgeM3EmbeddingAdapter


class RecordingEmbeddingModel:
    """Model giả ghi lại tham số encode để kiểm tra wiring adapter."""

    def __init__(self) -> None:
        self.sentences: list[str] = []
        self.kwargs: dict[str, object] = {}

    def encode(self, sentences: Sequence[str], **kwargs: object) -> list[list[float]]:
        """Ghi lời gọi và trả vector hai chiều xác định."""

        self.sentences = list(sentences)
        self.kwargs = kwargs
        return [[float(index), 1.0] for index, _ in enumerate(sentences)]


@pytest.mark.asyncio
async def test_embedding_adapter_forwards_effective_batch_size() -> None:
    model = RecordingEmbeddingModel()
    adapter = BgeM3EmbeddingAdapter(model_factory=lambda _name, _device: model)

    vectors = await adapter.embed_documents(("one", "two", "three"))

    assert vectors == ((0.0, 1.0), (1.0, 1.0), (2.0, 1.0))
    assert model.sentences == ["one", "two", "three"]
    assert model.kwargs["batch_size"] == 3
    assert model.kwargs["normalize_embeddings"] is True


@pytest.mark.asyncio
async def test_embedding_adapter_does_not_load_model_for_empty_batch() -> None:
    adapter = BgeM3EmbeddingAdapter(
        model_factory=lambda _name, _device: (_ for _ in ()).throw(AssertionError("unexpected"))
    )

    assert await adapter.embed_documents(()) == ()
    assert adapter.loaded is False
