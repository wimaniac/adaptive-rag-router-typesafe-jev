"""Cung cấp embedding BGE-M3 lazy-load, CUDA FP16 và CPU fallback float32."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from threading import Lock
from typing import Protocol, cast


class EmbeddingAdapter(Protocol):
    """Contract embedding bất đồng bộ dùng bởi vector retriever."""

    async def embed_documents(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        """Sinh vector cho một batch văn bản.

        Args:
            texts: Batch văn bản.

        Returns:
            Các vector theo đúng thứ tự đầu vào.
        """

    async def embed_query(self, query: str) -> tuple[float, ...]:
        """Sinh vector cho một truy vấn.

        Args:
            query: Truy vấn cần encode.

        Returns:
            Vector dense đã normalize.
        """


class EmbeddingModel(Protocol):
    """Bề mặt tối thiểu của model SentenceTransformer được adapter sử dụng."""

    def encode(self, sentences: Sequence[str], **kwargs: object) -> object:
        """Encode một batch câu thành ma trận vector."""


EmbeddingModelFactory = Callable[[str, str | None], EmbeddingModel]


class BgeM3EmbeddingAdapter:
    """Adapter lazy-load `BAAI/bge-m3` và chạy encode ngoài event loop."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        *,
        device: str | None = None,
        model_factory: EmbeddingModelFactory | None = None,
    ) -> None:
        """Khởi tạo adapter mà chưa tải model.

        Args:
            model_name: Hugging Face model ID.
            device: `cuda`, `cpu` hoặc `None` để tự phát hiện.
            model_factory: Factory inject cho unit test hoặc runtime tùy biến.
        """

        self._model_name = model_name
        self._requested_device = device
        self._model_factory = model_factory or _default_model_factory
        self._model: EmbeddingModel | None = None
        self._load_lock = Lock()

    @property
    def loaded(self) -> bool:
        """Cho biết model đã được tải vào bộ nhớ hay chưa."""

        return self._model is not None

    async def embed_documents(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        """Sinh normalized embeddings cho một batch văn bản.

        Args:
            texts: Batch văn bản; batch rỗng trả tuple rỗng và không tải model.

        Returns:
            Các vector float bất biến theo thứ tự đầu vào.
        """

        if not texts:
            return ()
        model = await asyncio.to_thread(self._get_model)
        raw = await asyncio.to_thread(
            model.encode,
            list(texts),
            batch_size=len(texts),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        matrix = raw.tolist() if hasattr(raw, "tolist") else raw
        rows = cast(Sequence[Sequence[float]], matrix)
        return tuple(tuple(float(value) for value in row) for row in rows)

    async def embed_query(self, query: str) -> tuple[float, ...]:
        """Sinh normalized embedding cho truy vấn.

        Args:
            query: Truy vấn không rỗng.

        Returns:
            Vector dense của truy vấn.

        Raises:
            ValueError: Nếu truy vấn rỗng.
        """

        normalized = query.strip()
        if not normalized:
            raise ValueError("query không được rỗng")
        return (await self.embed_documents((normalized,)))[0]

    def _get_model(self) -> EmbeddingModel:
        if self._model is not None:
            return self._model
        with self._load_lock:
            if self._model is None:
                self._model = self._model_factory(self._model_name, self._requested_device)
        return self._model


def _default_model_factory(model_name: str, requested_device: str | None) -> EmbeddingModel:
    from sentence_transformers import SentenceTransformer

    device = requested_device or _detect_device()
    try:
        if device == "cuda":
            import torch

            return cast(
                EmbeddingModel,
                SentenceTransformer(
                    model_name,
                    device=device,
                    model_kwargs={"torch_dtype": torch.float16},
                ),
            )
        return cast(EmbeddingModel, SentenceTransformer(model_name, device="cpu"))
    except (RuntimeError, OSError):
        if device != "cuda":
            raise
        return cast(EmbeddingModel, SentenceTransformer(model_name, device="cpu"))


def _detect_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"
