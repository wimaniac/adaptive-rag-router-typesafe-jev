"""Triển khai vector retrieval và ingestion trên Qdrant với dependency injection."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any, cast
from uuid import NAMESPACE_URL, uuid5

from adaptive_rag_router.domain.enums import RetrievalSource
from adaptive_rag_router.domain.errors import ProviderError
from adaptive_rag_router.domain.models import RetrievalResult, RetrievedDocument
from adaptive_rag_router.retrieval.base import deduplicate_documents, normalize_query
from adaptive_rag_router.retrieval.chunking import (
    HuggingFaceTokenCodec,
    SourceDocument,
    TextChunker,
)
from adaptive_rag_router.retrieval.embeddings import BgeM3EmbeddingAdapter, EmbeddingAdapter

ObjectFactory = Callable[..., object]


class QdrantVectorRetriever:
    """Vector retriever Qdrant dùng embedding adapter bất đồng bộ.

    Client và các factory Qdrant đều có thể inject, nhờ đó unit test không cần
    daemon, model hoặc filesystem database thật.
    """

    def __init__(
        self,
        *,
        collection_name: str,
        embedder: EmbeddingAdapter | None = None,
        client: object | None = None,
        path: Path | str = Path("data/qdrant"),
        top_k: int = 12,
        evidence_limit: int = 8,
        client_factory: Callable[[Path], object] | None = None,
        vector_config_factory: Callable[[int], object] | None = None,
        point_factory: Callable[[str, Sequence[float], Mapping[str, Any]], object] | None = None,
        default_chunker: TextChunker | None = None,
    ) -> None:
        """Khởi tạo Qdrant adapter mà chưa mở database hoặc tải model.

        Args:
            collection_name: Tên collection Qdrant.
            embedder: Adapter embedding; mặc định là BGE-M3 lazy-load.
            client: Qdrant client đã tạo sẵn.
            path: Đường dẫn Qdrant local khi không inject client.
            top_k: Số point lấy từ Qdrant trước deduplicate.
            evidence_limit: Số passage tối đa trả về pipeline.
            client_factory: Factory client tùy chọn cho test.
            vector_config_factory: Factory `VectorParams` tùy chọn.
            point_factory: Factory `PointStruct` tùy chọn.
            default_chunker: Chunker mặc định; nếu thiếu dùng tokenizer BGE-M3
                với cửa sổ 512 token và overlap 64.

        Raises:
            ValueError: Nếu tên collection rỗng hoặc limit không hợp lệ.
        """

        if not collection_name.strip():
            raise ValueError("collection_name không được rỗng")
        if top_k <= 0 or evidence_limit <= 0:
            raise ValueError("top_k và evidence_limit phải lớn hơn 0")
        self._collection_name = collection_name
        self._embedder = embedder or BgeM3EmbeddingAdapter()
        self._client = client
        self._path = Path(path)
        self._top_k = top_k
        self._evidence_limit = evidence_limit
        self._client_factory = client_factory or _default_client_factory
        self._vector_config_factory = vector_config_factory or _default_vector_config_factory
        self._point_factory = point_factory or _default_point_factory
        self._default_chunker = default_chunker or TextChunker(
            chunk_size=512,
            overlap=64,
            codec=HuggingFaceTokenCodec(),
        )

    async def retrieve(self, query: str, *, round_index: int = 0) -> RetrievalResult:
        """Embed truy vấn và tìm các passage gần nhất trong Qdrant.

        Args:
            query: Truy vấn tìm kiếm.
            round_index: Chỉ số retrieval round.

        Returns:
            Tối đa `evidence_limit` passage đã deduplicate.

        Raises:
            ProviderError: Nếu Qdrant không thể thực thi truy vấn.
        """

        started = perf_counter()
        normalized_query = normalize_query(query)
        vector = await self._embedder.embed_query(normalized_query)
        try:
            points = await asyncio.to_thread(self._query_points, vector)
            documents = tuple(
                document for point in points if (document := _point_to_document(point)) is not None
            )
        except ProviderError:
            raise
        except Exception as error:
            raise ProviderError(f"Qdrant retrieval thất bại ({type(error).__name__})") from error
        return RetrievalResult(
            source=RetrievalSource.VECTOR,
            query=normalized_query,
            documents=deduplicate_documents(documents, limit=self._evidence_limit),
            latency_ms=(perf_counter() - started) * 1_000,
            cached=False,
            provider="qdrant",
            round_index=round_index,
        )

    async def close(self) -> None:
        """Đóng Qdrant local client nếu runtime đã lazy-load nó."""

        if self._client is None:
            return
        close_method = getattr(self._client, "close", None)
        if callable(close_method):
            await asyncio.to_thread(close_method)

    async def ingest_documents(
        self,
        documents: Iterable[SourceDocument],
        *,
        chunker: TextChunker | None = None,
        recreate: bool = False,
        batch_size: int = 64,
    ) -> int:
        """Chunk, embed và upsert tài liệu vào collection Qdrant.

        Args:
            documents: Tài liệu thô cần ingest.
            chunker: Chunker tùy chọn; mặc định 512 token, overlap 64.
            recreate: Xóa collection hiện tại trước ingest khi caller yêu cầu rõ.
            batch_size: Số passage embed và upsert trong mỗi batch.

        Returns:
            Tổng số passage đã upsert.

        Raises:
            ValueError: Nếu batch size không dương hoặc không tạo được passage.
            ProviderError: Nếu thao tác Qdrant thất bại.
        """

        if batch_size <= 0:
            raise ValueError("batch_size phải lớn hơn 0")
        active_chunker = chunker or self._default_chunker
        passages = active_chunker.chunk_documents(documents, source=RetrievalSource.VECTOR)
        if not passages:
            return 0

        try:
            first_vectors = await self._embedder.embed_documents(
                tuple(passage.text for passage in passages[:batch_size])
            )
            if not first_vectors or not first_vectors[0]:
                raise ValueError("embedding adapter trả vector rỗng")
            await asyncio.to_thread(
                self._ensure_collection,
                len(first_vectors[0]),
                recreate,
            )
            await asyncio.to_thread(
                self._upsert_batch,
                passages[:batch_size],
                first_vectors,
            )
            for start in range(batch_size, len(passages), batch_size):
                batch = passages[start : start + batch_size]
                vectors = await self._embedder.embed_documents(
                    tuple(passage.text for passage in batch)
                )
                await asyncio.to_thread(self._upsert_batch, batch, vectors)
        except (ProviderError, ValueError):
            raise
        except Exception as error:
            raise ProviderError(f"Qdrant ingest thất bại ({type(error).__name__})") from error
        return len(passages)

    def _get_client(self) -> object:
        if self._client is None:
            self._client = self._client_factory(self._path)
        return self._client

    def _query_points(self, vector: Sequence[float]) -> tuple[object, ...]:
        client = self._get_client()
        query_method = getattr(client, "query_points", None)
        if callable(query_method):
            response = query_method(
                collection_name=self._collection_name,
                query=list(vector),
                limit=self._top_k,
                with_payload=True,
            )
            raw_points = getattr(response, "points", response)
        else:
            search_method = _required_method(client, "search")
            raw_points = search_method(
                collection_name=self._collection_name,
                query_vector=list(vector),
                limit=self._top_k,
                with_payload=True,
            )
        if not isinstance(raw_points, Sequence):
            raise ProviderError("Qdrant trả danh sách point không hợp lệ")
        return tuple(cast(Sequence[object], raw_points))

    def _ensure_collection(self, vector_size: int, recreate: bool) -> None:
        client = self._get_client()
        exists_method = _required_method(client, "collection_exists")
        exists = bool(exists_method(collection_name=self._collection_name))
        if exists and recreate:
            _required_method(client, "delete_collection")(collection_name=self._collection_name)
            exists = False
        if not exists:
            _required_method(client, "create_collection")(
                collection_name=self._collection_name,
                vectors_config=self._vector_config_factory(vector_size),
            )

    def _upsert_batch(
        self,
        passages: Sequence[RetrievedDocument],
        vectors: Sequence[Sequence[float]],
    ) -> None:
        if len(passages) != len(vectors):
            raise ValueError("số vector không khớp số passage")
        points = [
            self._point_factory(
                str(uuid5(NAMESPACE_URL, passage.document_id)),
                vector,
                _document_payload(passage),
            )
            for passage, vector in zip(passages, vectors, strict=True)
        ]
        _required_method(self._get_client(), "upsert")(
            collection_name=self._collection_name,
            points=points,
            wait=True,
        )


def _required_method(target: object, name: str) -> ObjectFactory:
    method = getattr(target, name, None)
    if not callable(method):
        raise ProviderError(f"Qdrant client thiếu method {name}")
    return cast(ObjectFactory, method)


def _point_to_document(point: object) -> RetrievedDocument | None:
    raw_payload = getattr(point, "payload", None)
    if not isinstance(raw_payload, Mapping):
        return None
    payload = cast(Mapping[str, Any], raw_payload)
    text = payload.get("text")
    if not isinstance(text, str) or not text.strip():
        return None
    raw_id = payload.get("document_id", getattr(point, "id", ""))
    score = getattr(point, "score", None)
    metadata = payload.get("metadata", {})
    return RetrievedDocument(
        document_id=str(raw_id),
        text=text,
        title=payload.get("title") if isinstance(payload.get("title"), str) else None,
        url=payload.get("url") if isinstance(payload.get("url"), str) else None,
        source=RetrievalSource.VECTOR,
        provider_score=float(score) if isinstance(score, int | float) else None,
        metadata=dict(metadata) if isinstance(metadata, Mapping) else {},
    )


def _document_payload(document: RetrievedDocument) -> dict[str, Any]:
    return {
        "document_id": document.document_id,
        "text": document.text,
        "title": document.title,
        "url": document.url,
        "metadata": document.metadata,
    }


def _default_client_factory(path: Path) -> object:
    from qdrant_client import QdrantClient

    path.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(path))


def _default_vector_config_factory(vector_size: int) -> object:
    from qdrant_client.models import Distance, VectorParams

    return VectorParams(size=vector_size, distance=Distance.COSINE)


def _default_point_factory(
    point_id: str,
    vector: Sequence[float],
    payload: Mapping[str, Any],
) -> object:
    from qdrant_client.models import PointStruct

    return PointStruct(id=point_id, vector=list(vector), payload=dict(payload))
