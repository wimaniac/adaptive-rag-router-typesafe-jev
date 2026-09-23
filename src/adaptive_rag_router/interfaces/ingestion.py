"""Nạp tài liệu local và ingest vào Qdrant mà không khởi tạo LLM provider."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from adaptive_rag_router.domain.errors import ConfigurationError
from adaptive_rag_router.evaluation import load_local_records
from adaptive_rag_router.retrieval import QdrantVectorRetriever, SourceDocument

_SUPPORTED_SUFFIXES = {".json", ".jsonl", ".ndjson", ".parquet"}


def load_source_documents(path: Path) -> tuple[SourceDocument, ...]:
    """Đọc một file hoặc directory dataset thành tài liệu nguồn.

    Args:
        path: File JSON/JSONL/Parquet hoặc directory chứa các file đó.

    Returns:
        Các tài liệu theo thứ tự file và dòng ổn định.

    Raises:
        ConfigurationError: Khi không có file hợp lệ hoặc record thiếu text.
    """

    resolved = path.expanduser().resolve()
    files: tuple[Path, ...]
    if resolved.is_file():
        files = (resolved,)
    elif resolved.is_dir():
        files = tuple(
            candidate
            for candidate in sorted(resolved.iterdir())
            if candidate.is_file() and candidate.suffix.lower() in _SUPPORTED_SUFFIXES
        )
        if not files:
            raise ConfigurationError(f"Directory không có file ingest được: {resolved}")
    else:
        raise ConfigurationError(f"Không tìm thấy nguồn ingest: {resolved}")

    documents: list[SourceDocument] = []
    for file_path in files:
        try:
            records = load_local_records(file_path)
        except (FileNotFoundError, ValueError) as error:
            raise ConfigurationError(f"Không thể đọc nguồn ingest {file_path}: {error}") from error
        for row_index, record in enumerate(records):
            documents.append(_to_source_document(record, file_path, row_index))
    return tuple(documents)


async def ingest_source_path(
    path: Path,
    retriever: QdrantVectorRetriever,
    *,
    recreate: bool = False,
    batch_size: int = 64,
) -> int:
    """Nạp, chunk và upsert tài liệu local vào Qdrant.

    Args:
        path: File hoặc directory tài liệu.
        retriever: Qdrant adapter đã cấu hình collection và embedder.
        recreate: Có xóa collection cũ trước khi ingest hay không.
        batch_size: Kích thước batch embedding/upsert.

    Returns:
        Tổng số passage đã upsert.
    """

    documents = load_source_documents(path)
    return await retriever.ingest_documents(
        documents,
        recreate=recreate,
        batch_size=batch_size,
    )


def _to_source_document(
    record: Mapping[str, Any],
    path: Path,
    row_index: int,
) -> SourceDocument:
    text = _first_text(record, ("text", "content", "body", "document"))
    if text is None:
        raise ConfigurationError(
            f"Record {row_index} trong {path} thiếu text/content/body/document"
        )
    document_id = _first_text(record, ("document_id", "doc_id", "id"))
    if document_id is None:
        document_id = f"{path.stem}-{row_index:08d}"
    title = _first_text(record, ("title", "name"))
    url = _first_text(record, ("url", "source_url"))
    excluded = {
        "text",
        "content",
        "body",
        "document",
        "document_id",
        "doc_id",
        "id",
        "title",
        "name",
        "url",
        "source_url",
    }
    metadata = {key: value for key, value in record.items() if key not in excluded}
    return SourceDocument(
        document_id=document_id,
        text=text,
        title=title,
        url=url,
        metadata=cast(Mapping[str, Any], metadata),
    )


def _first_text(record: Mapping[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = record.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None
