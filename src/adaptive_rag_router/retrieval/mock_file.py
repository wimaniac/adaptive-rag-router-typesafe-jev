"""Phát lại frozen web JSONL theo file-offset index, không nạp toàn bộ vào RAM."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any, cast

from adaptive_rag_router.domain.enums import RetrievalSource
from adaptive_rag_router.domain.errors import ConfigurationError
from adaptive_rag_router.domain.models import RetrievalResult, RetrievedDocument
from adaptive_rag_router.retrieval.base import deduplicate_documents, normalize_query


class JsonlMockWebRetriever:
    """Frozen-web retriever dùng persisted byte-offset index cho JSONL lớn."""

    def __init__(
        self,
        path: Path,
        *,
        max_results: int = 5,
        provider: str = "mock-web-jsonl",
    ) -> None:
        """Khởi tạo hoặc tái sử dụng index cạnh file fixture.

        Args:
            path: JSONL có mỗi dòng gồm `query` và `results`/`documents`.
            max_results: Số kết quả tối đa, bị khóa trong 1..5.
            provider: Tên provider ghi vào trace.

        Raises:
            ConfigurationError: Khi file hoặc JSONL index không hợp lệ.
            ValueError: Khi `max_results` ngoài giới hạn MVP.
        """

        if not 1 <= max_results <= 5:
            raise ValueError("max_results phải nằm trong khoảng 1..5")
        self._path = path.expanduser().resolve()
        if not self._path.is_file():
            raise ConfigurationError(f"Không tìm thấy frozen web fixture: {self._path}")
        self._max_results = max_results
        self._provider = provider
        self._index_path = self._path.with_suffix(self._path.suffix + ".index.json")
        self._offsets = self._load_or_build_index()

    async def retrieve(self, query: str, *, round_index: int = 0) -> RetrievalResult:
        """Đọc đúng một JSONL record bằng byte offset đã index.

        Args:
            query: Truy vấn gốc hoặc rewrite cần lookup.
            round_index: Chỉ số vòng retrieval.

        Returns:
            Frozen retrieval result, luôn `cached=True` và không gọi network.
        """

        started = perf_counter()
        normalized_query = normalize_query(query)
        key = normalized_query.casefold()
        offset = self._offsets.get(key, self._offsets.get("*"))
        documents = (
            await asyncio.to_thread(self._read_documents, offset) if offset is not None else ()
        )
        return RetrievalResult(
            source=RetrievalSource.WEB,
            query=normalized_query,
            documents=deduplicate_documents(documents, limit=self._max_results),
            latency_ms=(perf_counter() - started) * 1_000,
            cached=True,
            provider=self._provider,
            round_index=round_index,
        )

    def _load_or_build_index(self) -> dict[str, int]:
        source_stat = self._path.stat()
        if self._index_path.is_file():
            try:
                payload = json.loads(self._index_path.read_text(encoding="utf-8"))
                if (
                    isinstance(payload, Mapping)
                    and payload.get("source_size") == source_stat.st_size
                    and payload.get("source_mtime_ns") == source_stat.st_mtime_ns
                    and isinstance(payload.get("offsets"), Mapping)
                ):
                    return {
                        str(key): int(value)
                        for key, value in cast(Mapping[str, Any], payload["offsets"]).items()
                    }
            except (OSError, ValueError, TypeError):
                pass
        offsets = self._build_index()
        payload = {
            "version": 1,
            "source_size": source_stat.st_size,
            "source_mtime_ns": source_stat.st_mtime_ns,
            "offsets": offsets,
        }
        temporary = self._index_path.with_suffix(self._index_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self._index_path)
        return offsets

    def _build_index(self) -> dict[str, int]:
        offsets: dict[str, int] = {}
        try:
            with self._path.open("rb") as handle:
                line_number = 0
                while True:
                    offset = handle.tell()
                    line = handle.readline()
                    if not line:
                        break
                    line_number += 1
                    if not line.strip():
                        continue
                    payload = json.loads(line)
                    if not isinstance(payload, Mapping):
                        raise ConfigurationError(f"Frozen web dòng {line_number} không phải object")
                    query = payload.get("query")
                    if not isinstance(query, str) or not query.strip():
                        raise ConfigurationError(f"Frozen web dòng {line_number} thiếu query")
                    offsets[normalize_query(query).casefold()] = offset
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ConfigurationError(f"Không thể index frozen web {self._path}: {error}") from error
        return offsets

    def _read_documents(self, offset: int) -> tuple[RetrievedDocument, ...]:
        try:
            with self._path.open("rb") as handle:
                handle.seek(offset)
                payload = json.loads(handle.readline())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ConfigurationError(f"Không thể đọc frozen web {self._path}: {error}") from error
        if not isinstance(payload, Mapping):
            raise ConfigurationError("Frozen web indexed record không phải object")
        raw_documents = next(
            (payload[key] for key in ("results", "documents", "search_results") if key in payload),
            (),
        )
        if not isinstance(raw_documents, Sequence) or isinstance(raw_documents, str | bytes):
            raise ConfigurationError("Frozen web results phải là danh sách")
        return tuple(
            document
            for rank, item in enumerate(raw_documents[: self._max_results])
            if isinstance(item, Mapping)
            if (document := _coerce_document(item, rank)) is not None
        )


def _coerce_document(item: Mapping[str, Any], rank: int) -> RetrievedDocument | None:
    text = _first_text(item, ("text", "content", "page_result", "page_snippet"))
    if text is None:
        return None
    url = _first_text(item, ("url", "page_url"))
    document_id = _first_text(item, ("document_id", "id")) or f"mock-web-{rank}"
    raw_score = item.get("provider_score", item.get("score"))
    try:
        score = float(raw_score) if raw_score is not None else None
    except (TypeError, ValueError):
        score = None
    return RetrievedDocument(
        document_id=document_id,
        text=text,
        title=_first_text(item, ("title", "page_name")),
        url=url,
        source=RetrievalSource.WEB,
        provider_score=score,
        metadata={"rank": rank},
    )


def _first_text(item: Mapping[str, Any], keys: Sequence[str]) -> str | None:
    for key in keys:
        value = item.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None
