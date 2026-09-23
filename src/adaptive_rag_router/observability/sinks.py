"""Lưu DecisionTrace vào memory hoặc JSONL phục vụ audit và benchmark replay."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Protocol

from adaptive_rag_router.domain.models import DecisionTrace


class TraceSink(Protocol):
    """Giao diện bất đồng bộ nhận một decision trace đã được redaction."""

    async def write(self, trace: DecisionTrace) -> None:
        """Lưu một trace mà không thay đổi nội dung."""


class InMemoryTraceSink:
    """Sink nhẹ cho unit test và phiên chạy ngắn."""

    def __init__(self) -> None:
        """Khởi tạo danh sách trace rỗng."""

        self.traces: list[DecisionTrace] = []

    async def write(self, trace: DecisionTrace) -> None:
        """Thêm trace vào bộ nhớ theo thứ tự nhận."""

        self.traces.append(trace)


class JsonlTraceSink:
    """Append trace vào JSONL với lock để tránh ghi xen kẽ giữa các task."""

    def __init__(self, path: Path) -> None:
        """Khởi tạo sink tại đường dẫn artifact được chỉ định."""

        self._path = path
        self._lock = asyncio.Lock()

    async def write(self, trace: DecisionTrace) -> None:
        """Ghi một dòng JSON UTF-8; tự tạo thư mục cha khi cần."""

        payload = trace.model_dump_json(exclude_none=True)
        async with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            await asyncio.to_thread(self._append, payload)

    def _append(self, payload: str) -> None:
        with self._path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.write("\n")
