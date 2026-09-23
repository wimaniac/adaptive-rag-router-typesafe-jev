"""Kiểm thử chuẩn hóa nguồn ingestion local mà không tải embedding model."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from adaptive_rag_router.domain.errors import ConfigurationError
from adaptive_rag_router.interfaces.ingestion import load_source_documents


def test_load_source_documents_supports_directory(tmp_path: Path) -> None:
    source_dir = tmp_path / "corpus"
    source_dir.mkdir()
    (source_dir / "a.jsonl").write_text(
        json.dumps({"id": "doc-1", "content": "First", "domain": "test"}) + "\n",
        encoding="utf-8",
    )
    (source_dir / "b.json").write_text(
        json.dumps([{"document_id": "doc-2", "text": "Second"}]),
        encoding="utf-8",
    )

    documents = load_source_documents(source_dir)

    assert [document.document_id for document in documents] == ["doc-1", "doc-2"]
    assert documents[0].metadata["domain"] == "test"


def test_load_source_documents_rejects_missing_text(tmp_path: Path) -> None:
    source = tmp_path / "invalid.json"
    source.write_text(json.dumps([{"id": "doc-1"}]), encoding="utf-8")

    with pytest.raises(ConfigurationError, match="thiếu text"):
        load_source_documents(source)
