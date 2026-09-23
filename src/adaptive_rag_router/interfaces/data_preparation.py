"""Chuẩn hóa dữ liệu chính thức RAGRouter-Bench và CRAG cho benchmark local."""

from __future__ import annotations

import bz2
import hashlib
import html
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO, cast

from adaptive_rag_router.domain.errors import ConfigurationError
from adaptive_rag_router.evaluation import load_local_records

_LOCAL_SUFFIXES = {".json", ".jsonl", ".ndjson", ".parquet"}
_TAG_PATTERN = re.compile(r"<[^>]+>")
_SPACE_PATTERN = re.compile(r"\s+")
_MAX_MOCK_DOCUMENT_CHARACTERS = 8192


@dataclass(frozen=True, slots=True)
class PreparedDataCounts:
    """Số record đã ghi sau bước chuẩn hóa dữ liệu.

    Args:
        ragrouter_questions: Số câu hỏi RAGRouter-Bench.
        crag_questions: Số câu hỏi CRAG.
        mock_web_queries: Số query có frozen web fixture.
        corpus_documents: Số tài liệu nguồn dành cho vector ingestion.
    """

    ragrouter_questions: int
    crag_questions: int
    mock_web_queries: int
    corpus_documents: int


def prepare_benchmark_sources(
    ragrouter_source: Path,
    crag_source: Path,
    output_root: Path,
    *,
    web_results_per_query: int = 5,
) -> PreparedDataCounts:
    """Chuẩn hóa hai dataset thành layout mà config mặc định sử dụng.

    Hàm không download dữ liệu và không thay đổi file nguồn. CRAG chấp nhận
    JSON/JSONL/Parquet hoặc file ``.jsonl.bz2`` chính thức. RAGRouter-Bench có
    thể là một file câu hỏi hoặc directory chứa ``Question``/``Corpus`` files.

    Args:
        ragrouter_source: File/directory RAGRouter-Bench đã tải hợp pháp.
        crag_source: File CRAG Task 1/2 hoặc directory chứa file đó.
        output_root: Thư mục gốc, thường là ``data``.
        web_results_per_query: Số frozen web result giữ lại, tối đa 5 cho MVP.

    Returns:
        Số record của từng artifact đã ghi.

    Raises:
        ConfigurationError: Khi source/schema không hợp lệ hoặc không có dữ liệu.
    """

    if not 1 <= web_results_per_query <= 5:
        raise ConfigurationError("web_results_per_query phải nằm trong khoảng 1..5")
    rag_questions, rag_corpus = _prepare_ragrouter(ragrouter_source)
    if not rag_questions:
        raise ConfigurationError("RAGRouter-Bench không có question record hợp lệ")

    root = output_root.expanduser().resolve()
    _write_jsonl(root / "raw" / "ragrouter" / "questions.jsonl", rag_questions)
    crag_question_count, mock_web_count, corpus_count = _stream_crag_and_corpus(
        crag_source,
        root,
        rag_corpus,
        web_results_per_query=web_results_per_query,
    )
    if crag_question_count == 0:
        raise ConfigurationError("CRAG không có question record hợp lệ")
    return PreparedDataCounts(
        ragrouter_questions=len(rag_questions),
        crag_questions=crag_question_count,
        mock_web_queries=mock_web_count,
        corpus_documents=corpus_count,
    )


def _prepare_ragrouter(
    source: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    resolved = source.expanduser().resolve()
    if not resolved.exists():
        raise ConfigurationError(f"Không tìm thấy RAGRouter-Bench source: {resolved}")
    files = _source_files(resolved)
    question_files = [path for path in files if "corpus" not in path.stem.casefold()]
    corpus_files = [path for path in files if "corpus" in path.stem.casefold()]
    questions: list[dict[str, Any]] = []
    corpus: list[dict[str, Any]] = []
    for path in question_files:
        for index, record in enumerate(_iter_records(path)):
            question = _first_text(record, ("question", "query"))
            answer = _first_text(record, ("answer", "ground_truth"))
            if question is None or answer is None:
                continue
            raw_id = _first_text(record, ("id", "query_id")) or f"{path.stem}-{index}"
            domain = _infer_ragrouter_domain(path, record)
            supporting = record.get("supporting_facts")
            normalized_question = {
                "query_id": raw_id,
                "query": question,
                "answer": answer,
                "domain": domain,
                "query_type": _normalize_query_type(record.get("type")),
                "group_id": _group_id(record, raw_id, supporting),
                "supporting_facts": supporting if isinstance(supporting, list) else [],
                "source_file": path.name,
            }
            supporting_document_ids = _supporting_document_ids(supporting, domain)
            if supporting_document_ids:
                normalized_question["supporting_document_ids"] = supporting_document_ids
            questions.append(normalized_question)
    for path in corpus_files:
        for index, record in enumerate(_iter_records(path)):
            text = _first_text(record, ("context", "text", "content", "document"))
            if text is None:
                continue
            raw_id = _first_text(record, ("id", "document_id", "doc_id")) or str(index)
            domain = _infer_ragrouter_domain(path, record)
            corpus.append(
                {
                    "document_id": f"ragrouter:{domain}:{raw_id}",
                    "text": text,
                    "title": _first_text(record, ("title", "name")),
                    "domain": domain,
                    "dataset": "ragrouter-bench",
                }
            )
    return questions, corpus


def _supporting_document_ids(value: object, domain: str) -> list[str]:
    """Chuẩn hóa doc_id có annotation thành parent ID của vector corpus."""

    if not isinstance(value, list):
        return []
    document_ids: list[str] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        raw_id = item.get("doc_id", item.get("document_id"))
        if raw_id is None or not str(raw_id).strip():
            continue
        document_ids.append(f"ragrouter:{domain}:{str(raw_id).strip()}")
    return list(dict.fromkeys(document_ids))


def _stream_crag_and_corpus(
    source: Path,
    output_root: Path,
    rag_corpus: Sequence[Mapping[str, Any]],
    *,
    web_results_per_query: int,
) -> tuple[int, int, int]:
    """Stream CRAG lớn ra ba atomic JSONL artifacts để giới hạn RAM."""

    resolved = source.expanduser().resolve()
    if not resolved.exists():
        raise ConfigurationError(f"Không tìm thấy CRAG source: {resolved}")
    question_path = output_root / "raw" / "crag" / "questions.jsonl"
    mock_path = output_root / "raw" / "crag" / "mock_web.jsonl"
    corpus_path = output_root / "processed" / "corpus.jsonl"
    targets = (question_path, mock_path, corpus_path)
    for target in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
    temporary_paths = tuple(path.with_suffix(path.suffix + ".tmp") for path in targets)
    question_count = 0
    mock_count = 0
    corpus_count = 0
    seen_corpus: set[str] = set()
    try:
        with (
            temporary_paths[0].open("w", encoding="utf-8", newline="\n") as question_handle,
            temporary_paths[1].open("w", encoding="utf-8", newline="\n") as mock_handle,
            temporary_paths[2].open("w", encoding="utf-8", newline="\n") as corpus_handle,
        ):
            for document in rag_corpus:
                if _write_unique_corpus_record(corpus_handle, document, seen_corpus):
                    corpus_count += 1
            for path in _source_files(resolved, include_bz2=True):
                for index, record in enumerate(_iter_records(path)):
                    normalized = _normalize_crag_record(
                        record,
                        path=path,
                        row_index=index,
                        web_results_per_query=web_results_per_query,
                    )
                    if normalized is None:
                        continue
                    question, mock_record = normalized
                    question_handle.write(_jsonl_line(question))
                    mock_handle.write(_jsonl_line(mock_record))
                    question_count += 1
                    mock_count += 1
        for temporary, target in zip(temporary_paths, targets, strict=True):
            temporary.replace(target)
    except Exception:
        for temporary in temporary_paths:
            temporary.unlink(missing_ok=True)
        raise
    return question_count, mock_count, corpus_count


def _normalize_crag_record(
    record: Mapping[str, Any],
    *,
    path: Path,
    row_index: int,
    web_results_per_query: int,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    query = _first_text(record, ("query", "question"))
    answer = _first_text(record, ("answer", "ground_truth"))
    if query is None or answer is None:
        return None
    interaction_id = (
        _first_text(record, ("interaction_id", "query_id", "id")) or f"{path.stem}-{row_index}"
    )
    results = _normalize_crag_results(
        record.get("search_results"),
        interaction_id=interaction_id,
        limit=web_results_per_query,
    )
    question = {
        "interaction_id": interaction_id,
        "query": query,
        "answer": answer,
        "group_id": _group_id(record, interaction_id, None),
        "domain": _first_text(record, ("domain",)) or "unknown",
        "category": _first_text(record, ("question_type", "category")) or "unknown",
        "popularity": _first_text(record, ("popularity",)) or "web",
        "temporal_dynamism": _first_text(
            record,
            ("static_or_dynamic", "temporal_dynamism"),
        )
        or "unknown",
        "query_time": record.get("query_time"),
    }
    return question, {"query": query, "results": results}


def _normalize_crag_results(
    raw_results: object,
    *,
    interaction_id: str,
    limit: int,
) -> list[dict[str, Any]]:
    if not isinstance(raw_results, Sequence) or isinstance(raw_results, str | bytes):
        return []
    normalized: list[dict[str, Any]] = []
    for rank, item in enumerate(raw_results[:limit]):
        if not isinstance(item, Mapping):
            continue
        snippet = _first_text(item, ("page_snippet",))
        body = _first_text(item, ("page_result", "content", "text"))
        if snippet is None and body is None:
            continue
        clean_snippet = _clean_html(snippet or "")
        clean_body = _clean_html(body or "")
        combined = clean_snippet
        if clean_body and clean_body != clean_snippet:
            combined = f"{clean_snippet}\n{clean_body}".strip()
        text = combined[:_MAX_MOCK_DOCUMENT_CHARACTERS].rstrip()
        normalized.append(
            {
                "document_id": f"crag:{interaction_id}:{rank}",
                "title": _first_text(item, ("page_name", "title")),
                "url": _first_text(item, ("page_url", "url")),
                "text": text,
                "score": max(0.0, 1.0 - rank / max(limit, 1)),
            }
        )
    return normalized


def _source_files(path: Path, *, include_bz2: bool = False) -> list[Path]:
    if path.is_file():
        candidates = [path]
    else:
        candidates = sorted(candidate for candidate in path.rglob("*") if candidate.is_file())
    files = [
        candidate
        for candidate in candidates
        if candidate.suffix.lower() in _LOCAL_SUFFIXES
        or (include_bz2 and candidate.name.casefold().endswith(".jsonl.bz2"))
    ]
    if not files:
        raise ConfigurationError(f"Source không có file dữ liệu được hỗ trợ: {path}")
    return files


def _iter_records(path: Path) -> Iterable[dict[str, Any]]:
    if path.name.casefold().endswith(".jsonl.bz2"):
        try:
            with bz2.open(path, "rt", encoding="utf-8") as handle:
                yield from _iter_jsonl_handle(handle, path)
        except (OSError, json.JSONDecodeError) as error:
            raise ConfigurationError(f"Không thể đọc CRAG bz2 {path}: {error}") from error
        return
    if path.suffix.casefold() in {".jsonl", ".ndjson"}:
        try:
            with path.open("r", encoding="utf-8") as handle:
                yield from _iter_jsonl_handle(handle, path)
        except (OSError, json.JSONDecodeError) as error:
            raise ConfigurationError(f"Không thể đọc JSONL {path}: {error}") from error
        return
    if path.suffix.casefold() == ".json":
        try:
            yield from load_local_records(path)
            return
        except (ValueError, json.JSONDecodeError):
            # RAGRouter-Bench đặt JSON Lines trong file có hậu tố `.json`.
            try:
                with path.open("r", encoding="utf-8") as handle:
                    yield from _iter_jsonl_handle(handle, path)
            except (OSError, json.JSONDecodeError) as error:
                raise ConfigurationError(f"Không thể đọc JSON/JSONL {path}: {error}") from error
            return
    try:
        yield from load_local_records(path)
    except (FileNotFoundError, ValueError) as error:
        raise ConfigurationError(f"Không thể đọc dataset {path}: {error}") from error


def _iter_jsonl_handle(handle: TextIO, path: Path) -> Iterable[dict[str, Any]]:
    for line_number, line in enumerate(handle, start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, Mapping):
            raise ConfigurationError(f"Record dòng {line_number} trong {path} không phải object")
        yield dict(cast(Mapping[str, Any], payload))


def _infer_ragrouter_domain(path: Path, record: Mapping[str, Any]) -> str:
    explicit = _first_text(record, ("domain",))
    if explicit is not None:
        return explicit
    identity = str(path).casefold()
    for marker, domain in (
        ("musique", "wikipedia"),
        ("quality", "literature"),
        ("ultra", "legal"),
        ("legal", "legal"),
        ("graphrag", "medical"),
        ("medical", "medical"),
    ):
        if marker in identity:
            return domain
    return "unknown"


def _normalize_query_type(value: object) -> str:
    normalized = str(value or "unknown").strip().casefold().replace("-", "_")
    if "summary" in normalized:
        return "summary"
    if "multi" in normalized or "reason" in normalized:
        return "reasoning"
    if "single" in normalized or "fact" in normalized:
        return "factual"
    return normalized


def _group_id(record: Mapping[str, Any], fallback: str, supporting: object) -> str:
    explicit = _first_text(
        record,
        ("group_id", "entity_id", "corpus_id", "document_id", "question_family"),
    )
    if explicit is not None:
        return explicit
    if isinstance(supporting, Sequence) and not isinstance(supporting, str | bytes) and supporting:
        payload = json.dumps(list(supporting), ensure_ascii=False, sort_keys=True)
        return f"support-{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:20]}"
    return fallback


def _first_text(record: Mapping[str, Any], keys: Iterable[str]) -> str | None:
    for key in keys:
        value = record.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _clean_html(value: str) -> str:
    return _SPACE_PATTERN.sub(" ", html.unescape(_TAG_PATTERN.sub(" ", value))).strip()


def _write_unique_corpus_record(
    handle: TextIO,
    record: Mapping[str, Any],
    seen: set[str],
) -> bool:
    identity = str(record.get("url") or record.get("document_id") or record.get("text"))
    if identity in seen:
        return False
    seen.add(identity)
    handle.write(_jsonl_line(record))
    return True


def _jsonl_line(record: Mapping[str, Any]) -> str:
    return json.dumps(record, ensure_ascii=False) + "\n"


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = "\n".join(json.dumps(record, ensure_ascii=False) for record in records)
    temporary.write_text(payload + ("\n" if payload else ""), encoding="utf-8")
    temporary.replace(path)
