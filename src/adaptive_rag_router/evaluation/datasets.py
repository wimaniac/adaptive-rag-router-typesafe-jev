"""Nạp dataset local, stratified sampling và tạo manifest không dùng network."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from adaptive_rag_router.evaluation.models import (
    BenchmarkSample,
    DatasetBuildConfig,
    DatasetFileManifest,
    DatasetManifest,
    LocalDatasetSpec,
    PreparedBenchmarkDataset,
)
from adaptive_rag_router.evaluation.split import create_grouped_benchmark_splits


def sha256_file(path: Path) -> str:
    """Tính SHA-256 của file local theo streaming.

    Args:
        path: Đường dẫn file cần băm.

    Returns:
        Chuỗi SHA-256 dạng hexadecimal.
    """

    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        while chunk := file_handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_local_records(path: Path) -> list[dict[str, Any]]:
    """Đọc record từ JSON, JSONL hoặc Parquet trên filesystem local.

    JSON chấp nhận list ở top-level hoặc object có trường ``records`` là list.
    Hàm không nhận URL và không thực hiện download.

    Args:
        path: File dữ liệu local.

    Returns:
        Danh sách dictionary, mỗi dictionary là một record.

    Raises:
        FileNotFoundError: Khi file không tồn tại.
        ValueError: Khi extension hoặc cấu trúc file không được hỗ trợ.
    """

    local_path = path.expanduser().resolve()
    if not local_path.is_file():
        raise FileNotFoundError(f"không tìm thấy dataset local: {local_path}")
    suffix = local_path.suffix.lower()

    raw_records: object
    if suffix == ".json":
        with local_path.open("r", encoding="utf-8") as file_handle:
            raw_records = json.load(file_handle)
        if isinstance(raw_records, dict):
            raw_records = raw_records.get("records")
    elif suffix in {".jsonl", ".ndjson"}:
        records: list[object] = []
        with local_path.open("r", encoding="utf-8") as file_handle:
            for line_number, line in enumerate(file_handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    records.append(json.loads(stripped))
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"JSONL không hợp lệ tại dòng {line_number}: {local_path}"
                    ) from error
        raw_records = records
    elif suffix == ".parquet":
        import pandas as pd  # type: ignore[import-untyped]

        frame = pd.read_parquet(local_path)
        raw_records = frame.to_dict(orient="records")
    else:
        raise ValueError("chỉ hỗ trợ file .json, .jsonl, .ndjson hoặc .parquet")

    if not isinstance(raw_records, list):
        raise ValueError("dataset phải là list record hoặc object có key 'records'")
    normalized: list[dict[str, Any]] = []
    for index, record in enumerate(raw_records):
        if not isinstance(record, Mapping):
            raise ValueError(f"record thứ {index} không phải object")
        normalized.append(dict(cast(Mapping[str, Any], record)))
    return normalized


def _optional_text(record: Mapping[str, Any], field: str | None) -> str | None:
    """Lấy optional field dưới dạng chuỗi không rỗng."""

    if field is None:
        return None
    value = record.get(field)
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _to_sample(
    record: Mapping[str, Any],
    spec: LocalDatasetSpec,
    row_index: int,
) -> BenchmarkSample:
    """Chuẩn hóa một source record thành BenchmarkSample."""

    query_value = record.get(spec.query_field)
    if query_value is None or not str(query_value).strip():
        raise ValueError(f"{spec.name} dòng {row_index} thiếu query field {spec.query_field!r}")
    raw_id = _optional_text(record, spec.id_field) or f"row-{row_index:08d}"
    query_id = f"{spec.name}:{raw_id}"
    stratum_parts = [_optional_text(record, field) or "unknown" for field in spec.stratum_fields]
    group_value = _optional_text(record, spec.group_field) or query_id
    group_id = f"{spec.name}:{group_value}"
    excluded_fields = {
        spec.query_field,
        spec.id_field,
        spec.group_field,
        *spec.stratum_fields,
    }
    if spec.reference_answer_field is not None:
        excluded_fields.add(spec.reference_answer_field)
    if spec.expected_route_field is not None:
        excluded_fields.add(spec.expected_route_field)
    metadata = {key: value for key, value in record.items() if key not in excluded_fields}
    metadata["source_row_index"] = row_index
    return BenchmarkSample(
        query_id=query_id,
        query=str(query_value),
        dataset=spec.name,
        stratum=f"{spec.name}|{'|'.join(stratum_parts)}",
        group_id=group_id,
        expected_route=_optional_text(record, spec.expected_route_field),
        reference_answer=_optional_text(record, spec.reference_answer_field),
        metadata=metadata,
    )


def stratified_sample(
    samples: Sequence[BenchmarkSample],
    sample_size: int,
    *,
    seed: int,
) -> tuple[BenchmarkSample, ...]:
    """Lấy mẫu xác định theo stratum bằng largest-remainder allocation.

    Args:
        samples: Các sample đã chuẩn hóa.
        sample_size: Tổng số mẫu cần lấy.
        seed: Seed dùng shuffle trong từng stratum.

    Returns:
        Tuple sample có đúng kích thước và được sắp xếp theo query ID.

    Raises:
        ValueError: Khi sample_size ngoài phạm vi hợp lệ.
    """

    if sample_size < 0 or sample_size > len(samples):
        raise ValueError("sample_size phải nằm trong [0, len(samples)]")
    if sample_size == 0:
        return ()

    by_stratum: dict[str, list[BenchmarkSample]] = defaultdict(list)
    for sample in samples:
        by_stratum[sample.stratum].append(sample)

    quotas: dict[str, int] = {}
    fractional_parts: list[tuple[float, str]] = []
    for stratum, members in by_stratum.items():
        ideal = sample_size * len(members) / len(samples)
        base = math.floor(ideal)
        quotas[stratum] = base
        fractional_parts.append((ideal - base, stratum))

    remaining = sample_size - sum(quotas.values())
    for _, stratum in sorted(fractional_parts, key=lambda item: (-item[0], item[1])):
        if remaining == 0:
            break
        if quotas[stratum] < len(by_stratum[stratum]):
            quotas[stratum] += 1
            remaining -= 1
    if remaining:
        for stratum in sorted(by_stratum):
            available = len(by_stratum[stratum]) - quotas[stratum]
            take = min(remaining, available)
            quotas[stratum] += take
            remaining -= take
            if remaining == 0:
                break

    rng = random.Random(seed)
    selected: list[BenchmarkSample] = []
    for stratum in sorted(by_stratum):
        members = sorted(by_stratum[stratum], key=lambda sample: sample.query_id)
        rng.shuffle(members)
        selected.extend(members[: quotas[stratum]])
    return tuple(sorted(selected, key=lambda sample: sample.query_id))


def _load_and_sample(
    spec: LocalDatasetSpec,
    *,
    seed: int,
) -> tuple[tuple[BenchmarkSample, ...], DatasetFileManifest]:
    """Đọc, chuẩn hóa và sample một dataset spec."""

    records = load_local_records(spec.path)
    normalized = tuple(
        _to_sample(record, spec, row_index) for row_index, record in enumerate(records)
    )
    selected = stratified_sample(normalized, spec.sample_size, seed=seed)
    resolved_path = spec.path.expanduser().resolve()
    return selected, DatasetFileManifest(
        dataset=spec.name,
        path=resolved_path,
        sha256=sha256_file(resolved_path),
        source_record_count=len(records),
        selected_record_count=len(selected),
    )


def prepare_benchmark_dataset(config: DatasetBuildConfig) -> PreparedBenchmarkDataset:
    """Dựng dataset formal 700/300, grouped split và provenance manifest.

    Args:
        config: Hai nguồn local, seed, split sizes và optional frozen web path.

    Returns:
        Dataset đã sample/split cùng hash manifest.

    Raises:
        FileNotFoundError: Khi source hoặc frozen web file không tồn tại.
        ValueError: Khi dữ liệu không đủ hoặc không thể chia đúng group.
    """

    rag_samples, rag_manifest = _load_and_sample(
        config.ragrouter_bench,
        seed=config.seed,
    )
    crag_samples, crag_manifest = _load_and_sample(
        config.crag,
        seed=config.seed + 1,
    )
    samples = tuple(sorted((*rag_samples, *crag_samples), key=lambda item: item.query_id))
    splits = create_grouped_benchmark_splits(
        samples,
        dev_size=config.dev_size,
        calibration_size=config.calibration_size,
        test_size=config.test_size,
        seed=config.seed,
    )

    frozen_path: Path | None = None
    frozen_hash: str | None = None
    if config.frozen_web_records_path is not None:
        frozen_path = config.frozen_web_records_path.expanduser().resolve()
        if not frozen_path.is_file():
            raise FileNotFoundError(f"không tìm thấy frozen web records: {frozen_path}")
        frozen_hash = sha256_file(frozen_path)

    selected_ids_payload = "\n".join(sample.query_id for sample in samples).encode("utf-8")
    selected_ids_hash = hashlib.sha256(selected_ids_payload).hexdigest()
    manifest = DatasetManifest(
        seed=config.seed,
        files=(rag_manifest, crag_manifest),
        frozen_web_records_path=frozen_path,
        frozen_web_sha256=frozen_hash,
        selected_query_ids_sha256=selected_ids_hash,
        split_sizes={
            "dev": len(splits.dev),
            "calibration": len(splits.calibration),
            "test": len(splits.test),
        },
    )
    return PreparedBenchmarkDataset(samples=samples, splits=splits, manifest=manifest)
