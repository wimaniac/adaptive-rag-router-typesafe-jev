"""Kiểm thử loader local, sampling và manifest dataset không dùng network."""

from __future__ import annotations

import json
from pathlib import Path

from adaptive_rag_router.evaluation.datasets import (
    load_local_records,
    prepare_benchmark_dataset,
)
from adaptive_rag_router.evaluation.models import DatasetBuildConfig, LocalDatasetSpec


def _write_jsonl(path: Path, count: int, prefix: str) -> None:
    with path.open("w", encoding="utf-8") as file_handle:
        for index in range(count):
            record = {
                "id": f"{prefix}-{index}",
                "question": f"Question {prefix} {index}",
                "domain": f"domain-{index % 4}",
                "family": f"family-{index}",
                "answer": f"Answer {index}",
            }
            file_handle.write(json.dumps(record) + "\n")


def test_load_json_object_with_records(tmp_path: Path) -> None:
    path = tmp_path / "records.json"
    path.write_text(json.dumps({"records": [{"query": "hello"}]}), encoding="utf-8")

    assert load_local_records(path) == [{"query": "hello"}]


def test_prepare_formal_dataset_and_manifest(tmp_path: Path) -> None:
    rag_path = tmp_path / "rag.jsonl"
    crag_path = tmp_path / "crag.jsonl"
    frozen_path = tmp_path / "frozen.jsonl"
    _write_jsonl(rag_path, 700, "rag")
    _write_jsonl(crag_path, 300, "crag")
    frozen_path.write_text('{"query":"cached"}\n', encoding="utf-8")
    config = DatasetBuildConfig(
        ragrouter_bench=LocalDatasetSpec(
            name="ragrouter-bench",
            path=rag_path,
            sample_size=700,
            query_field="question",
            id_field="id",
            stratum_fields=("domain",),
            group_field="family",
            reference_answer_field="answer",
        ),
        crag=LocalDatasetSpec(
            name="crag",
            path=crag_path,
            sample_size=300,
            query_field="question",
            id_field="id",
            stratum_fields=("domain",),
            group_field="family",
            reference_answer_field="answer",
        ),
        frozen_web_records_path=frozen_path,
        seed=19,
    )

    prepared = prepare_benchmark_dataset(config)

    assert len(prepared.samples) == 1_000
    assert len(prepared.splits.dev) == 200
    assert len(prepared.splits.calibration) == 200
    assert len(prepared.splits.test) == 600
    assert [item.selected_record_count for item in prepared.manifest.files] == [700, 300]
    assert prepared.manifest.frozen_web_sha256 is not None
    assert prepared.manifest.split_sizes == {"dev": 200, "calibration": 200, "test": 600}
