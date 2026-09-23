"""Kiểm thử group-sequential confirmatory analysis hoàn toàn offline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from adaptive_rag_router.domain.enums import CompletionStatus, RouterKind
from adaptive_rag_router.evaluation.confirmatory import (
    ConfirmatoryDecision,
    analyze_confirmatory_look,
    write_confirmatory_manifest,
)
from adaptive_rag_router.evaluation.models import BenchmarkRecord, BenchmarkSample


def _records(*, quality_delta: float) -> tuple[BenchmarkRecord, ...]:
    records: list[BenchmarkRecord] = []
    for index in range(4):
        for router in (RouterKind.JEV, RouterKind.LLM):
            records.append(
                BenchmarkRecord(
                    run_id="confirmatory",
                    query_id=f"q-{index}",
                    dataset="fixture",
                    stratum=f"s-{index % 2}",
                    group_id=f"g-{index}",
                    router=router,
                    answer="answer",
                    quality_score=(0.8 + quality_delta if router is RouterKind.JEV else 0.8),
                    cost_usd=0.01 if router is RouterKind.JEV else 0.02,
                    latency_ms=10 if router is RouterKind.JEV else 20,
                    status=CompletionStatus.COMPLETED,
                )
            )
    return tuple(records)


def test_confirmatory_look_supports_non_inferior_cheaper_faster_candidate() -> None:
    report = analyze_confirmatory_look(
        _records(quality_delta=0.01),
        run_id="confirmatory",
        look_index=1,
        planned_looks=(4, 8, 12),
        manifest_sha256="a" * 64,
        resamples=100,
        seed=7,
    )

    assert report.decision == ConfirmatoryDecision.SUPPORTED
    assert report.quality_lower == pytest.approx(0.01)
    assert report.cost_reduction == pytest.approx(0.5)
    assert report.p50_latency_reduction == pytest.approx(0.5)
    assert all(report.criteria.values())


def test_confirmatory_final_look_rejects_clear_quality_inferiority() -> None:
    report = analyze_confirmatory_look(
        _records(quality_delta=-0.10),
        run_id="confirmatory",
        look_index=3,
        planned_looks=(2, 3, 4),
        manifest_sha256="b" * 64,
        resamples=100,
        seed=7,
    )

    assert report.decision == ConfirmatoryDecision.NOT_SUPPORTED
    assert report.quality_upper is not None
    assert report.quality_upper < -0.02


def test_confirmatory_manifest_is_deterministic_and_records_exclusions(tmp_path: Path) -> None:
    samples = (
        BenchmarkSample(
            query_id="q-1",
            query="Question",
            dataset="fixture",
            stratum="fixture|fact",
            group_id="g-1",
        ),
    )
    path = tmp_path / "manifest.json"

    first = write_confirmatory_manifest(path, samples, excluded_run_ids=("pilot",))
    second = write_confirmatory_manifest(path, samples, excluded_run_ids=("pilot",))
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert first == second == payload["manifest_sha256"]
    assert payload["excluded_run_ids"] == ["pilot"]
