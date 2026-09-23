"""Kiểm thử human audit sampling, router blinding và artifact templates."""

from __future__ import annotations

from pathlib import Path

from adaptive_rag_router.domain.enums import CompletionStatus, RouterKind
from adaptive_rag_router.evaluation import (
    BenchmarkRecord,
    BenchmarkSample,
    build_human_audit_bundle,
    write_human_audit_bundle,
)


def _samples(count: int) -> tuple[BenchmarkSample, ...]:
    return tuple(
        BenchmarkSample(
            query_id=f"q-{index}",
            query=f"Question {index}?",
            dataset="fixture",
            stratum=f"fixture|{index % 2}",
            group_id=f"g-{index}",
            reference_answer=f"Reference {index}",
        )
        for index in range(count)
    )


def _records(samples: tuple[BenchmarkSample, ...]) -> tuple[BenchmarkRecord, ...]:
    return tuple(
        BenchmarkRecord(
            run_id="formal",
            query_id=sample.query_id,
            dataset=sample.dataset,
            stratum=sample.stratum,
            group_id=sample.group_id,
            router=router,
            answer=f"Answer from {router.value} for {sample.query_id}",
            status=CompletionStatus.COMPLETED,
        )
        for sample in samples
        for router in RouterKind
    )


def test_human_audit_blinds_router_and_creates_two_annotation_forms() -> None:
    samples = _samples(6)

    bundle = build_human_audit_bundle(samples, _records(samples), sample_size=4, seed=9)

    assert len(bundle.items) == 4
    assert len(bundle.blind_key) == 12
    assert len(bundle.annotator_one) == 12
    assert len(bundle.annotator_two) == 12
    assert len(bundle.adjudication) == 12
    for item in bundle.items:
        assert [candidate.candidate_id for candidate in item.candidates] == ["A", "B", "C"]
        assert "router" not in item.model_dump_json()
        key_routers = {
            entry.router for entry in bundle.blind_key if entry.audit_id == item.audit_id
        }
        assert key_routers == set(RouterKind)


def test_human_audit_writer_uses_separate_blind_key(tmp_path: Path) -> None:
    samples = _samples(3)
    bundle = build_human_audit_bundle(samples, _records(samples), sample_size=2, seed=3)

    paths = write_human_audit_bundle(tmp_path / "audit", bundle)

    assert paths.packet.is_file()
    assert paths.blind_key.is_file()
    assert len(paths.packet.read_text(encoding="utf-8").splitlines()) == 2
    assert len(paths.blind_key.read_text(encoding="utf-8").splitlines()) == 6
    assert '"router"' not in paths.packet.read_text(encoding="utf-8")
    assert '"router"' in paths.blind_key.read_text(encoding="utf-8")
