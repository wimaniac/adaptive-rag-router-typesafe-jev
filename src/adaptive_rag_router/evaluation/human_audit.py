"""Tạo gói human audit mù danh tính router cho hai annotator và adjudication."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from adaptive_rag_router.domain.enums import RouterKind
from adaptive_rag_router.evaluation.datasets import stratified_sample
from adaptive_rag_router.evaluation.models import BenchmarkRecord, BenchmarkSample


class HumanAuditModel(BaseModel):
    """Base model bất biến cho artifact human audit."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class BlindedAuditAnswer(HumanAuditModel):
    """Một answer đã thay router bằng candidate ID mù."""

    candidate_id: str = Field(pattern=r"^[A-Z]$")
    answer: str


class HumanAuditItem(HumanAuditModel):
    """Một truy vấn và các answer đã random hóa thứ tự để annotator chấm."""

    audit_id: str = Field(min_length=1)
    query_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    reference_answer: str | None = None
    candidates: tuple[BlindedAuditAnswer, ...]


class HumanAuditKeyEntry(HumanAuditModel):
    """Khóa giải mù candidate, phải tách khỏi packet của annotator."""

    audit_id: str = Field(min_length=1)
    candidate_id: str = Field(pattern=r"^[A-Z]$")
    router: RouterKind


class HumanAnnotationForm(HumanAuditModel):
    """Form chấm một answer; score rỗng cho tới khi annotator điền."""

    audit_id: str = Field(min_length=1)
    candidate_id: str = Field(pattern=r"^[A-Z]$")
    annotator_id: str = Field(min_length=1)
    correctness_score: int | None = Field(default=None, ge=1, le=5)
    groundedness_score: int | None = Field(default=None, ge=1, le=5)
    citation_support_score: int | None = Field(default=None, ge=1, le=5)
    unsupported_claim: bool | None = None
    notes: str = ""


class AdjudicationForm(HumanAuditModel):
    """Form kết luận cuối cho một candidate sau hai lượt chấm độc lập."""

    audit_id: str = Field(min_length=1)
    candidate_id: str = Field(pattern=r"^[A-Z]$")
    final_correctness_score: int | None = Field(default=None, ge=1, le=5)
    final_groundedness_score: int | None = Field(default=None, ge=1, le=5)
    final_citation_support_score: int | None = Field(default=None, ge=1, le=5)
    final_unsupported_claim: bool | None = None
    adjudicator_notes: str = ""


class HumanAuditBundle(HumanAuditModel):
    """Gói packet mù, khóa giải mù và các form annotation."""

    items: tuple[HumanAuditItem, ...]
    blind_key: tuple[HumanAuditKeyEntry, ...]
    annotator_one: tuple[HumanAnnotationForm, ...]
    annotator_two: tuple[HumanAnnotationForm, ...]
    adjudication: tuple[AdjudicationForm, ...]


class HumanAuditPaths(HumanAuditModel):
    """Các đường dẫn JSONL của một gói human audit đã ghi."""

    packet: Path
    blind_key: Path
    annotator_one: Path
    annotator_two: Path
    adjudication: Path


def build_human_audit_bundle(
    samples: tuple[BenchmarkSample, ...],
    records: tuple[BenchmarkRecord, ...],
    *,
    sample_size: int = 100,
    seed: int = 42,
) -> HumanAuditBundle:
    """Chọn held-out samples và tạo packet mù ba router.

    Args:
        samples: Held-out samples chứa query/reference answer.
        records: Benchmark records của cùng run và đủ ba router mỗi query.
        sample_size: Số query cần audit, mặc định 100.
        seed: Seed cho stratified sampling và hoán vị candidate.

    Returns:
        Bundle gồm packet, blind key, hai form annotator và adjudication.

    Raises:
        ValueError: Khi thiếu hoặc trùng record của một router/query.
    """

    selected = stratified_sample(samples, sample_size, seed=seed)
    by_query: dict[str, dict[RouterKind, BenchmarkRecord]] = {}
    for record in records:
        router_records = by_query.setdefault(record.query_id, {})
        if record.router in router_records:
            raise ValueError(
                f"record trùng cho query={record.query_id!r}, router={record.router.value!r}"
            )
        router_records[record.router] = record

    items: list[HumanAuditItem] = []
    key_entries: list[HumanAuditKeyEntry] = []
    annotation_one: list[HumanAnnotationForm] = []
    annotation_two: list[HumanAnnotationForm] = []
    adjudication: list[AdjudicationForm] = []
    required_routers = tuple(RouterKind)
    for index, sample in enumerate(selected, start=1):
        available = by_query.get(sample.query_id, {})
        missing = set(required_routers) - set(available)
        if missing:
            labels = ", ".join(sorted(router.value for router in missing))
            raise ValueError(f"query {sample.query_id!r} thiếu records: {labels}")

        routers = list(required_routers)
        rng = random.Random(_stable_seed(seed, sample.query_id))
        rng.shuffle(routers)
        audit_id = f"audit-{index:03d}"
        candidates: list[BlindedAuditAnswer] = []
        for candidate_index, router in enumerate(routers):
            candidate_id = chr(ord("A") + candidate_index)
            candidates.append(
                BlindedAuditAnswer(
                    candidate_id=candidate_id,
                    answer=available[router].answer,
                )
            )
            key_entries.append(
                HumanAuditKeyEntry(
                    audit_id=audit_id,
                    candidate_id=candidate_id,
                    router=router,
                )
            )
            annotation_one.append(
                HumanAnnotationForm(
                    audit_id=audit_id,
                    candidate_id=candidate_id,
                    annotator_id="annotator-1",
                )
            )
            annotation_two.append(
                HumanAnnotationForm(
                    audit_id=audit_id,
                    candidate_id=candidate_id,
                    annotator_id="annotator-2",
                )
            )
            adjudication.append(AdjudicationForm(audit_id=audit_id, candidate_id=candidate_id))
        items.append(
            HumanAuditItem(
                audit_id=audit_id,
                query_id=sample.query_id,
                query=sample.query,
                reference_answer=sample.reference_answer,
                candidates=tuple(candidates),
            )
        )

    return HumanAuditBundle(
        items=tuple(items),
        blind_key=tuple(key_entries),
        annotator_one=tuple(annotation_one),
        annotator_two=tuple(annotation_two),
        adjudication=tuple(adjudication),
    )


def write_human_audit_bundle(output_dir: Path, bundle: HumanAuditBundle) -> HumanAuditPaths:
    """Ghi bundle thành năm file JSONL bằng atomic replace.

    Args:
        output_dir: Thư mục đích của human audit.
        bundle: Bundle đã tạo từ held-out records.

    Returns:
        Các đường dẫn artifact vừa ghi.
    """

    root = output_dir.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    paths = HumanAuditPaths(
        packet=root / "packet.jsonl",
        blind_key=root / "blind-key.jsonl",
        annotator_one=root / "annotator-1.jsonl",
        annotator_two=root / "annotator-2.jsonl",
        adjudication=root / "adjudication.jsonl",
    )
    _write_jsonl(paths.packet, bundle.items)
    _write_jsonl(paths.blind_key, bundle.blind_key)
    _write_jsonl(paths.annotator_one, bundle.annotator_one)
    _write_jsonl(paths.annotator_two, bundle.annotator_two)
    _write_jsonl(paths.adjudication, bundle.adjudication)
    return paths


def _stable_seed(seed: int, query_id: str) -> int:
    digest = hashlib.sha256(f"{seed}:{query_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _write_jsonl(path: Path, models: tuple[BaseModel, ...]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = "\n".join(
        json.dumps(model.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
        for model in models
    )
    temporary.write_text(payload + ("\n" if payload else ""), encoding="utf-8", newline="\n")
    temporary.replace(path)
