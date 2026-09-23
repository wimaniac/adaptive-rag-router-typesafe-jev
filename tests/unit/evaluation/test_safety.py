"""Kiểm thử tạo và chấm safety context variants hoàn toàn offline."""

from __future__ import annotations

from collections import Counter

from adaptive_rag_router.domain.enums import (
    ContextQuality,
    ProbabilityProvenance,
    RouterKind,
)
from adaptive_rag_router.domain.models import (
    ContextAssessment,
    DecisionEvidence,
    QueryRequest,
    RetrievalResult,
)
from adaptive_rag_router.evaluation import (
    BenchmarkSample,
    SafetyVariantKind,
    build_safety_markdown_report,
    build_safety_variants,
    evaluate_safety_assessment,
    run_safety_exploratory,
)


class PassingEngine:
    """Context engine giả nhận support và chặn injection."""

    async def assess_context(
        self,
        request: QueryRequest,
        retrieval: RetrievalResult,
    ) -> ContextAssessment:
        """Trả assessment đúng kỳ vọng của injection fixture."""

        del request
        support_id = next(
            item.document_id for item in retrieval.documents if ":support" in item.document_id
        )
        injection_id = next(
            item.document_id for item in retrieval.documents if ":injection" in item.document_id
        )
        return _assessment(
            quality=ContextQuality.SUFFICIENT,
            accepted=(support_id,),
            injections=(injection_id,),
        )


class FailingEngine:
    """Context engine giả ném lỗi provider để kiểm tra controlled result."""

    async def assess_context(
        self,
        request: QueryRequest,
        retrieval: RetrievalResult,
    ) -> ContextAssessment:
        """Mô phỏng provider failure không làm dừng toàn safety run."""

        del request, retrieval
        raise RuntimeError("provider unavailable")


def _samples(count: int) -> tuple[BenchmarkSample, ...]:
    return tuple(
        BenchmarkSample(
            query_id=f"q-{index:03d}",
            query=f"Question {index}?",
            dataset="fixture",
            stratum=f"fixture|{index % 4}",
            group_id=f"g-{index:03d}",
            reference_answer=f"Answer {index}",
        )
        for index in range(count)
    )


def _assessment(
    *,
    quality: ContextQuality,
    accepted: tuple[str, ...] = (),
    injections: tuple[str, ...] = (),
) -> ContextAssessment:
    probabilities = {item.value: float(item is quality) for item in ContextQuality}
    return ContextAssessment(
        quality=DecisionEvidence[ContextQuality](
            selected=quality,
            raw_probabilities=probabilities,
            provenance=ProbabilityProvenance.HEURISTIC,
            engine=RouterKind.RULE,
            model_id="rule-v1",
            prompt_version="rule-v1",
        ),
        accepted_document_ids=accepted,
        rejected_injection_ids=injections,
    )


def test_build_safety_variants_balances_four_kinds() -> None:
    variants = build_safety_variants(_samples(120), sample_size=100, seed=11)

    assert len(variants) == 100
    assert Counter(item.kind for item in variants) == {
        SafetyVariantKind.NOISE: 25,
        SafetyVariantKind.CONFLICT: 25,
        SafetyVariantKind.STALE: 25,
        SafetyVariantKind.PROMPT_INJECTION: 25,
    }
    injection = next(item for item in variants if item.kind is SafetyVariantKind.PROMPT_INJECTION)
    assert len(injection.documents) == 2
    assert injection.required_rejected_injection_ids


def test_evaluate_safety_assessment_checks_required_signals() -> None:
    variant = next(
        item
        for item in build_safety_variants(_samples(8), sample_size=4, seed=5)
        if item.kind is SafetyVariantKind.PROMPT_INJECTION
    )
    assessment = _assessment(
        quality=variant.expected_quality,
        accepted=variant.required_accepted_ids,
        injections=variant.required_rejected_injection_ids,
    )

    result = evaluate_safety_assessment(variant, assessment, router=RouterKind.RULE)

    assert result.passed is True
    assert result.injection_ids_match is True


async def test_safety_runner_records_success_and_provider_error() -> None:
    variant = next(
        item
        for item in build_safety_variants(_samples(8), sample_size=4, seed=5)
        if item.kind is SafetyVariantKind.PROMPT_INJECTION
    )

    results = await run_safety_exploratory(
        (variant,),
        {
            RouterKind.RULE: PassingEngine(),  # type: ignore[dict-item]
            RouterKind.JEV: FailingEngine(),  # type: ignore[dict-item]
        },
        max_concurrency=2,
    )

    assert len(results) == 2
    assert results[0].passed is True
    assert results[1].passed is False
    assert results[1].error == "RuntimeError"
    report = build_safety_markdown_report((variant,), results)
    assert "Pass rate theo loại context" in report
    assert "| rule | 1 | 1.0000" in report
