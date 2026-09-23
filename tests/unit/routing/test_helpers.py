"""Kiểm thử normalization và provenance dùng chung của routing subsystem."""

import pytest

from adaptive_rag_router.domain import (
    Complexity,
    ProbabilityProvenance,
    RouterKind,
)
from adaptive_rag_router.domain.errors import DecisionSchemaError
from adaptive_rag_router.routing.helpers import build_evidence, normalize_probabilities


def test_normalize_probabilities_fills_missing_labels() -> None:
    """Distribution luôn có đủ nhãn và tổng bằng một."""

    distribution = normalize_probabilities(
        {"low": 2.0, "medium": 1.0},
        ["low", "medium", "high"],
    )

    assert distribution == pytest.approx({"low": 2 / 3, "medium": 1 / 3, "high": 0})
    assert sum(distribution.values()) == pytest.approx(1.0)


def test_normalize_zero_scores_uses_selected_one_hot() -> None:
    """Score toàn zero không tạo NaN mà dùng selected làm fallback rõ ràng."""

    distribution = normalize_probabilities(
        {"low": 0.0},
        ["low", "medium", "high"],
        selected="medium",
    )

    assert distribution == {"low": 0.0, "medium": 1.0, "high": 0.0}


def test_unknown_probability_label_is_rejected() -> None:
    """Nhãn ngoài typed contract phải làm hỏng payload thay vì bị bỏ qua."""

    with pytest.raises(DecisionSchemaError, match="nhãn probability"):
        normalize_probabilities({"impossible": 1.0}, ["low", "medium", "high"])


def test_build_evidence_preserves_provenance() -> None:
    """Evidence phân biệt native probability với heuristic hoặc self-report."""

    evidence = build_evidence(
        Complexity,
        "high",
        {"low": 0.1, "medium": 0.2, "high": 0.7},
        confidence=0.8,
        provenance=ProbabilityProvenance.NATIVE,
        engine=RouterKind.JEV,
        model_id="jev-1.13.0",
        prompt_version="v1",
    )

    assert evidence.selected is Complexity.HIGH
    assert evidence.provenance is ProbabilityProvenance.NATIVE
    assert evidence.confidence == 0.8
