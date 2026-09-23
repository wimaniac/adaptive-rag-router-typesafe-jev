"""Kiểm thử invariant của typed domain models và cấu hình secret-safe."""

import pytest
from pydantic import ValidationError

from adaptive_rag_router.config.settings import Settings
from adaptive_rag_router.domain.enums import (
    Complexity,
    ModelTier,
    ProbabilityProvenance,
    RetrievalSource,
    RouterKind,
)
from adaptive_rag_router.domain.models import DecisionEvidence, PreRouteDecision, QueryRequest


def _evidence(selected: object, probabilities: dict[str, float]) -> DecisionEvidence:
    return DecisionEvidence(
        selected=selected,
        raw_probabilities=probabilities,
        provenance=ProbabilityProvenance.HEURISTIC,
        engine=RouterKind.RULE,
        model_id="test",
        prompt_version="test-v1",
    )


def test_query_is_trimmed_and_gets_id() -> None:
    """Query hợp lệ được trim và nhận query ID tự động."""

    request = QueryRequest(query="  What is RAG?  ")

    assert request.query == "What is RAG?"
    assert request.query_id


def test_empty_query_is_rejected() -> None:
    """Query chỉ chứa khoảng trắng phải bị từ chối."""

    with pytest.raises(ValidationError):
        QueryRequest(query="   ")


def test_probability_distribution_must_sum_to_one() -> None:
    """Distribution sai tổng không được đi vào routing policy."""

    with pytest.raises(ValidationError):
        _evidence(RetrievalSource.NONE, {"none": 0.2, "vector": 0.2, "web": 0.2})


def test_needs_retrieval_is_derived_from_none_probability() -> None:
    """P(retrieval) luôn được suy ra từ cùng một Choice distribution."""

    route = PreRouteDecision(
        retrieval_source=_evidence(
            RetrievalSource.VECTOR,
            {"none": 0.1, "vector": 0.7, "web": 0.2},
        ),
        complexity=_evidence(
            Complexity.MEDIUM,
            {"low": 0.1, "medium": 0.8, "high": 0.1},
        ),
        initial_model_tier=_evidence(
            ModelTier.ECONOMY,
            {"economy": 0.9, "strong": 0.1},
        ),
    )

    assert route.needs_retrieval_probability == pytest.approx(0.9)


def test_settings_do_not_reveal_secret_in_json() -> None:
    """Pydantic SecretStr phải che API key khi cấu hình bị serialize."""

    settings = Settings(
        typesafe_api_key="typesafe-secret",
        deepseek_api_key="deepseek-secret",
        tavily_api_key="tavily-secret",
    )

    payload = settings.model_dump_json()

    assert "typesafe-secret" not in payload
    assert "deepseek-secret" not in payload
    assert "tavily-secret" not in payload
