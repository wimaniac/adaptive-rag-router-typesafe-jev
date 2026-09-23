"""Kiểm thử Jev router bằng fake System One client, không gọi TypeSafe API."""

from typing import Any

import pytest

from adaptive_rag_router.domain import (
    ContextQuality,
    ProbabilityProvenance,
    QueryRequest,
    RetrievalResult,
    RetrievalSource,
    RetrievedDocument,
)
from adaptive_rag_router.domain.errors import DecisionSchemaError
from adaptive_rag_router.routing import JevRouter


def _choice(selected: str, labels: list[str]) -> dict[str, Any]:
    probability = 1.0 / (len(labels) + 1)
    values = {label: probability for label in labels}
    values[selected] = probability * 2
    return {
        "type": "choice",
        "choice": selected,
        "probabilities": values,
        "confidence": 0.7,
    }


class _FakeJevClient:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def system_one(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return self.responses.pop(0)


async def test_jev_batches_pre_route_choices() -> None:
    """Ba quyết định pre-route được gửi trong đúng một System One request."""

    response = {
        "model": "jev-1.13.0",
        "usage": {"input_tokens": 1_000, "output_tokens": 3},
        "answers": {
            "retrieval_source": _choice("vector", ["none", "vector", "web"]),
            "complexity": _choice("medium", ["low", "medium", "high"]),
            "initial_model_tier": _choice("economy", ["economy", "strong"]),
        },
    }
    client = _FakeJevClient([response])
    router = JevRouter(client)

    decision = await router.pre_route(QueryRequest(query="Explain the internal policy"))

    assert len(client.calls) == 1
    assert len(client.calls[0]["questions"]) == 3
    assert decision.retrieval_source.selected is RetrievalSource.VECTOR
    assert decision.retrieval_source.provenance is ProbabilityProvenance.NATIVE
    assert decision.usage is not None
    assert decision.usage.cost_usd == pytest.approx(0.000042)


async def test_jev_context_uses_noul_thresholds_in_code() -> None:
    """Passage filtering dùng Noul probabilities và threshold của ứng dụng."""

    response = {
        "model": "jev-1.13.0",
        "answers": {
            "context_quality": _choice("sufficient", ["insufficient", "partial", "sufficient"]),
            "relevant_0": {"type": "noul", "noul": 0.9},
            "evidence_0": {"type": "noul", "noul": 0.8},
            "contradiction_0": {"type": "noul", "noul": 0.1},
            "injection_0": {"type": "noul", "noul": 0.2},
        },
    }
    client = _FakeJevClient([response])
    router = JevRouter(client, passage_threshold=0.5)
    request = QueryRequest(query="What is the policy?")
    retrieval = RetrievalResult(
        source=RetrievalSource.VECTOR,
        query=request.query,
        provider="fixture",
        documents=(
            RetrievedDocument(
                document_id="doc-1",
                text="The policy says thirty days.",
                source=RetrievalSource.VECTOR,
            ),
        ),
    )

    context = await router.assess_context(request, retrieval)

    assert context.quality.selected is ContextQuality.SUFFICIENT
    assert context.accepted_document_ids == ("doc-1",)
    assert len(client.calls[0]["questions"]) == 5


async def test_jev_missing_choice_answer_is_schema_error() -> None:
    """Response thiếu answer bắt buộc không được tự đoán fallback tại adapter."""

    router = JevRouter(_FakeJevClient([{"model": "jev-1.13.0", "answers": {}}]))

    with pytest.raises(DecisionSchemaError, match="thiếu answer"):
        await router.pre_route(QueryRequest(query="Question"))
