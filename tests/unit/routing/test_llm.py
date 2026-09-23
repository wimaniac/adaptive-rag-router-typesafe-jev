"""Kiểm thử LLM router bằng OpenAI-compatible fake client, không dùng network."""

import json
from types import SimpleNamespace
from typing import Any

import pytest

from adaptive_rag_router.domain import (
    ProbabilityProvenance,
    QueryRequest,
    RetrievalResult,
    RetrievalSource,
    RetrievedDocument,
)
from adaptive_rag_router.domain.errors import DecisionSchemaError, ProviderError
from adaptive_rag_router.routing import LLMRouter


class _FakeCompletions:
    def __init__(self, responses: list[str] | None = None, error: Exception | None = None) -> None:
        self.responses = list(responses or [])
        self.error = error
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        content = self.responses.pop(0)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            model="deepseek-flash-resolved",
            usage=SimpleNamespace(
                prompt_tokens=100,
                completion_tokens=10,
                prompt_tokens_details=SimpleNamespace(cached_tokens=20),
            ),
        )


def _client(completions: _FakeCompletions) -> Any:
    return SimpleNamespace(chat=SimpleNamespace(completions=completions))


def _choice(selected: str, labels: list[str]) -> dict[str, Any]:
    return {
        "selected": selected,
        "probabilities": {label: 3.0 if label == selected else 1.0 for label in labels},
        "confidence": 0.75,
        "reasons": ["fixture"],
    }


async def test_llm_pre_route_validates_and_normalizes_json() -> None:
    """Self-reported score được chuẩn hóa nhưng không bị gắn nhãn native."""

    payload = {
        "retrieval_source": _choice("web", ["none", "vector", "web"]),
        "complexity": _choice("medium", ["low", "medium", "high"]),
        "initial_model_tier": _choice("economy", ["economy", "strong"]),
    }
    completions = _FakeCompletions([json.dumps(payload)])
    router = LLMRouter(_client(completions))

    result = await router.pre_route(QueryRequest(query="What happened today?"))

    assert result.retrieval_source.selected is RetrievalSource.WEB
    assert result.retrieval_source.provenance is ProbabilityProvenance.SELF_REPORTED
    assert sum(result.retrieval_source.raw_probabilities.values()) == pytest.approx(1.0)
    assert completions.calls[0]["response_format"] == {"type": "json_object"}
    assert completions.calls[0]["extra_body"] == {"thinking": {"type": "disabled"}}
    assert result.retrieval_source.model_id == "deepseek-flash-resolved"
    assert result.usage is not None
    assert result.usage.input_tokens == 100


async def test_llm_context_rejects_unknown_document_ids() -> None:
    """LLM không được tự tạo document ID ngoài retrieval result."""

    payload = {
        "quality": _choice("partial", ["insufficient", "partial", "sufficient"]),
        "passages": [
            {
                "document_id": "invented",
                "relevance_probability": 0.8,
                "evidence_probability": 0.8,
                "contradiction_probability": 0.0,
                "injection_probability": 0.0,
            }
        ],
        "accepted_document_ids": ["invented"],
    }
    router = LLMRouter(_client(_FakeCompletions([json.dumps(payload)])))
    request = QueryRequest(query="Question")
    retrieval = RetrievalResult(
        source=RetrievalSource.VECTOR,
        query="Question",
        provider="fixture",
        documents=(
            RetrievedDocument(
                document_id="real",
                text="Real evidence",
                source=RetrievalSource.VECTOR,
            ),
        ),
    )

    with pytest.raises(DecisionSchemaError, match="document ID"):
        await router.assess_context(request, retrieval)


async def test_llm_invalid_json_becomes_schema_error() -> None:
    """Text tự do của provider không lọt qua typed contract."""

    router = LLMRouter(_client(_FakeCompletions(["not-json"])))

    with pytest.raises(DecisionSchemaError, match="sai schema"):
        await router.pre_route(QueryRequest(query="Question"))


async def test_llm_provider_failure_is_normalized() -> None:
    """Lỗi client được chuyển thành ProviderError không chứa secret/payload."""

    router = LLMRouter(_client(_FakeCompletions(error=RuntimeError("provider down"))))

    with pytest.raises(ProviderError, match="RuntimeError"):
        await router.pre_route(QueryRequest(query="Question"))
