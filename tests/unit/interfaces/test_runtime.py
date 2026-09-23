"""Kiểm thử runtime adapter, frozen web loader và DI hoàn toàn offline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from adaptive_rag_router.application import AdaptiveRAGRouter
from adaptive_rag_router.config import Settings
from adaptive_rag_router.domain.enums import (
    CompletionStatus,
    Complexity,
    ContextQuality,
    ModelTier,
    ProbabilityProvenance,
    RetrievalSource,
    RouterKind,
)
from adaptive_rag_router.domain.models import (
    AnswerResponse,
    ContextAssessment,
    DecisionEvidence,
    DecisionTrace,
    GenerationResult,
    PreRouteDecision,
    QueryRequest,
    RetrievalResult,
    RetrievedDocument,
    StageTrace,
)
from adaptive_rag_router.evaluation import (
    BenchmarkRecord,
    BenchmarkSample,
    DatasetSplit,
    TemperatureCalibrationModel,
)
from adaptive_rag_router.generation import Generator, QueryRewriter, QueryRewriteResult
from adaptive_rag_router.interfaces.runtime import (
    CalibratedBenchmarkAdapter,
    FileBenchmarkRecordStore,
    PipelineBenchmarkAdapter,
    build_runtime,
    load_mock_web_fixtures,
    token_f1,
)
from adaptive_rag_router.routing import DecisionEngine, RuleBasedRouter


class _Generator:
    async def generate(
        self,
        request: QueryRequest,
        documents: tuple[RetrievedDocument, ...],
        model_tier: ModelTier,
    ) -> GenerationResult:
        return GenerationResult(
            text="grounded answer",
            model_tier=model_tier,
            model_id="fake",
        )


class _Rewriter:
    async def rewrite(
        self,
        request: QueryRequest,
        missing_information: tuple[str, ...],
        documents: tuple[RetrievedDocument, ...],
    ) -> QueryRewriteResult:
        return QueryRewriteResult(query=request.query)


class _Pipeline:
    async def answer(
        self,
        request: QueryRequest,
        engine: RouterKind,
        *,
        include_trace: bool | None = None,
    ) -> AnswerResponse:
        source = DecisionEvidence(
            selected=RetrievalSource.NONE,
            raw_probabilities={"none": 0.8, "vector": 0.1, "web": 0.1},
            confidence=0.8,
            provenance=ProbabilityProvenance.HEURISTIC,
            engine=engine,
            model_id="fake",
            prompt_version="test",
        )
        complexity = DecisionEvidence(
            selected=Complexity.LOW,
            raw_probabilities={"low": 0.8, "medium": 0.1, "high": 0.1},
            confidence=0.8,
            provenance=ProbabilityProvenance.HEURISTIC,
            engine=engine,
            model_id="fake",
            prompt_version="test",
        )
        tier = DecisionEvidence(
            selected=ModelTier.ECONOMY,
            raw_probabilities={"economy": 0.9, "strong": 0.1},
            confidence=0.9,
            provenance=ProbabilityProvenance.HEURISTIC,
            engine=engine,
            model_id="fake",
            prompt_version="test",
        )
        trace = DecisionTrace(
            router=engine,
            pre_route=PreRouteDecision(
                retrieval_source=source,
                complexity=complexity,
                initial_model_tier=tier,
            ),
            stages=(
                StageTrace(
                    stage="pre_route",
                    cost_usd=0.0002,
                    metadata={"provider": "typesafe-jev"},
                ),
                StageTrace(
                    stage="generation_initial",
                    cost_usd=0.0008,
                    metadata={"provider": "deepseek"},
                ),
            ),
            total_cost_usd=0.001,
        )
        return AnswerResponse(
            query_id=request.query_id,
            answer="the grounded answer",
            status=CompletionStatus.COMPLETED,
            trace=trace,
        )


class _RetrievalPipeline:
    async def answer(
        self,
        request: QueryRequest,
        engine: RouterKind,
        *,
        include_trace: bool | None = None,
    ) -> AnswerResponse:
        del include_trace
        source = DecisionEvidence(
            selected=RetrievalSource.VECTOR,
            raw_probabilities={"none": 0.1, "vector": 0.8, "web": 0.1},
            confidence=0.8,
            provenance=ProbabilityProvenance.HEURISTIC,
            engine=engine,
            model_id="fake",
            prompt_version="test",
        )
        complexity = DecisionEvidence(
            selected=Complexity.LOW,
            raw_probabilities={"low": 0.8, "medium": 0.1, "high": 0.1},
            confidence=0.8,
            provenance=ProbabilityProvenance.HEURISTIC,
            engine=engine,
            model_id="fake",
            prompt_version="test",
        )
        tier = DecisionEvidence(
            selected=ModelTier.ECONOMY,
            raw_probabilities={"economy": 0.9, "strong": 0.1},
            confidence=0.9,
            provenance=ProbabilityProvenance.HEURISTIC,
            engine=engine,
            model_id="fake",
            prompt_version="test",
        )
        quality = DecisionEvidence(
            selected=ContextQuality.SUFFICIENT,
            raw_probabilities={"insufficient": 0.1, "partial": 0.1, "sufficient": 0.8},
            confidence=0.8,
            provenance=ProbabilityProvenance.HEURISTIC,
            engine=engine,
            model_id="fake",
            prompt_version="test",
        )
        passage_id = "ragrouter:medical:doc-1#chunk-0000"
        document = RetrievedDocument(
            document_id=passage_id,
            text="BCC is a skin cancer.",
            source=RetrievalSource.VECTOR,
            metadata={"parent_document_id": "ragrouter:medical:doc-1"},
        )
        trace = DecisionTrace(
            router=engine,
            pre_route=PreRouteDecision(
                retrieval_source=source,
                complexity=complexity,
                initial_model_tier=tier,
            ),
            retrieval_rounds=(
                RetrievalResult(
                    source=RetrievalSource.VECTOR,
                    query=request.query,
                    documents=(document,),
                    provider="fixture",
                ),
            ),
            context_assessments=(
                ContextAssessment(
                    quality=quality,
                    accepted_document_ids=(passage_id,),
                ),
            ),
        )
        return AnswerResponse(
            query_id=request.query_id,
            answer="A skin cancer. [ragrouter:medical:doc-1#chunk-0000]",
            citations=(passage_id,),
            status=CompletionStatus.COMPLETED,
            trace=trace,
        )


def test_token_f1_handles_overlap_and_empty_text() -> None:
    assert token_f1("a b", "a c") == pytest.approx(0.5)
    assert token_f1("", "") == 1.0
    assert token_f1("answer", "different") == 0.0


@pytest.mark.asyncio
async def test_file_record_store_round_trips_jsonl_atomically(tmp_path: Path) -> None:
    checkpoint = tmp_path / "records.jsonl"
    store = FileBenchmarkRecordStore(checkpoint)
    record = BenchmarkRecord(
        run_id="resume-run",
        query_id="q-1",
        dataset="fixture",
        stratum="fixture|fact",
        group_id="g-1",
        router=RouterKind.JEV,
        answer="cached answer",
        status=CompletionStatus.COMPLETED,
    )

    assert await store.load() == ()
    await store.save((record,))

    assert await store.load() == (record,)
    assert checkpoint.read_text(encoding="utf-8").count("\n") == 1
    assert not checkpoint.with_suffix(".jsonl.tmp").exists()


def test_load_mock_web_fixtures_accepts_provider_style_json(tmp_path: Path) -> None:
    fixture = tmp_path / "web.json"
    fixture.write_text(
        json.dumps(
            {
                "What?": [
                    {
                        "title": "Source",
                        "url": "https://example.test",
                        "content": "Evidence",
                        "score": 0.9,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    loaded = load_mock_web_fixtures(fixture)

    assert loaded["What?"][0].text == "Evidence"
    assert loaded["What?"][0].source is RetrievalSource.WEB
    assert loaded["What?"][0].document_id.startswith("mock-")


def test_build_runtime_with_injected_services_does_not_require_keys(tmp_path: Path) -> None:
    settings = Settings(qdrant_path=tmp_path / "qdrant", cache_dir=tmp_path / "cache")

    bundle = build_runtime(
        settings,
        engine_kinds=(RouterKind.RULE,),
        engine_factory=lambda _kind, configured: cast(
            DecisionEngine,
            RuleBasedRouter(max_repair_rounds=configured.max_repair_rounds),
        ),
        generator=cast(Generator, _Generator()),
        query_rewriter=cast(QueryRewriter, _Rewriter()),
    )

    assert isinstance(bundle.pipeline, AdaptiveRAGRouter)
    assert bundle.tavily_ledger is None


@pytest.mark.asyncio
async def test_pipeline_benchmark_adapter_extracts_route_and_quality() -> None:
    adapter = PipelineBenchmarkAdapter(cast(AdaptiveRAGRouter, _Pipeline()))
    sample = BenchmarkSample(
        query_id="q-1",
        query="question",
        dataset="fixture",
        stratum="fixture|fact",
        group_id="g-1",
        reference_answer="grounded answer",
    )

    result = await adapter(sample, RouterKind.RULE)

    assert result.predicted_route == "none:economy"
    assert result.route_probabilities["none:economy"] == pytest.approx(0.72)
    assert sum(result.route_probabilities.values()) == pytest.approx(1.0)
    assert result.quality_score == pytest.approx(0.8)
    assert result.cost_usd == pytest.approx(0.001)
    assert result.metadata["provider_costs_usd"] == pytest.approx(
        {"deepseek": 0.0008, "typesafe-jev": 0.0002}
    )


@pytest.mark.asyncio
async def test_pipeline_adapter_canonicalizes_passage_lineage_for_metrics() -> None:
    adapter = PipelineBenchmarkAdapter(cast(AdaptiveRAGRouter, _RetrievalPipeline()))
    sample = BenchmarkSample(
        query_id="q-lineage",
        query="What is BCC?",
        dataset="fixture",
        stratum="fixture|fact",
        group_id="g-lineage",
        reference_answer="A skin cancer.",
        metadata={"supporting_document_ids": ["ragrouter:medical:doc-1"]},
    )

    result = await adapter(sample, RouterKind.RULE)

    assert result.metadata["retrieved_passage_ids"] == ["ragrouter:medical:doc-1#chunk-0000"]
    assert result.metadata["retrieved_document_ids"] == ["ragrouter:medical:doc-1"]
    assert result.metadata["citations"] == ["ragrouter:medical:doc-1"]
    assert result.metadata["expected_context_quality"] == "sufficient"
    assert result.metadata["predicted_context_quality"] == "sufficient"


@pytest.mark.asyncio
async def test_calibrated_adapter_preserves_raw_distribution_for_audit() -> None:
    raw_adapter = PipelineBenchmarkAdapter(cast(AdaptiveRAGRouter, _Pipeline()))
    sample = BenchmarkSample(
        query_id="q-cal",
        query="question",
        dataset="fixture",
        stratum="fixture|fact",
        group_id="g-cal",
    )
    labels = tuple(
        f"{source.value}:{tier.value}" for source in RetrievalSource for tier in ModelTier
    )
    model = TemperatureCalibrationModel(
        labels=labels,
        temperature=2.0,
        fitted_split=DatasetSplit.CALIBRATION,
        sample_count=200,
        negative_log_likelihood_before=1.0,
        negative_log_likelihood_after=0.8,
    )
    adapter = CalibratedBenchmarkAdapter(raw_adapter, {RouterKind.RULE: model})

    result = await adapter(sample, RouterKind.RULE)

    raw = result.metadata["raw_route_probabilities"]
    assert isinstance(raw, dict)
    assert raw["none:economy"] == pytest.approx(0.72)
    assert result.route_probabilities["none:economy"] < raw["none:economy"]
    assert result.metadata["calibration_version"] == "temperature-scaling-v1"
