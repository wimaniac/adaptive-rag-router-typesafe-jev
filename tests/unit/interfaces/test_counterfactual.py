"""Kiểm thử tạo gold route từ sáu branch hoàn toàn offline."""

from pathlib import Path

import pytest

from adaptive_rag_router.domain import (
    GenerationResult,
    ModelTier,
    QueryRequest,
    RetrievalResult,
    RetrievalSource,
    RetrievedDocument,
)
from adaptive_rag_router.evaluation import BenchmarkSample
from adaptive_rag_router.interfaces.counterfactual import (
    StaticBranchEvaluator,
    attach_counterfactual_gold,
)
from adaptive_rag_router.retrieval import RetrieverRegistry


class _Retriever:
    """Retriever fixture ghi số lần gọi của một nguồn."""

    def __init__(self, source: RetrievalSource) -> None:
        self.source = source
        self.calls = 0

    async def retrieve(self, query: str, *, round_index: int = 0) -> RetrievalResult:
        """Trả một passage xác định cho query."""

        self.calls += 1
        return RetrievalResult(
            source=self.source,
            query=query,
            provider=f"fixture-{self.source.value}",
            round_index=round_index,
            documents=(
                RetrievedDocument(
                    document_id=f"{self.source.value}-doc",
                    text="correct answer",
                    source=self.source,
                    provider_score=0.9,
                ),
            ),
        )


class _Generator:
    """Generator fixture cho quality/cost khác nhau theo branch."""

    def __init__(self) -> None:
        self.calls = 0

    async def generate(
        self,
        request: QueryRequest,
        documents: tuple[RetrievedDocument, ...],
        model_tier: ModelTier,
    ) -> GenerationResult:
        """Sinh đáp án đúng khi branch có retrieval."""

        del request
        self.calls += 1
        source = documents[0].source if documents else RetrievalSource.NONE
        costs = {
            (RetrievalSource.NONE, ModelTier.ECONOMY): 0.01,
            (RetrievalSource.NONE, ModelTier.STRONG): 0.04,
            (RetrievalSource.VECTOR, ModelTier.ECONOMY): 0.02,
            (RetrievalSource.VECTOR, ModelTier.STRONG): 0.05,
            (RetrievalSource.WEB, ModelTier.ECONOMY): 0.03,
            (RetrievalSource.WEB, ModelTier.STRONG): 0.06,
        }
        return GenerationResult(
            text="wrong" if source is RetrievalSource.NONE else "correct answer",
            model_tier=model_tier,
            model_id=f"fixture-{model_tier.value}",
            latency_ms=1.0,
            cost_usd=costs[(source, model_tier)],
        )


@pytest.mark.asyncio
async def test_attach_counterfactual_gold_runs_six_branches_and_reuses_cache(
    tmp_path: Path,
) -> None:
    """Gold chọn branch rẻ nhất đạt floor và cache không gọi provider lần hai."""

    vector = _Retriever(RetrievalSource.VECTOR)
    web = _Retriever(RetrievalSource.WEB)
    generator = _Generator()
    evaluator = StaticBranchEvaluator(
        RetrieverRegistry({RetrievalSource.VECTOR: vector, RetrievalSource.WEB: web}),
        generator,
    )
    samples = (
        BenchmarkSample(
            query_id="q-1",
            query="question",
            dataset="fixture",
            stratum="fixture|fact",
            group_id="g-1",
            reference_answer="correct answer",
        ),
    )
    artifact = tmp_path / "counterfactual.jsonl"

    first = await attach_counterfactual_gold(
        samples,
        evaluator,
        quality_floor=0.8,
        artifact_path=artifact,
    )
    second = await attach_counterfactual_gold(
        samples,
        evaluator,
        quality_floor=0.8,
        artifact_path=artifact,
    )

    assert first[0].expected_route == "vector:economy"
    assert second[0].expected_route == "vector:economy"
    assert first[0].metadata["counterfactual_no_good_route"] is False
    assert generator.calls == 6
    assert vector.calls == 1
    assert web.calls == 1


@pytest.mark.asyncio
async def test_attach_counterfactual_gold_resumes_partial_cache(tmp_path: Path) -> None:
    """Cache một query phải được giữ khi lần sau bổ sung query còn thiếu."""

    vector = _Retriever(RetrievalSource.VECTOR)
    web = _Retriever(RetrievalSource.WEB)
    generator = _Generator()
    evaluator = StaticBranchEvaluator(
        RetrieverRegistry({RetrievalSource.VECTOR: vector, RetrievalSource.WEB: web}),
        generator,
    )
    samples = tuple(
        BenchmarkSample(
            query_id=f"q-{index}",
            query=f"question {index}",
            dataset="fixture",
            stratum="fixture|fact",
            group_id=f"g-{index}",
            reference_answer="correct answer",
        )
        for index in (1, 2)
    )
    artifact = tmp_path / "counterfactual.jsonl"

    await attach_counterfactual_gold(
        samples[:1],
        evaluator,
        quality_floor=0.8,
        artifact_path=artifact,
    )
    resumed = await attach_counterfactual_gold(
        samples,
        evaluator,
        quality_floor=0.8,
        artifact_path=artifact,
    )

    assert [sample.expected_route for sample in resumed] == [
        "vector:economy",
        "vector:economy",
    ]
    assert generator.calls == 12
    assert len(artifact.read_text(encoding="utf-8").splitlines()) == 2
