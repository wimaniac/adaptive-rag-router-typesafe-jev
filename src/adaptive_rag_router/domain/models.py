"""Các Pydantic model biểu diễn request, decision, retrieval và trace end-to-end."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator

from adaptive_rag_router.domain.enums import (
    CompletionStatus,
    Complexity,
    ContextQuality,
    FallbackAction,
    ModelTier,
    ProbabilityProvenance,
    RepairAction,
    RetrievalSource,
    RouterKind,
)


class DomainModel(BaseModel):
    """Cấu hình chung cho các model miền bất biến sau khi khởi tạo."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class QueryRequest(DomainModel):
    """Yêu cầu đầu vào cho router.

    Args:
        query: Câu hỏi hoặc chỉ dẫn cần xử lý.
        query_id: Định danh ổn định phục vụ trace và benchmark.
        metadata: Metadata công khai mà router được phép quan sát.
        max_cost_usd: Ngân sách biến đổi tối đa, nếu caller cung cấp.
        max_latency_ms: Mục tiêu latency tối đa, nếu caller cung cấp.
    """

    query: str = Field(min_length=1)
    query_id: str = Field(default_factory=lambda: uuid4().hex)
    metadata: dict[str, Any] = Field(default_factory=dict)
    max_cost_usd: float | None = Field(default=None, ge=0)
    max_latency_ms: float | None = Field(default=None, gt=0)

    @field_validator("query")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        """Loại khoảng trắng dư nhưng không thay đổi nội dung truy vấn."""

        normalized = value.strip()
        if not normalized:
            raise ValueError("query không được rỗng")
        return normalized


class DecisionUsage(DomainModel):
    """Usage và variable cost của đúng một provider decision call.

    Args:
        provider: Tên provider không chứa credential.
        model_id: Model thực tế được provider trả về nếu có.
        input_tokens: Số input token bị tính cho call.
        output_tokens: Số output token; Jev hiện tính giá output bằng 0.
        cached_input_tokens: Phần input token cache hit nếu provider báo cáo.
        cost_usd: Chi phí theo pricing snapshot đã khóa.
    """

    provider: str
    model_id: str
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cached_input_tokens: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0, ge=0)


class DecisionEvidence[DecisionT](DomainModel):
    """Giá trị được chọn cùng distribution, confidence và nguồn gốc score.

    Args:
        selected: Quyết định được engine chọn.
        raw_probabilities: Distribution gốc với key là giá trị enum dạng chuỗi.
        calibrated_probabilities: Distribution sau calibration, nếu đã fit.
        confidence: Confidence riêng của provider nếu primitive có hỗ trợ.
        provenance: Nguồn gốc của raw distribution.
        engine: Router tạo ra quyết định.
        model_id: Model hoặc rule version thực tế.
        prompt_version: Phiên bản prompt/rubric.
        calibration_version: Phiên bản calibrator nếu có.
        reasons: Reason codes có cấu trúc, không chứa chain-of-thought.
    """

    selected: DecisionT
    raw_probabilities: dict[str, float]
    calibrated_probabilities: dict[str, float] | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    provenance: ProbabilityProvenance
    engine: RouterKind
    model_id: str
    prompt_version: str
    calibration_version: str | None = None
    reasons: tuple[str, ...] = ()

    @field_validator("raw_probabilities", "calibrated_probabilities")
    @classmethod
    def validate_distribution(cls, value: dict[str, float] | None) -> dict[str, float] | None:
        """Đảm bảo distribution không rỗng, nằm trong [0, 1] và có tổng xấp xỉ 1."""

        if value is None:
            return None
        if not value:
            raise ValueError("probability distribution không được rỗng")
        if any(probability < 0 or probability > 1 for probability in value.values()):
            raise ValueError("probability phải nằm trong [0, 1]")
        total = sum(value.values())
        if abs(total - 1.0) > 1e-3:
            raise ValueError(f"tổng probability phải bằng 1, nhận được {total}")
        return value

    @computed_field  # type: ignore[prop-decorator]
    @property
    def probabilities(self) -> dict[str, float]:
        """Trả distribution calibrated nếu có, nếu không dùng distribution gốc."""

        return self.calibrated_probabilities or self.raw_probabilities


class PreRouteDecision(DomainModel):
    """Ba quyết định trước retrieval dùng để lập execution plan."""

    retrieval_source: DecisionEvidence[RetrievalSource]
    complexity: DecisionEvidence[Complexity]
    initial_model_tier: DecisionEvidence[ModelTier]
    usage: DecisionUsage | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def needs_retrieval_probability(self) -> float:
        """Suy ra xác suất cần retrieval từ duy nhất distribution nguồn retrieval."""

        return 1.0 - self.retrieval_source.probabilities.get(RetrievalSource.NONE.value, 0.0)


class RetrievedDocument(DomainModel):
    """Một passage đã chuẩn hóa từ vector store hoặc web provider."""

    document_id: str
    text: str = Field(min_length=1)
    title: str | None = None
    url: str | None = None
    source: RetrievalSource
    provider_score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalResult(DomainModel):
    """Kết quả của một retrieval round cùng metadata vận hành."""

    source: RetrievalSource
    query: str
    documents: tuple[RetrievedDocument, ...] = ()
    latency_ms: float = Field(default=0, ge=0)
    cached: bool = False
    provider: str
    round_index: int = Field(default=0, ge=0)
    external_credits: int = Field(default=0, ge=0)
    errors: tuple[str, ...] = ()


class PassageAssessment(DomainModel):
    """Các xác suất nguyên tử dùng để lọc một passage retrieval."""

    document_id: str
    relevance_probability: float = Field(ge=0, le=1)
    evidence_probability: float = Field(ge=0, le=1)
    contradiction_probability: float = Field(ge=0, le=1)
    injection_probability: float = Field(ge=0, le=1)


class ContextAssessment(DomainModel):
    """Đánh giá toàn bộ context và danh sách passage được chấp nhận."""

    quality: DecisionEvidence[ContextQuality]
    passages: tuple[PassageAssessment, ...] = ()
    accepted_document_ids: tuple[str, ...] = ()
    conflicting_document_ids: tuple[str, ...] = ()
    rejected_injection_ids: tuple[str, ...] = ()
    usage: DecisionUsage | None = None


class RetrievalRepairDecision(DomainModel):
    """Quyết định tiếp tục hoặc dừng adaptive retrieval loop."""

    action: DecisionEvidence[RepairAction]
    missing_information: tuple[str, ...] = ()
    usage: DecisionUsage | None = None


class FallbackDecision(DomainModel):
    """Quyết định chấp nhận draft, regenerate bằng model mạnh hoặc abstain."""

    action: DecisionEvidence[FallbackAction]
    usage: DecisionUsage | None = None


class TokenUsage(DomainModel):
    """Số token đã dùng trong một provider call."""

    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cached_input_tokens: int = Field(default=0, ge=0)


class GenerationResult(DomainModel):
    """Kết quả sinh câu trả lời từ một model tier cụ thể."""

    text: str
    model_tier: ModelTier
    model_id: str
    usage: TokenUsage = Field(default_factory=TokenUsage)
    latency_ms: float = Field(default=0, ge=0)
    cost_usd: float = Field(default=0, ge=0)
    finish_reason: str | None = None


class StageTrace(DomainModel):
    """Trace gọn của một stage, không chứa secret hoặc chain-of-thought."""

    stage: str
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    latency_ms: float = Field(default=0, ge=0)
    cost_usd: float = Field(default=0, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class DecisionTrace(DomainModel):
    """Audit trail đầy đủ của một lượt xử lý Adaptive RAG."""

    trace_id: str = Field(default_factory=lambda: uuid4().hex)
    router: RouterKind
    pre_route: PreRouteDecision
    retrieval_rounds: tuple[RetrievalResult, ...] = ()
    context_assessments: tuple[ContextAssessment, ...] = ()
    repair_decisions: tuple[RetrievalRepairDecision, ...] = ()
    fallback: FallbackDecision | None = None
    stages: tuple[StageTrace, ...] = ()
    total_cost_usd: float = Field(default=0, ge=0)
    degraded: bool = False


class AnswerResponse(DomainModel):
    """Kết quả công khai của pipeline cùng citation và optional trace."""

    query_id: str
    answer: str
    citations: tuple[str, ...] = ()
    status: CompletionStatus
    trace: DecisionTrace | None = None
