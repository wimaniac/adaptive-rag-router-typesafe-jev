"""Định nghĩa các kiểu dữ liệu có kiểm tra cho benchmark và báo cáo đánh giá."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from adaptive_rag_router.domain.enums import (
    CompletionStatus,
    ModelTier,
    RetrievalSource,
    RouterKind,
)


class EvaluationModel(BaseModel):
    """Cấu hình chung cho các model bất biến của subsystem evaluation."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class DatasetSplit(StrEnum):
    """Tên ba phần dữ liệu dùng trong quy trình benchmark."""

    DEV = "dev"
    CALIBRATION = "calibration"
    TEST = "test"


class BenchmarkSample(EvaluationModel):
    """Một truy vấn benchmark cùng nhãn và khóa chống data leakage.

    Args:
        query_id: Định danh duy nhất của truy vấn.
        query: Nội dung truy vấn gửi vào pipeline.
        dataset: Tên dataset nguồn.
        stratum: Nhóm dùng để stratify split và bootstrap.
        group_id: Khóa nhóm không được xuất hiện ở nhiều split.
        expected_route: Nhãn route chuẩn nếu đã suy ra bằng counterfactual.
        reference_answer: Câu trả lời tham chiếu tùy chọn.
        metadata: Thuộc tính bổ sung không chứa secret.
    """

    query_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    dataset: str = Field(min_length=1)
    stratum: str = Field(min_length=1)
    group_id: str = Field(min_length=1)
    expected_route: str | None = None
    reference_answer: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("query_id", "query", "dataset", "stratum", "group_id")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        """Loại khoảng trắng dư và từ chối chuỗi rỗng."""

        normalized = value.strip()
        if not normalized:
            raise ValueError("giá trị không được rỗng")
        return normalized


class SuccessCriteriaThresholds(EvaluationModel):
    """Các ngưỡng acceptance được khóa trước khi chạy held-out benchmark.

    Mọi tỷ lệ dùng thang ``[0, 1]``. Non-inferiority margin được lưu dưới dạng
    số dương và evaluator so sánh cận dưới quality với giá trị âm tương ứng.
    """

    quality_non_inferiority_margin: float = Field(default=0.02, ge=0, le=1)
    minimum_cost_reduction: float = Field(default=0.20, ge=0, le=1)
    minimum_p50_latency_reduction: float = Field(default=0.15, ge=0, le=1)
    maximum_p95_latency_regression: float = Field(default=0.05, ge=0, le=1)
    critical_recall: float = Field(default=0.90, ge=0, le=1)
    maximum_ece: float = Field(default=0.08, ge=0, le=1)
    maximum_unhandled_error_rate: float = Field(default=0.005, ge=0, le=1)


class BenchmarkConfig(EvaluationModel):
    """Cấu hình một benchmark run có thể tái lập.

    Args:
        run_id: Định danh run, dùng làm tên artifact mặc định.
        routers: Các router cần đánh giá theo thứ tự ổn định.
        split: Split được phép chạy trong run này.
        bootstrap_resamples: Số lần resample cho paired bootstrap.
        confidence_level: Mức confidence của khoảng bootstrap.
        ece_bins: Số bin dùng tính Expected Calibration Error.
        seed: Seed chung cho split và bootstrap.
        max_concurrency: Số pipeline invocation đồng thời tối đa.
        max_samples: Giới hạn số sample, hữu ích cho smoke test.
        fail_fast: Có ném lỗi pipeline ngay thay vì ghi failed record hay không.
        live: Bật các hook budget dành cho live subset.
        checkpoint_every: Chu kỳ ghi checkpoint theo số record hoàn tất.
        stop_on_budget_denial: Dừng dispatch live run khi budget từ chối lần đầu.
        comparison_baseline: Router baseline cho paired comparison.
        alternate_router_order: Đảo thứ tự router theo query để giảm order bias.
        publish_acceptance: Có phát hành kết luận pass/fail formal hay không.
        success_thresholds: Ngưỡng acceptance preregistered của run.
        output_dir: Thư mục artifact do caller lựa chọn.
    """

    run_id: str = Field(default_factory=lambda: uuid4().hex, min_length=1)
    routers: tuple[RouterKind, ...] = (
        RouterKind.JEV,
        RouterKind.RULE,
        RouterKind.LLM,
    )
    split: DatasetSplit = DatasetSplit.TEST
    bootstrap_resamples: int = Field(default=10_000, ge=100)
    confidence_level: float = Field(default=0.95, gt=0, lt=1)
    ece_bins: int = Field(default=10, ge=2, le=100)
    seed: int = 42
    max_concurrency: int = Field(default=4, ge=1, le=128)
    max_samples: int | None = Field(default=None, ge=1)
    fail_fast: bool = False
    live: bool = False
    checkpoint_every: int = Field(default=25, ge=1)
    stop_on_budget_denial: bool = True
    comparison_baseline: RouterKind = RouterKind.LLM
    alternate_router_order: bool = False
    publish_acceptance: bool = True
    success_thresholds: SuccessCriteriaThresholds = Field(default_factory=SuccessCriteriaThresholds)
    output_dir: Path = Path("artifacts/benchmarks")

    @field_validator("routers")
    @classmethod
    def validate_routers(cls, value: tuple[RouterKind, ...]) -> tuple[RouterKind, ...]:
        """Đảm bảo danh sách router không rỗng và không trùng lặp."""

        if not value:
            raise ValueError("routers không được rỗng")
        if len(set(value)) != len(value):
            raise ValueError("routers không được trùng lặp")
        return value

    @model_validator(mode="after")
    def validate_baseline(self) -> BenchmarkConfig:
        """Đảm bảo baseline thuộc tập router đang benchmark khi có so sánh."""

        if len(self.routers) > 1 and self.comparison_baseline not in self.routers:
            raise ValueError("comparison_baseline phải thuộc routers")
        return self


class PipelineRunResult(EvaluationModel):
    """Kết quả chuẩn hóa mà pipeline adapter trả về cho runner.

    Args:
        answer: Câu trả lời cuối cùng của pipeline.
        predicted_route: Nhãn route dự đoán, thường là retrieval source.
        route_probabilities: Distribution dùng cho calibration metrics.
        quality_score: Điểm chất lượng đã được evaluator bên ngoài chuẩn hóa [0, 1].
        cost_usd: Variable cost của toàn bộ lượt chạy.
        latency_ms: End-to-end latency; 0 cho phép runner dùng wall-clock fallback.
        status: Trạng thái hoàn thành của pipeline.
        external_credits: Số credit provider bên ngoài đã tiêu thụ.
        metadata: Trace tóm tắt đã loại secret.
    """

    answer: str
    predicted_route: str | None = None
    route_probabilities: dict[str, float] = Field(default_factory=dict)
    quality_score: float | None = Field(default=None, ge=0, le=1)
    cost_usd: float = Field(default=0, ge=0)
    latency_ms: float = Field(default=0, ge=0)
    status: CompletionStatus = CompletionStatus.COMPLETED
    external_credits: int = Field(default=0, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("route_probabilities")
    @classmethod
    def validate_probabilities(cls, value: dict[str, float]) -> dict[str, float]:
        """Kiểm tra distribution khi pipeline cung cấp probability."""

        if not value:
            return value
        if any(probability < 0 or probability > 1 for probability in value.values()):
            raise ValueError("probability phải nằm trong [0, 1]")
        if abs(sum(value.values()) - 1.0) > 1e-3:
            raise ValueError("tổng route_probabilities phải bằng 1")
        return value


class BenchmarkRecord(EvaluationModel):
    """Một quan sát hoàn chỉnh cho cặp query-router."""

    run_id: str
    query_id: str
    dataset: str
    stratum: str
    group_id: str
    router: RouterKind
    expected_route: str | None = None
    predicted_route: str | None = None
    route_probabilities: dict[str, float] = Field(default_factory=dict)
    reference_answer: str | None = None
    answer: str = ""
    quality_score: float | None = Field(default=None, ge=0, le=1)
    cost_usd: float = Field(default=0, ge=0)
    latency_ms: float = Field(default=0, ge=0)
    status: CompletionStatus
    external_credits: int = Field(default=0, ge=0)
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RoutingMetrics(EvaluationModel):
    """Các metric phân loại route và confusion matrix."""

    sample_count: int = Field(ge=0)
    accuracy: float = Field(ge=0, le=1)
    macro_f1: float = Field(ge=0, le=1)
    balanced_accuracy: float = Field(ge=0, le=1)
    labels: tuple[str, ...]
    confusion_matrix: dict[str, dict[str, int]]


class CalibrationMetrics(EvaluationModel):
    """Các metric calibration multiclass của route probabilities."""

    sample_count: int = Field(ge=0)
    brier_score: float = Field(ge=0)
    negative_log_likelihood: float = Field(ge=0)
    expected_calibration_error: float = Field(ge=0, le=1)
    bin_count: int = Field(ge=2)


class TemperatureCalibrationModel(EvaluationModel):
    """Trạng thái có version của multiclass temperature scaling.

    Model chỉ hợp lệ khi được fit trên calibration split, giúp ngăn test leakage
    ở cả runtime lẫn lúc đọc artifact đã lưu.
    """

    calibrator_version: str = "temperature-scaling-v1"
    labels: tuple[str, ...]
    temperature: float = Field(gt=0)
    fitted_split: DatasetSplit
    sample_count: int = Field(ge=1)
    negative_log_likelihood_before: float = Field(ge=0)
    negative_log_likelihood_after: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_calibration_split(self) -> TemperatureCalibrationModel:
        """Từ chối artifact được fit bằng dev/test split."""

        if self.fitted_split != DatasetSplit.CALIBRATION:
            raise ValueError("calibrator chỉ được fit trên calibration split")
        if not self.labels or len(set(self.labels)) != len(self.labels):
            raise ValueError("labels phải không rỗng và không trùng lặp")
        return self


class OrdinalMetrics(EvaluationModel):
    """Metric dành cho nhãn có thứ tự như độ phức tạp truy vấn.

    Args:
        sample_count: Số quan sát được đánh giá.
        labels: Thứ tự tăng dần của các nhãn.
        mean_absolute_error: Sai số tuyệt đối trung bình theo chỉ số nhãn.
        quadratic_weighted_kappa: Cohen kappa với trọng số bình phương.
    """

    sample_count: int = Field(ge=1)
    labels: tuple[str, ...]
    mean_absolute_error: float = Field(ge=0)
    quadratic_weighted_kappa: float = Field(le=1)


class BinaryRankingMetrics(EvaluationModel):
    """Metric one-vs-rest cho một lớp routing quan trọng.

    AUROC không xác định khi dữ liệu chỉ có một lớp. AUPRC và recall không xác
    định khi không có mẫu dương; các trường hợp đó được biểu diễn bằng ``None``.
    """

    sample_count: int = Field(ge=1)
    positive_label: str = Field(min_length=1)
    positive_count: int = Field(ge=0)
    negative_count: int = Field(ge=0)
    threshold: float = Field(ge=0, le=1)
    auroc: float | None = Field(default=None, ge=0, le=1)
    auprc: float | None = Field(default=None, ge=0, le=1)
    critical_class_recall: float | None = Field(default=None, ge=0, le=1)


class AnswerQualityMetrics(EvaluationModel):
    """Exact match và token-F1 trung bình của câu trả lời."""

    sample_count: int = Field(ge=1)
    exact_match: float = Field(ge=0, le=1)
    token_f1: float = Field(ge=0, le=1)


class CitationMetrics(EvaluationModel):
    """Citation precision/recall dạng micro trên các định danh evidence."""

    sample_count: int = Field(ge=1)
    predicted_citation_count: int = Field(ge=0)
    reference_citation_count: int = Field(ge=0)
    supported_citation_count: int = Field(ge=0)
    precision: float = Field(ge=0, le=1)
    recall: float = Field(ge=0, le=1)
    f1: float = Field(ge=0, le=1)


class RiskCoveragePoint(EvaluationModel):
    """Một điểm của đường risk-coverage tại ngưỡng confidence cụ thể."""

    threshold: float = Field(ge=0, le=1)
    coverage: float = Field(gt=0, le=1)
    risk: float = Field(ge=0, le=1)
    accepted_count: int = Field(ge=1)


class RiskCoverageMetrics(EvaluationModel):
    """Đường risk-coverage và diện tích dưới đường cong."""

    sample_count: int = Field(ge=1)
    points: tuple[RiskCoveragePoint, ...]
    area_under_curve: float = Field(ge=0, le=1)


class RetrievalMetrics(EvaluationModel):
    """Metric evidence recall và phân loại chất lượng context."""

    sample_count: int = Field(ge=1)
    supporting_fact_hits: int = Field(ge=0)
    supporting_fact_count: int = Field(ge=0)
    supporting_fact_recall: float = Field(ge=0, le=1)
    context_quality_macro_f1: float = Field(ge=0, le=1)
    insufficient_recall: float | None = Field(default=None, ge=0, le=1)


class CriterionStatus(StrEnum):
    """Trạng thái đánh giá của một tiêu chí thành công."""

    PASSED = "passed"
    FAILED = "failed"
    NOT_EVALUATED = "not_evaluated"


class CriterionComparator(StrEnum):
    """Phép so sánh dùng để kiểm tra một tiêu chí định lượng."""

    GREATER_OR_EQUAL = ">="
    LESS_OR_EQUAL = "<="
    LESS_THAN = "<"
    PARETO = "pareto"


class SuccessCriteriaInputs(EvaluationModel):
    """Các quan sát cần thiết để đánh giá tiêu chí thành công của MVP.

    Giá trị ``None`` nghĩa là benchmark chưa cung cấp đủ bằng chứng để kết luận.
    Các tỷ lệ dùng thang [0, 1], không dùng phần trăm.
    """

    quality_difference_ci_lower: float | None = None
    jev_mean_cost_usd: float | None = Field(default=None, ge=0)
    llm_mean_cost_usd: float | None = Field(default=None, ge=0)
    jev_p50_latency_ms: float | None = Field(default=None, ge=0)
    llm_p50_latency_ms: float | None = Field(default=None, ge=0)
    jev_p95_latency_ms: float | None = Field(default=None, ge=0)
    llm_p95_latency_ms: float | None = Field(default=None, ge=0)
    critical_class_recall: float | None = Field(default=None, ge=0, le=1)
    expected_calibration_error: float | None = Field(default=None, ge=0, le=1)
    unhandled_router_schema_error_rate: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )
    jev_quality: float | None = Field(default=None, ge=0, le=1)
    rule_quality: float | None = Field(default=None, ge=0, le=1)
    rule_mean_cost_usd: float | None = Field(default=None, ge=0)


class SuccessCriterion(EvaluationModel):
    """Kết quả typed của một tiêu chí thành công riêng lẻ."""

    name: str = Field(min_length=1)
    status: CriterionStatus
    comparator: CriterionComparator
    threshold: float | None = None
    observed: float | None = None
    description: str = Field(min_length=1)


class SuccessCriteriaEvaluation(EvaluationModel):
    """Đánh giá tổng hợp các tiêu chí đã khóa trước benchmark held-out."""

    overall_status: CriterionStatus
    criteria: tuple[SuccessCriterion, ...]


class NumericSummary(EvaluationModel):
    """Thống kê mô tả cho một đại lượng số."""

    count: int = Field(ge=0)
    mean: float | None = None
    minimum: float | None = None
    maximum: float | None = None
    p50: float | None = None
    p95: float | None = None
    p99: float | None = None


class RouterMetrics(EvaluationModel):
    """Metric tổng hợp cho một router trong benchmark run."""

    router: RouterKind
    record_count: int = Field(ge=0)
    routing: RoutingMetrics | None = None
    calibration: CalibrationMetrics | None = None
    ordinal_model_tier: OrdinalMetrics | None = None
    critical_route: BinaryRankingMetrics | None = None
    risk_coverage: RiskCoverageMetrics | None = None
    answer_quality: AnswerQualityMetrics | None = None
    citations: CitationMetrics | None = None
    retrieval: RetrievalMetrics | None = None
    quality: NumericSummary
    total_tokens: NumericSummary
    cost_usd: NumericSummary
    latency_ms: NumericSummary
    error_rate: float = Field(ge=0, le=1)
    abstention_rate: float = Field(ge=0, le=1)
    retrieval_rate: float = Field(ge=0, le=1)
    fallback_rate: float = Field(ge=0, le=1)
    total_external_credits: int = Field(ge=0)


class BootstrapInterval(EvaluationModel):
    """Ước lượng chênh lệch cùng percentile bootstrap confidence interval."""

    estimate: float
    lower: float
    upper: float
    confidence_level: float = Field(gt=0, lt=1)
    resamples: int = Field(ge=1)
    pair_count: int = Field(ge=1)


class PairedComparison(EvaluationModel):
    """So sánh paired theo query giữa candidate và baseline."""

    candidate: RouterKind
    baseline: RouterKind
    quality_difference: BootstrapInterval | None = None
    quality_one_sided_lower_95: float | None = None
    cost_difference_usd: BootstrapInterval | None = None
    latency_difference_ms: BootstrapInterval | None = None


class BenchmarkCheckpoint(EvaluationModel):
    """Snapshot tiến độ tối thiểu để caller ghi checkpoint an toàn."""

    run_id: str
    completed_records: int = Field(ge=0)
    planned_records: int = Field(ge=0)
    denied_by_budget: int = Field(default=0, ge=0)
    failed_records: int = Field(default=0, ge=0)
    stopped_early: bool = False
    stop_reason: str | None = None
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class BenchmarkReport(EvaluationModel):
    """Báo cáo benchmark đầy đủ, sẵn sàng ghi JSONL/Parquet/Markdown."""

    run_id: str
    config: BenchmarkConfig
    started_at: datetime
    finished_at: datetime
    sample_count: int = Field(ge=0)
    records: tuple[BenchmarkRecord, ...]
    router_metrics: tuple[RouterMetrics, ...]
    comparisons: tuple[PairedComparison, ...] = ()
    success_criteria: SuccessCriteriaEvaluation | None = None
    denied_by_budget: int = Field(default=0, ge=0)
    stopped_early: bool = False
    stop_reason: str | None = None


class BenchmarkSplits(EvaluationModel):
    """Ba split dev/calibration/test không giao nhau theo group."""

    dev: tuple[BenchmarkSample, ...]
    calibration: tuple[BenchmarkSample, ...]
    test: tuple[BenchmarkSample, ...]


class CounterfactualOutcome(EvaluationModel):
    """Kết quả của một trong sáu nhánh retrieval-source x model-tier."""

    query_id: str
    retrieval_source: RetrievalSource
    model_tier: ModelTier
    quality_score: float = Field(ge=0, le=1)
    cost_usd: float = Field(ge=0)
    latency_ms: float = Field(ge=0)
    succeeded: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class CounterfactualGoldAction(EvaluationModel):
    """Gold action được chọn từ sáu nhánh counterfactual."""

    query_id: str
    retrieval_source: RetrievalSource
    model_tier: ModelTier
    quality_score: float = Field(ge=0, le=1)
    cost_usd: float = Field(ge=0)
    latency_ms: float = Field(ge=0)
    met_quality_floor: bool
    no_good_route: bool

    @property
    def route_label(self) -> str:
        """Trả nhãn ổn định ghép retrieval source và model tier."""

        return f"{self.retrieval_source.value}:{self.model_tier.value}"


class CounterfactualRunResult(EvaluationModel):
    """Sáu outcomes và gold action của một truy vấn counterfactual."""

    query_id: str = Field(min_length=1)
    outcomes: tuple[CounterfactualOutcome, ...]
    gold_action: CounterfactualGoldAction


class ArtifactPaths(EvaluationModel):
    """Các đường dẫn record, report và visualization đã được writer tạo ra."""

    records_jsonl: Path
    records_parquet: Path
    report_markdown: Path
    pareto_svg: Path
    reliability_svg: Path
    risk_coverage_svg: Path


class LocalDatasetSpec(EvaluationModel):
    """Ánh xạ một file dataset local sang BenchmarkSample chuẩn hóa.

    Args:
        name: Tên dataset ổn định ghi vào record.
        path: File JSON, JSONL hoặc Parquet local; loader không tự download.
        sample_size: Số sample cần lấy sau stratification.
        query_field: Tên cột chứa query.
        id_field: Tên cột chứa query ID; thiếu giá trị sẽ dùng ID xác định theo dòng.
        stratum_fields: Các cột ghép thành stratum.
        group_field: Cột group chống leakage; thiếu giá trị sẽ dùng query ID.
        reference_answer_field: Cột câu trả lời chuẩn tùy chọn.
        expected_route_field: Cột route chuẩn tùy chọn.
    """

    name: str = Field(min_length=1)
    path: Path
    sample_size: int = Field(gt=0)
    query_field: str = "query"
    id_field: str = "query_id"
    stratum_fields: tuple[str, ...] = ("domain",)
    group_field: str = "group_id"
    reference_answer_field: str | None = "reference_answer"
    expected_route_field: str | None = "expected_route"

    @field_validator("stratum_fields")
    @classmethod
    def validate_stratum_fields(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Yêu cầu ít nhất một field để tạo strata tái lập."""

        if not value or any(not item.strip() for item in value):
            raise ValueError("stratum_fields phải chứa ít nhất một tên cột hợp lệ")
        return value


class DatasetBuildConfig(EvaluationModel):
    """Cấu hình dựng bộ 1.000 mẫu từ hai nguồn local và frozen web data."""

    ragrouter_bench: LocalDatasetSpec
    crag: LocalDatasetSpec
    frozen_web_records_path: Path | None = None
    seed: int = 42
    dev_size: int = Field(default=200, ge=0)
    calibration_size: int = Field(default=200, ge=0)
    test_size: int = Field(default=600, ge=0)

    @model_validator(mode="after")
    def validate_expected_plan_sizes(self) -> DatasetBuildConfig:
        """Khóa tỷ lệ nguồn 700/300 và tổng split bằng tổng sample."""

        if self.ragrouter_bench.sample_size != 700 or self.crag.sample_size != 300:
            raise ValueError("benchmark chính thức yêu cầu 700 RAGRouter-Bench và 300 CRAG")
        expected_total = self.ragrouter_bench.sample_size + self.crag.sample_size
        if self.dev_size + self.calibration_size + self.test_size != expected_total:
            raise ValueError("tổng kích thước split phải bằng tổng sample_size")
        return self


class DatasetFileManifest(EvaluationModel):
    """Thông tin provenance của một file dữ liệu local."""

    dataset: str
    path: Path
    sha256: str
    source_record_count: int = Field(ge=0)
    selected_record_count: int = Field(ge=0)


class DatasetManifest(EvaluationModel):
    """Manifest tái lập cho sampling, split và frozen web records."""

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    seed: int
    files: tuple[DatasetFileManifest, ...]
    frozen_web_records_path: Path | None = None
    frozen_web_sha256: str | None = None
    selected_query_ids_sha256: str
    split_sizes: dict[str, int]


class PreparedBenchmarkDataset(EvaluationModel):
    """Dataset đã sample và split cùng manifest provenance."""

    samples: tuple[BenchmarkSample, ...]
    splits: BenchmarkSplits
    manifest: DatasetManifest
