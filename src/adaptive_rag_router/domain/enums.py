"""Định nghĩa các enum ổn định cho quyết định, trạng thái và nguồn dữ liệu."""

from enum import StrEnum


class RouterKind(StrEnum):
    """Loại decision engine được dùng để điều phối truy vấn."""

    JEV = "jev"
    RULE = "rule"
    LLM = "llm"


class RetrievalSource(StrEnum):
    """Nguồn retrieval loại trừ lẫn nhau cho một vòng tìm kiếm."""

    NONE = "none"
    VECTOR = "vector"
    WEB = "web"


class Complexity(StrEnum):
    """Mức độ phức tạp ngữ nghĩa và suy luận của truy vấn."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ModelTier(StrEnum):
    """Tầng model logic, tách biệt với tên model của provider."""

    ECONOMY = "economy"
    STRONG = "strong"


class ContextQuality(StrEnum):
    """Mức đầy đủ của evidence sau retrieval và lọc context."""

    INSUFFICIENT = "insufficient"
    PARTIAL = "partial"
    SUFFICIENT = "sufficient"


class RepairAction(StrEnum):
    """Hành động tiếp theo của vòng adaptive retrieval."""

    STOP = "stop"
    REWRITE_VECTOR = "rewrite_vector"
    SWITCH_WEB = "switch_web"
    REFINE_WEB = "refine_web"
    ABSTAIN = "abstain"


class FallbackAction(StrEnum):
    """Hành động sau khi decision engine đánh giá draft answer."""

    ACCEPT = "accept"
    REGENERATE_STRONG = "regenerate_strong"
    ABSTAIN = "abstain"


class ProbabilityProvenance(StrEnum):
    """Nguồn gốc của distribution để tránh diễn giải sai confidence."""

    NATIVE = "native"
    SELF_REPORTED = "self_reported"
    HEURISTIC = "heuristic"
    CALIBRATED = "calibrated"


class CompletionStatus(StrEnum):
    """Trạng thái kết thúc của một lần xử lý truy vấn."""

    COMPLETED = "completed"
    ABSTAINED = "abstained"
    DEGRADED = "degraded"
    FAILED = "failed"
