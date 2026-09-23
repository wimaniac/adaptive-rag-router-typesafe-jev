"""Chuẩn hóa lỗi provider và lỗi policy để orchestration xử lý nhất quán."""


class AdaptiveRAGError(Exception):
    """Lỗi gốc của hệ thống Adaptive RAG Router."""


class ConfigurationError(AdaptiveRAGError):
    """Cấu hình thiếu, không hợp lệ hoặc không an toàn."""


class ProviderError(AdaptiveRAGError):
    """Lỗi đã chuẩn hóa khi gọi một provider bên ngoài."""


class ProviderRateLimitError(ProviderError):
    """Provider từ chối request vì rate limit hoặc quota."""


class CreditBudgetExceededError(ProviderRateLimitError):
    """Request bị chặn trước khi vượt credit budget đã cấu hình."""


class DecisionSchemaError(AdaptiveRAGError):
    """Decision engine trả kết quả không khớp typed contract."""


class EvidenceUnavailableError(AdaptiveRAGError):
    """Không còn nguồn evidence hợp lệ để tiếp tục pipeline."""
