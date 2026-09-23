"""Ánh xạ model tier sang model DeepSeek và bảng giá chuẩn hóa theo snapshot."""

from datetime import date

from pydantic import BaseModel, ConfigDict, Field

from adaptive_rag_router.config.settings import Settings
from adaptive_rag_router.domain.enums import ModelTier
from adaptive_rag_router.domain.models import TokenUsage


class ModelSpec(BaseModel):
    """Thông số model cần cho call và chuẩn hóa variable cost.

    Giá là USD trên một triệu token ở mức peak/cache-miss, được khóa theo ngày
    để benchmark không phụ thuộc thời điểm thực thi.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    tier: ModelTier
    model_id: str
    thinking_enabled: bool
    reasoning_effort: str | None = None
    input_usd_per_million: float = Field(ge=0)
    cached_input_usd_per_million: float = Field(ge=0)
    output_usd_per_million: float = Field(ge=0)
    pricing_effective_at: date

    def calculate_cost(self, usage: TokenUsage) -> float:
        """Tính variable cost theo token usage và pricing snapshot.

        Args:
            usage: Số input, cached-input và output tokens do provider trả về.

        Returns:
            Chi phí chuẩn hóa bằng USD.
        """

        uncached_input = max(usage.input_tokens - usage.cached_input_tokens, 0)
        return (
            uncached_input * self.input_usd_per_million
            + usage.cached_input_tokens * self.cached_input_usd_per_million
            + usage.output_tokens * self.output_usd_per_million
        ) / 1_000_000


class ModelCatalog:
    """Registry bất biến ánh xạ economy/strong tier sang model thực tế."""

    def __init__(self, settings: Settings) -> None:
        """Khởi tạo catalog từ model aliases trong cấu hình."""

        effective_at = date(2026, 9, 10)
        self._specs = {
            ModelTier.ECONOMY: ModelSpec(
                tier=ModelTier.ECONOMY,
                model_id=settings.deepseek_economy_model,
                thinking_enabled=False,
                input_usd_per_million=0.30,
                cached_input_usd_per_million=0.006,
                output_usd_per_million=1.20,
                pricing_effective_at=effective_at,
            ),
            ModelTier.STRONG: ModelSpec(
                tier=ModelTier.STRONG,
                model_id=settings.deepseek_strong_model,
                thinking_enabled=True,
                reasoning_effort="high",
                input_usd_per_million=1.32,
                cached_input_usd_per_million=0.044,
                output_usd_per_million=3.96,
                pricing_effective_at=effective_at,
            ),
        }

    def get(self, tier: ModelTier) -> ModelSpec:
        """Trả cấu hình model tương ứng với logical tier."""

        return self._specs[tier]

    def as_manifest(self) -> dict[str, dict[str, str | float | bool | None]]:
        """Xuất metadata không chứa secret để đưa vào benchmark manifest."""

        return {
            tier.value: {
                "model_id": spec.model_id,
                "thinking_enabled": spec.thinking_enabled,
                "reasoning_effort": spec.reasoning_effort,
                "input_usd_per_million": spec.input_usd_per_million,
                "cached_input_usd_per_million": spec.cached_input_usd_per_million,
                "output_usd_per_million": spec.output_usd_per_million,
                "pricing_effective_at": spec.pricing_effective_at.isoformat(),
            }
            for tier, spec in self._specs.items()
        }
