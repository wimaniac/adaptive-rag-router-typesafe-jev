"""Kiểm thử model catalog và phép tính normalized token cost."""

import pytest

from adaptive_rag_router.config.settings import Settings
from adaptive_rag_router.domain.enums import ModelTier
from adaptive_rag_router.domain.models import TokenUsage
from adaptive_rag_router.generation.catalog import ModelCatalog


def test_catalog_maps_logical_tiers() -> None:
    """Economy và strong tier phải ánh xạ đúng aliases đã cấu hình."""

    catalog = ModelCatalog(Settings(_env_file=None))

    assert catalog.get(ModelTier.ECONOMY).model_id == "deepseek-flash"
    assert catalog.get(ModelTier.STRONG).model_id == "deepseek-v4-pro"


def test_cost_separates_cached_input_tokens() -> None:
    """Cached input được tính theo rate riêng thay vì tính hai lần."""

    spec = ModelCatalog(Settings(_env_file=None)).get(ModelTier.ECONOMY)
    usage = TokenUsage(input_tokens=1_000_000, cached_input_tokens=500_000, output_tokens=100)

    cost = spec.calculate_cost(usage)

    expected = 0.5 * spec.input_usd_per_million + 0.5 * spec.cached_input_usd_per_million
    expected += 0.0001 * spec.output_usd_per_million
    assert cost == pytest.approx(expected)
