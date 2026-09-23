"""Khởi tạo decision engine và cô lập import SDK provider khỏi domain logic."""

from typing import Any, cast

from adaptive_rag_router.config import Settings
from adaptive_rag_router.domain import ModelTier, RouterKind
from adaptive_rag_router.domain.errors import ConfigurationError
from adaptive_rag_router.generation.catalog import ModelCatalog
from adaptive_rag_router.routing.base import DecisionEngine
from adaptive_rag_router.routing.jev import JevClient, JevRouter
from adaptive_rag_router.routing.llm import LLMRouter
from adaptive_rag_router.routing.rule_based import RuleBasedRouter


def _secret_value(secret: Any, variable_name: str) -> str:
    if secret is None:
        raise ConfigurationError(f"Thiếu {variable_name} cho live decision engine")
    value = cast(str, secret.get_secret_value()).strip()
    if not value:
        raise ConfigurationError(f"{variable_name} không được rỗng")
    return value


def build_decision_engine(
    kind: RouterKind | str,
    settings: Settings,
    *,
    llm_client: Any | None = None,
    jev_client: JevClient | None = None,
) -> DecisionEngine:
    """Tạo router theo cấu hình mà không thực hiện network request.

    Args:
        kind: Loại router cần khởi tạo.
        settings: Cấu hình model, endpoint, timeout và secret.
        llm_client: OpenAI-compatible client đã inject, chủ yếu cho test/DI.
        jev_client: System One client đã inject, chủ yếu cho test/DI.

    Returns:
        Decision engine tuân thủ contract chung.

    Raises:
        ConfigurationError: Khi loại router, SDK hoặc API key không hợp lệ.
    """

    try:
        router_kind = RouterKind(kind)
    except ValueError as error:
        raise ConfigurationError(f"Router kind không được hỗ trợ: {kind!r}") from error

    if router_kind is RouterKind.RULE:
        return RuleBasedRouter(max_repair_rounds=settings.max_repair_rounds)

    if router_kind is RouterKind.LLM:
        client = llm_client
        if client is None:
            api_key = _secret_value(settings.deepseek_api_key, "DEEPSEEK_API_KEY")
            try:
                from openai import AsyncOpenAI
            except ImportError as error:
                raise ConfigurationError("Chưa cài dependency openai") from error
            client = AsyncOpenAI(
                api_key=api_key,
                base_url=settings.deepseek_base_url,
                timeout=settings.request_timeout_seconds,
            )
        return LLMRouter(
            client,
            model=settings.deepseek_router_model,
            timeout_seconds=settings.request_timeout_seconds,
            pricing=ModelCatalog(settings).get(ModelTier.ECONOMY),
        )

    if jev_client is None:
        api_key = _secret_value(settings.typesafe_api_key, "TYPESAFE_API_KEY")
        try:
            from typesafe_sdk import AsyncTypeSafeClient
        except ImportError as error:
            raise ConfigurationError("Chưa cài dependency typesafe-sdk") from error
        client_jev = cast(
            JevClient,
            AsyncTypeSafeClient(
                api_key=api_key,
                model=settings.typesafe_model,
                base_url=settings.typesafe_base_url,
                timeout=settings.request_timeout_seconds,
            ),
        )
    else:
        client_jev = jev_client
    return JevRouter(
        client_jev,
        model=settings.typesafe_model,
        timeout_seconds=settings.request_timeout_seconds,
        input_usd_per_million=settings.typesafe_input_usd_per_million,
    )
