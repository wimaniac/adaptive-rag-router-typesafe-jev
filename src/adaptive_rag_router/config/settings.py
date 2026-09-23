"""Đọc cấu hình an toàn từ environment và cung cấp defaults có version."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Cấu hình ứng dụng; secret được che khi serialize hoặc log.

    Các tham số provider có default phục vụ research MVP. API key là tùy chọn ở
    thời điểm khởi tạo để unit test offline không cần secret; adapter live sẽ tự
    kiểm tra key trước khi gọi provider.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    typesafe_api_key: SecretStr | None = None
    deepseek_api_key: SecretStr | None = None
    tavily_api_key: SecretStr | None = None
    hf_token: SecretStr | None = None

    typesafe_base_url: str = "https://api.typesafe.ai"
    typesafe_model: str = "jev-1.13.0"
    typesafe_input_usd_per_million: float = Field(default=0.042, ge=0)
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_router_model: str = "deepseek-flash"
    deepseek_economy_model: str = "deepseek-flash"
    deepseek_strong_model: str = "deepseek-v4-pro"
    embedding_model: str = "BAAI/bge-m3"

    qdrant_path: Path = Path("data/qdrant")
    vector_collection: str = "adaptive_rag_documents"
    cache_dir: Path = Path(".cache/adaptive-rag-router")
    artifacts_dir: Path = Path("artifacts")

    vector_top_k: int = Field(default=12, ge=1)
    evidence_limit: int = Field(default=8, ge=1)
    web_max_results: int = Field(default=5, ge=1, le=20)
    max_repair_rounds: int = Field(default=2, ge=0, le=5)
    max_output_tokens: int = Field(default=1024, ge=1, le=8192)

    tavily_requests_per_minute: int = Field(default=90, ge=1, le=100)
    tavily_max_concurrency: int = Field(default=4, ge=1, le=10)
    tavily_credit_budget: int = Field(default=350, ge=1, le=1000)
    tavily_max_retries: int = Field(default=4, ge=0, le=8)

    request_timeout_seconds: float = Field(default=60.0, gt=0)
    show_trace_by_default: bool = False


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Trả singleton cấu hình đã đọc từ `.env` và environment.

    Returns:
        Cấu hình runtime đã được Pydantic kiểm tra.
    """

    return Settings()
