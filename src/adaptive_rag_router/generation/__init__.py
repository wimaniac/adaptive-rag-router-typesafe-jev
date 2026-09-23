"""Các adapter sinh câu trả lời và rewrite truy vấn bằng model DeepSeek."""

from adaptive_rag_router.generation.catalog import ModelCatalog, ModelSpec
from adaptive_rag_router.generation.deepseek import DeepSeekGenerator
from adaptive_rag_router.generation.ports import Generator
from adaptive_rag_router.generation.rewriter import (
    DeepSeekQueryRewriter,
    QueryRewriter,
    QueryRewriteResult,
)

__all__ = [
    "DeepSeekGenerator",
    "DeepSeekQueryRewriter",
    "Generator",
    "ModelCatalog",
    "ModelSpec",
    "QueryRewriteResult",
    "QueryRewriter",
]
