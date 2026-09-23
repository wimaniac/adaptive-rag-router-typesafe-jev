"""Cung cấp import path ổn định cho registry các retrieval adapter."""

from adaptive_rag_router.retrieval.base import Retriever, RetrieverRegistry

__all__ = ["Retriever", "RetrieverRegistry"]
