"""Public API của subsystem retrieval cho vector, mock web và Tavily live."""

from adaptive_rag_router.retrieval.base import (
    Retriever,
    RetrieverRegistry,
    canonicalize_url,
    deduplicate_documents,
    merge_documents,
    normalize_query,
)
from adaptive_rag_router.retrieval.chunking import (
    HuggingFaceTokenCodec,
    SourceDocument,
    TextChunker,
    TokenCodec,
    WhitespaceTokenCodec,
)
from adaptive_rag_router.retrieval.embeddings import (
    BgeM3EmbeddingAdapter,
    EmbeddingAdapter,
)
from adaptive_rag_router.retrieval.in_memory import InMemoryRetriever, MockWebRetriever
from adaptive_rag_router.retrieval.mock_file import JsonlMockWebRetriever
from adaptive_rag_router.retrieval.qdrant import QdrantVectorRetriever
from adaptive_rag_router.retrieval.tavily import (
    CreditLedger,
    FileSearchCache,
    HttpxTavilyTransport,
    InMemorySearchCache,
    RollingWindowRateLimiter,
    SearchCache,
    TavilyRateLimitError,
    TavilyRetriever,
    TavilySearchTransport,
    build_tavily_cache_key,
)

__all__ = [
    "BgeM3EmbeddingAdapter",
    "CreditLedger",
    "EmbeddingAdapter",
    "FileSearchCache",
    "HttpxTavilyTransport",
    "HuggingFaceTokenCodec",
    "InMemoryRetriever",
    "InMemorySearchCache",
    "JsonlMockWebRetriever",
    "MockWebRetriever",
    "QdrantVectorRetriever",
    "Retriever",
    "RetrieverRegistry",
    "RollingWindowRateLimiter",
    "SearchCache",
    "SourceDocument",
    "TavilyRateLimitError",
    "TavilyRetriever",
    "TavilySearchTransport",
    "TextChunker",
    "TokenCodec",
    "WhitespaceTokenCodec",
    "build_tavily_cache_key",
    "canonicalize_url",
    "deduplicate_documents",
    "merge_documents",
    "normalize_query",
]
