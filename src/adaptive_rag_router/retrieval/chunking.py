"""Định nghĩa helper chia tài liệu thành passage có overlap để ingest vector store."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Protocol, cast

from adaptive_rag_router.domain.enums import RetrievalSource
from adaptive_rag_router.domain.models import RetrievedDocument

type Token = int | str


class TokenCodec(Protocol):
    """Contract tối thiểu để chunk theo token mà không khóa vào một tokenizer."""

    def encode(self, text: str) -> Sequence[Token]:
        """Mã hóa văn bản thành token.

        Args:
            text: Văn bản nguồn.

        Returns:
            Chuỗi token ổn định.
        """

    def decode(self, tokens: Sequence[Token]) -> str:
        """Giải mã token thành đoạn văn bản.

        Args:
            tokens: Chuỗi token cần giải mã.

        Returns:
            Văn bản của chunk.
        """


class WhitespaceTokenCodec:
    """Codec nhẹ dùng từ phân tách bởi khoảng trắng làm token.

    Benchmark chính có thể inject codec của tokenizer BGE-M3 để giới hạn 512 là
    token thật; codec này giữ unit test và ingestion nhỏ không cần tải model.
    """

    def encode(self, text: str) -> Sequence[Token]:
        """Tách văn bản theo khoảng trắng.

        Args:
            text: Văn bản nguồn.

        Returns:
            Danh sách token dạng chuỗi.
        """

        return text.split()

    def decode(self, tokens: Sequence[Token]) -> str:
        """Ghép token bằng một khoảng trắng.

        Args:
            tokens: Token của một chunk.

        Returns:
            Văn bản chunk đã ghép.
        """

        return " ".join(str(token) for token in tokens)


class TransformerTokenizer(Protocol):
    """Bề mặt tokenizer tối thiểu cần cho chunking theo token thật."""

    def encode(self, text: str, *, add_special_tokens: bool) -> Sequence[int]:
        """Mã hóa text mà không thêm token điều khiển."""

    def decode(
        self,
        tokens: Sequence[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        """Giải mã token về passage có thể đưa vào embedding model."""


class HuggingFaceTokenCodec:
    """Codec lazy-load tokenizer của BGE-M3 để chunk đúng 512/64 token."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        *,
        tokenizer: TransformerTokenizer | None = None,
    ) -> None:
        """Khởi tạo codec mà chưa tải tokenizer nếu caller không inject.

        Args:
            model_name: Hugging Face model ID dùng cùng embedding adapter.
            tokenizer: Tokenizer inject cho test hoặc môi trường đã preload.
        """

        self._model_name = model_name
        self._tokenizer = tokenizer
        self._load_lock = Lock()

    def encode(self, text: str) -> Sequence[Token]:
        """Mã hóa văn bản bằng tokenizer của embedding model."""

        return tuple(self._get_tokenizer().encode(text, add_special_tokens=False))

    def decode(self, tokens: Sequence[Token]) -> str:
        """Giải mã một cửa sổ token, bỏ token điều khiển nếu có."""

        integer_tokens = tuple(int(token) for token in tokens)
        return self._get_tokenizer().decode(
            integer_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    def _get_tokenizer(self) -> TransformerTokenizer:
        if self._tokenizer is not None:
            return self._tokenizer
        with self._load_lock:
            if self._tokenizer is None:
                from transformers import AutoTokenizer

                self._tokenizer = cast(
                    TransformerTokenizer,
                    AutoTokenizer.from_pretrained(self._model_name),
                )
        return self._tokenizer


@dataclass(frozen=True, slots=True)
class SourceDocument:
    """Tài liệu thô trước khi chia chunk và chuyển thành evidence."""

    document_id: str
    text: str
    title: str | None = None
    url: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


class TextChunker:
    """Chia tài liệu theo cửa sổ token với overlap cố định."""

    def __init__(
        self,
        *,
        chunk_size: int = 512,
        overlap: int = 64,
        codec: TokenCodec | None = None,
    ) -> None:
        """Khởi tạo chunker.

        Args:
            chunk_size: Số token tối đa trong một chunk.
            overlap: Số token lặp lại giữa hai chunk liên tiếp.
            codec: Tokenizer adapter; mặc định là codec theo khoảng trắng.

        Raises:
            ValueError: Nếu kích thước không hợp lệ hoặc overlap không nhỏ hơn chunk.
        """

        if chunk_size <= 0:
            raise ValueError("chunk_size phải lớn hơn 0")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap phải không âm và nhỏ hơn chunk_size")
        self._chunk_size = chunk_size
        self._overlap = overlap
        self._codec = codec or WhitespaceTokenCodec()

    def split(self, text: str) -> tuple[str, ...]:
        """Chia một văn bản thành các chunk.

        Args:
            text: Văn bản cần chia.

        Returns:
            Các chunk không rỗng theo thứ tự ban đầu.
        """

        tokens = list(self._codec.encode(text.strip()))
        if not tokens:
            return ()
        step = self._chunk_size - self._overlap
        chunks: list[str] = []
        for start in range(0, len(tokens), step):
            token_slice = tokens[start : start + self._chunk_size]
            decoded = self._codec.decode(token_slice).strip()
            if decoded:
                chunks.append(decoded)
            if start + self._chunk_size >= len(tokens):
                break
        return tuple(chunks)

    def chunk_document(
        self,
        document: SourceDocument,
        *,
        source: RetrievalSource = RetrievalSource.VECTOR,
    ) -> tuple[RetrievedDocument, ...]:
        """Chuyển tài liệu thô thành các `RetrievedDocument` có lineage.

        Args:
            document: Tài liệu thô.
            source: Nguồn logic gắn vào passage.

        Returns:
            Các passage kèm parent ID và chunk index trong metadata.
        """

        chunks = self.split(document.text)
        return tuple(
            RetrievedDocument(
                document_id=f"{document.document_id}#chunk-{index:04d}",
                text=chunk,
                title=document.title,
                url=document.url,
                source=source,
                metadata={
                    **dict(document.metadata),
                    "parent_document_id": document.document_id,
                    "chunk_index": index,
                    "chunk_count": len(chunks),
                },
            )
            for index, chunk in enumerate(chunks)
        )

    def chunk_documents(
        self,
        documents: Iterable[SourceDocument],
        *,
        source: RetrievalSource = RetrievalSource.VECTOR,
    ) -> tuple[RetrievedDocument, ...]:
        """Chia một batch tài liệu và giữ thứ tự đầu vào.

        Args:
            documents: Các tài liệu thô.
            source: Nguồn logic gắn vào passage.

        Returns:
            Toàn bộ passage theo thứ tự tài liệu và chunk.
        """

        return tuple(
            chunk
            for document in documents
            for chunk in self.chunk_document(document, source=source)
        )
