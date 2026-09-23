"""Kiểm thử helper chunk tài liệu phục vụ vector ingestion."""

from collections.abc import Sequence

from adaptive_rag_router.retrieval import HuggingFaceTokenCodec, SourceDocument, TextChunker


class _FakeTokenizer:
    """Tokenizer ký tự nhỏ dùng để kiểm tra codec mà không tải model."""

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        """Chuyển từng ký tự thành mã số ổn định."""

        assert add_special_tokens is False
        return [ord(character) for character in text]

    def decode(
        self,
        tokens: Sequence[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        """Khôi phục chuỗi ký tự từ mã số."""

        assert skip_special_tokens is True
        assert clean_up_tokenization_spaces is False
        return "".join(chr(token) for token in tokens)


def test_chunker_uses_overlap_and_lineage_metadata() -> None:
    chunker = TextChunker(chunk_size=4, overlap=1)
    source = SourceDocument(document_id="doc", text="a b c d e f g", metadata={"domain": "x"})

    chunks = chunker.chunk_document(source)

    assert [chunk.text for chunk in chunks] == ["a b c d", "d e f g"]
    assert chunks[1].metadata == {
        "domain": "x",
        "parent_document_id": "doc",
        "chunk_index": 1,
        "chunk_count": 2,
    }


def test_chunker_rejects_overlap_equal_to_chunk_size() -> None:
    try:
        TextChunker(chunk_size=4, overlap=4)
    except ValueError as error:
        assert "overlap" in str(error)
    else:
        raise AssertionError("TextChunker phải từ chối overlap không hợp lệ")


def test_huggingface_codec_supports_true_token_windows_without_network() -> None:
    """Codec tokenizer thật có thể inject và giữ chính xác cửa sổ/overlap."""

    codec = HuggingFaceTokenCodec(tokenizer=_FakeTokenizer())
    chunker = TextChunker(chunk_size=4, overlap=1, codec=codec)

    assert chunker.split("abcdef") == ("abcd", "def")
