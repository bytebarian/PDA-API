"""Tests for the text extraction adapter interface."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.services.text_extraction import (
    ExtractedTextResult,
    MarkdownAdapter,
    PlainTextAdapter,
    TextExtractionAdapter,
    TextExtractionDecodeError,
    TextExtractionFileNotFoundError,
    UnsupportedTextExtractionTypeError,
    _resolve_adapter,
    extract_text_from_file,
)


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_plain_text_adapter_satisfies_protocol() -> None:
    assert isinstance(PlainTextAdapter(), TextExtractionAdapter)


def test_markdown_adapter_satisfies_protocol() -> None:
    assert isinstance(MarkdownAdapter(), TextExtractionAdapter)


# ---------------------------------------------------------------------------
# Plain text extraction
# ---------------------------------------------------------------------------


async def test_plain_text_extract_returns_result(tmp_path: Path) -> None:
    f = tmp_path / "hello.txt"
    f.write_text("Hello, world!", encoding="utf-8")

    result = await PlainTextAdapter().extract(f)

    assert isinstance(result, ExtractedTextResult)
    assert result.text == "Hello, world!"


async def test_plain_text_extract_normalises_crlf(tmp_path: Path) -> None:
    f = tmp_path / "crlf.txt"
    f.write_bytes(b"line1\r\nline2\r\n")

    result = await PlainTextAdapter().extract(f)

    assert result.text == "line1\nline2\n"


async def test_plain_text_extract_metadata_keys(tmp_path: Path) -> None:
    f = tmp_path / "doc.txt"
    content = "some text"
    f.write_text(content, encoding="utf-8")

    result = await PlainTextAdapter().extract(f)

    assert result.metadata["extractor"] == "PlainTextAdapter"
    assert result.metadata["source_extension"] == ".txt"
    assert result.metadata["mime_type"] == "text/plain"
    assert result.metadata["byte_size"] == len(content.encode("utf-8"))
    assert result.metadata["char_count"] == len(content)


async def test_plain_text_extract_missing_file_raises(tmp_path: Path) -> None:
    missing = tmp_path / "nope.txt"
    with pytest.raises(TextExtractionFileNotFoundError, match="nope.txt"):
        await PlainTextAdapter().extract(missing)


async def test_plain_text_extract_decode_error_raises(tmp_path: Path) -> None:
    f = tmp_path / "bad.txt"
    f.write_bytes(b"\xff\xfe invalid utf-8 \x80\x81")

    with pytest.raises(TextExtractionDecodeError):
        await PlainTextAdapter().extract(f)


# ---------------------------------------------------------------------------
# Markdown extraction
# ---------------------------------------------------------------------------


async def test_markdown_extract_returns_result(tmp_path: Path) -> None:
    f = tmp_path / "readme.md"
    f.write_text("# Title\n\nSome text.", encoding="utf-8")

    result = await MarkdownAdapter().extract(f)

    assert isinstance(result, ExtractedTextResult)
    assert "# Title" in result.text


async def test_markdown_extract_metadata_keys(tmp_path: Path) -> None:
    f = tmp_path / "notes.md"
    content = "## Notes\n\nDetails here."
    f.write_text(content, encoding="utf-8")

    result = await MarkdownAdapter().extract(f)

    assert result.metadata["extractor"] == "MarkdownAdapter"
    assert result.metadata["source_extension"] == ".md"
    assert result.metadata["mime_type"] == "text/markdown"
    assert result.metadata["byte_size"] == len(content.encode("utf-8"))
    assert result.metadata["char_count"] == len(content)


async def test_markdown_extract_dot_markdown_extension(tmp_path: Path) -> None:
    f = tmp_path / "doc.markdown"
    f.write_text("content", encoding="utf-8")

    result = await MarkdownAdapter().extract(f)

    assert result.metadata["source_extension"] == ".markdown"


async def test_markdown_extract_missing_file_raises(tmp_path: Path) -> None:
    missing = tmp_path / "missing.md"
    with pytest.raises(TextExtractionFileNotFoundError, match="missing.md"):
        await MarkdownAdapter().extract(missing)


async def test_markdown_extract_decode_error_raises(tmp_path: Path) -> None:
    f = tmp_path / "bad.md"
    f.write_bytes(b"\xff\xfe bad encoding \x80")

    with pytest.raises(TextExtractionDecodeError):
        await MarkdownAdapter().extract(f)


# ---------------------------------------------------------------------------
# Resolver behaviour
# ---------------------------------------------------------------------------


def test_resolver_picks_plain_text_by_mime() -> None:
    adapter = _resolve_adapter(mime_type="text/plain")
    assert isinstance(adapter, PlainTextAdapter)


def test_resolver_picks_plain_text_by_extension() -> None:
    adapter = _resolve_adapter(filename="notes.txt")
    assert isinstance(adapter, PlainTextAdapter)


def test_resolver_picks_markdown_by_mime() -> None:
    adapter = _resolve_adapter(mime_type="text/markdown")
    assert isinstance(adapter, MarkdownAdapter)


def test_resolver_picks_markdown_by_x_mime() -> None:
    adapter = _resolve_adapter(mime_type="text/x-markdown")
    assert isinstance(adapter, MarkdownAdapter)


def test_resolver_picks_markdown_by_md_extension() -> None:
    adapter = _resolve_adapter(filename="readme.md")
    assert isinstance(adapter, MarkdownAdapter)


def test_resolver_picks_markdown_by_markdown_extension() -> None:
    adapter = _resolve_adapter(filename="doc.markdown")
    assert isinstance(adapter, MarkdownAdapter)


def test_resolver_mime_takes_precedence_over_extension() -> None:
    # Passing a text/plain MIME with a .md filename should resolve to plain text.
    adapter = _resolve_adapter(mime_type="text/plain", filename="doc.md")
    assert isinstance(adapter, PlainTextAdapter)


def test_resolver_strips_mime_charset_parameter() -> None:
    adapter = _resolve_adapter(mime_type="text/plain; charset=utf-8")
    assert isinstance(adapter, PlainTextAdapter)


def test_resolver_unsupported_mime_raises() -> None:
    with pytest.raises(UnsupportedTextExtractionTypeError, match="application/pdf"):
        _resolve_adapter(mime_type="application/pdf")


def test_resolver_unsupported_extension_raises() -> None:
    with pytest.raises(UnsupportedTextExtractionTypeError, match="document.pdf"):
        _resolve_adapter(filename="document.pdf")


def test_resolver_no_hints_raises() -> None:
    with pytest.raises(UnsupportedTextExtractionTypeError):
        _resolve_adapter()


# ---------------------------------------------------------------------------
# High-level helper
# ---------------------------------------------------------------------------


async def test_extract_text_from_file_plain_text(tmp_path: Path) -> None:
    f = tmp_path / "test.txt"
    f.write_text("plain content", encoding="utf-8")

    result = await extract_text_from_file(f, mime_type="text/plain")

    assert result.text == "plain content"
    assert result.metadata["extractor"] == "PlainTextAdapter"


async def test_extract_text_from_file_infers_adapter_from_path(
    tmp_path: Path,
) -> None:
    f = tmp_path / "readme.md"
    f.write_text("# Hello", encoding="utf-8")

    # No mime_type supplied – resolver should use the path's extension.
    result = await extract_text_from_file(f)

    assert "# Hello" in result.text
    assert result.metadata["extractor"] == "MarkdownAdapter"


async def test_extract_text_from_file_filename_hint_overrides_path_extension(
    tmp_path: Path,
) -> None:
    # File on disk has no extension; hint says it's a .txt file.
    f = tmp_path / "datafile"
    f.write_text("raw data", encoding="utf-8")

    result = await extract_text_from_file(f, filename="datafile.txt")

    assert result.text == "raw data"
    assert result.metadata["extractor"] == "PlainTextAdapter"


async def test_extract_text_from_file_missing_raises(tmp_path: Path) -> None:
    missing = tmp_path / "ghost.txt"
    with pytest.raises(TextExtractionFileNotFoundError):
        await extract_text_from_file(missing)


async def test_extract_text_from_file_unsupported_type_raises(
    tmp_path: Path,
) -> None:
    f = tmp_path / "archive.zip"
    f.write_bytes(b"PK\x03\x04")

    with pytest.raises(UnsupportedTextExtractionTypeError):
        await extract_text_from_file(f, mime_type="application/zip")


async def test_extract_text_from_file_passes_document_to_adapter(
    tmp_path: Path,
) -> None:
    """Verify that Document is forwarded without causing errors."""
    from app.models.document import Document

    f = tmp_path / "with_doc.txt"
    f.write_text("content", encoding="utf-8")

    doc = Document(filename="with_doc.txt")
    result = await extract_text_from_file(f, document=doc)

    assert result.text == "content"


# ---------------------------------------------------------------------------
# ExtractedTextResult immutability
# ---------------------------------------------------------------------------


def test_extracted_text_result_is_frozen() -> None:
    result = ExtractedTextResult(text="hi", metadata={"k": "v"})
    with pytest.raises((AttributeError, TypeError)):
        result.text = "bye"  # type: ignore[misc]
