"""Tests for the text extraction adapter interface."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.services.text_extraction import (
    ExtractedTextResult,
    MarkdownAdapter,
    PdfAdapter,
    PlainTextAdapter,
    TextExtractionAdapter,
    TextExtractionDecodeError,
    TextExtractionFileNotFoundError,
    TextExtractionParseError,
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


def test_pdf_adapter_satisfies_protocol() -> None:
    assert isinstance(PdfAdapter(), TextExtractionAdapter)


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
    raw = f.read_bytes()

    result = await MarkdownAdapter().extract(f)

    assert result.metadata["extractor"] == "MarkdownAdapter"
    assert result.metadata["source_extension"] == ".md"
    assert result.metadata["mime_type"] == "text/markdown"
    assert result.metadata["byte_size"] == len(raw)
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
# PDF extraction
# ---------------------------------------------------------------------------


def _write_simple_pdf(path: Path, lines: list[str]) -> None:
    """Write a minimal single-page PDF containing *lines* of text."""
    # Escape parentheses/backslashes for PDF string literals.
    escaped = []
    for line in lines:
        escaped.append(
            line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        )

    content_ops = ["BT", "/F1 12 Tf", "50 750 Td"]
    for index, line in enumerate(escaped):
        if index == 0:
            content_ops.append(f"({line}) Tj")
        else:
            content_ops.append(f"0 -16 Td ({line}) Tj")
    content_ops.append("ET")
    stream = "\n".join(content_ops)
    stream_bytes = stream.encode("latin-1")

    objects = [
        "1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n",
        "2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n",
        (
            "3 0 obj\n"
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            "/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\n"
            "endobj\n"
        ),
        (
            f"4 0 obj\n<< /Length {len(stream_bytes)} >>\nstream\n"
            f"{stream}\nendstream\nendobj\n"
        ),
        "5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n",
    ]

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for obj in objects:
        offsets.append(len(pdf))
        pdf.extend(obj.encode("latin-1"))

    xref_pos = len(pdf)
    pdf.extend(f"xref\n0 {len(offsets)}\n".encode("latin-1"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("latin-1"))
    pdf.extend(
        (
            f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\n"
            f"startxref\n{xref_pos}\n%%EOF\n"
        ).encode("latin-1")
    )
    path.write_bytes(bytes(pdf))


async def test_pdf_extract_returns_result(tmp_path: Path) -> None:
    f = tmp_path / "invoice.pdf"
    _write_simple_pdf(f, ["Hello PDF", "Second line"])

    result = await PdfAdapter().extract(f)

    assert isinstance(result, ExtractedTextResult)
    assert "Hello PDF" in result.text
    assert result.page_count == 1


async def test_pdf_extract_metadata_keys(tmp_path: Path) -> None:
    f = tmp_path / "notes.pdf"
    _write_simple_pdf(f, ["Notes page"])
    raw = f.read_bytes()

    result = await PdfAdapter().extract(f)

    assert result.metadata["extractor"] == "PdfAdapter"
    assert result.metadata["source_extension"] == ".pdf"
    assert result.metadata["mime_type"] == "application/pdf"
    assert result.metadata["byte_size"] == len(raw)
    assert result.metadata["page_count"] == 1
    assert result.metadata["char_count"] == len(result.text)


async def test_pdf_extract_missing_file_raises(tmp_path: Path) -> None:
    missing = tmp_path / "missing.pdf"
    with pytest.raises(TextExtractionFileNotFoundError, match="missing.pdf"):
        await PdfAdapter().extract(missing)


async def test_pdf_extract_invalid_pdf_raises(tmp_path: Path) -> None:
    f = tmp_path / "corrupt.pdf"
    f.write_bytes(b"not a real pdf")

    with pytest.raises(TextExtractionParseError):
        await PdfAdapter().extract(f)


async def test_extract_text_from_file_pdf(tmp_path: Path) -> None:
    f = tmp_path / "contract.pdf"
    _write_simple_pdf(f, ["Contract clause one"])

    result = await extract_text_from_file(f, mime_type="application/pdf")

    assert "Contract clause one" in result.text
    assert result.metadata["extractor"] == "PdfAdapter"


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


def test_resolver_picks_pdf_by_mime() -> None:
    adapter = _resolve_adapter(mime_type="application/pdf")
    assert isinstance(adapter, PdfAdapter)


def test_resolver_picks_pdf_by_extension() -> None:
    adapter = _resolve_adapter(filename="document.pdf")
    assert isinstance(adapter, PdfAdapter)


def test_resolver_mime_takes_precedence_over_extension() -> None:
    # Passing a text/plain MIME with a .md filename should resolve to plain text.
    adapter = _resolve_adapter(mime_type="text/plain", filename="doc.md")
    assert isinstance(adapter, PlainTextAdapter)


def test_resolver_strips_mime_charset_parameter() -> None:
    adapter = _resolve_adapter(mime_type="text/plain; charset=utf-8")
    assert isinstance(adapter, PlainTextAdapter)


def test_resolver_unsupported_mime_raises() -> None:
    with pytest.raises(UnsupportedTextExtractionTypeError, match="application/zip"):
        _resolve_adapter(mime_type="application/zip")


def test_resolver_unsupported_extension_raises() -> None:
    with pytest.raises(UnsupportedTextExtractionTypeError, match="archive.zip"):
        _resolve_adapter(filename="archive.zip")


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
