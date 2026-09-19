"""Render PDF pages to images for local OCR processing."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable


class PDFRenderError(RuntimeError):
    """Raised when a PDF cannot be rendered for OCR."""


@runtime_checkable
class PDFRenderer(Protocol):
    """Contract for rendering PDF pages as image files."""

    async def render_pages(
        self,
        pdf_path: Path,
        output_directory: Path,
        *,
        dpi: int,
    ) -> Sequence[Path]:
        """Render all pages in source order and return their image paths."""
        ...


class PDFiumRenderer:
    """Render PDF pages to PNG files using pypdfium2."""

    async def render_pages(
        self,
        pdf_path: Path,
        output_directory: Path,
        *,
        dpi: int,
    ) -> Sequence[Path]:
        return await asyncio.to_thread(
            self._render_pages_sync,
            pdf_path,
            output_directory,
            dpi,
        )

    @staticmethod
    def _render_pages_sync(
        pdf_path: Path,
        output_directory: Path,
        dpi: int,
    ) -> list[Path]:
        try:
            import pypdfium2 as pdfium  # type: ignore[import-untyped]
        except ImportError as exc:  # pragma: no cover - required dependency
            raise PDFRenderError(
                "Scanned PDF OCR requires the 'pypdfium2' package"
            ) from exc

        output_directory.mkdir(parents=True, exist_ok=True)
        rendered_paths: list[Path] = []
        document = None
        try:
            document = pdfium.PdfDocument(pdf_path)
            scale = dpi / 72
            for page_index in range(len(document)):
                page = document[page_index]
                try:
                    bitmap = page.render(scale=scale)
                    try:
                        image = bitmap.to_pil()
                        page_path = output_directory / f"page-{page_index + 1:06d}.png"
                        image.save(page_path, format="PNG")
                        rendered_paths.append(page_path)
                    finally:
                        bitmap.close()
                finally:
                    page.close()
        except Exception as exc:
            if isinstance(exc, PDFRenderError):
                raise
            raise PDFRenderError(f"Cannot render PDF {pdf_path}: {exc}") from exc
        finally:
            if document is not None:
                document.close()

        return rendered_paths
