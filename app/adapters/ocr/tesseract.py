"""Local Tesseract OCR provider."""

from __future__ import annotations

import asyncio
import csv
import io
import subprocess
from pathlib import Path

from app.adapters.ocr.base import (
    OCRPageResult,
    OCRProviderUnavailableError,
    OCRProviderResponseError,
    OCRResult,
    OCRTimeoutError,
    OCRUnreadableImageError,
    OCRUnsupportedMimeTypeError,
    SUPPORTED_IMAGE_MIME_TYPES,
    mime_type_requires_ocr,
    normalize_mime_type,
)


class TesseractOCRProvider:
    """OCR provider backed by the local ``tesseract`` CLI."""

    name = "tesseract"

    def __init__(
        self,
        *,
        command: str = "tesseract",
        default_languages: list[str] | tuple[str, ...] | None = None,
        default_timeout_seconds: int | None = 120,
        psm: int | None = 6,
        oem: int | None = 3,
    ) -> None:
        self.command = command
        self.default_languages = list(default_languages or ["eng"])
        self.default_timeout_seconds = default_timeout_seconds
        self.psm = psm
        self.oem = oem

    async def extract_text(
        self,
        file_path: Path,
        *,
        mime_type: str,
        languages: list[str] | None = None,
        timeout_seconds: int | None = None,
    ) -> OCRResult:
        normalized_mime = normalize_mime_type(mime_type)
        if normalized_mime not in SUPPORTED_IMAGE_MIME_TYPES:
            raise OCRUnsupportedMimeTypeError(
                f"Unsupported OCR MIME type: {normalized_mime or '<unknown>'}"
            )
        if not file_path.exists():
            raise OCRUnreadableImageError(f"Image file not found: {file_path}")

        selected_languages = list(languages or self.default_languages or ["eng"])
        resolved_timeout = timeout_seconds or self.default_timeout_seconds

        text_process, tsv_process, engine_version = await asyncio.gather(
            self._run_tesseract(
                file_path,
                languages=selected_languages,
                output_format="txt",
                timeout_seconds=resolved_timeout,
            ),
            self._run_tesseract(
                file_path,
                languages=selected_languages,
                output_format="tsv",
                timeout_seconds=resolved_timeout,
            ),
            self._get_engine_version(),
        )

        page_texts = self._split_pages(text_process.stdout)
        page_confidences = self._parse_tsv_confidences(tsv_process.stdout)

        pages = [
            OCRPageResult(
                page_number=page_number,
                text=text,
                confidence=page_confidences.get(page_number),
                metadata={
                    "mime_type": normalized_mime,
                    "source_path": str(file_path),
                },
            )
            for page_number, text in enumerate(page_texts, start=1)
        ]

        warnings: list[str] = []
        for message in (text_process.stderr.strip(), tsv_process.stderr.strip()):
            if message and message not in warnings:
                warnings.append(message)

        return OCRResult(
            provider=self.name,
            engine_version=engine_version,
            languages=selected_languages,
            pages=pages,
            warnings=warnings,
            metadata={"mime_type": normalized_mime},
        )

    async def healthcheck(self) -> bool:
        try:
            await self._get_engine_version()
        except OCRProviderUnavailableError:
            return False
        return True

    async def _get_engine_version(self) -> str:
        process = await self._run_command(
            [self.command, "--version"],
            timeout_seconds=self.default_timeout_seconds,
        )
        first_line = process.stdout.splitlines()[0].strip() if process.stdout else ""
        if not first_line:
            raise OCRProviderResponseError("Tesseract did not return a version string")
        return first_line

    async def _run_tesseract(
        self,
        file_path: Path,
        *,
        languages: list[str],
        output_format: str,
        timeout_seconds: int | None,
    ) -> subprocess.CompletedProcess[str]:
        command = [
            self.command,
            str(file_path),
            "stdout",
            "-l",
            "+".join(languages),
        ]
        if self.oem is not None:
            command.extend(["--oem", str(self.oem)])
        if self.psm is not None:
            command.extend(["--psm", str(self.psm)])
        if output_format == "tsv":
            command.append("tsv")

        process = await self._run_command(command, timeout_seconds=timeout_seconds)
        if process.returncode == 0:
            return process

        self._raise_process_error(process.stderr or process.stdout or "Tesseract OCR failed")
        raise AssertionError("unreachable")

    async def _run_command(
        self, command: list[str], *, timeout_seconds: int | None
    ) -> subprocess.CompletedProcess[str]:
        try:
            return await asyncio.to_thread(
                subprocess.run,
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout_seconds,
            )
        except FileNotFoundError as exc:
            raise OCRProviderUnavailableError(
                f"Tesseract command is not available: {self.command}"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            resolved_timeout = timeout_seconds or 0
            raise OCRTimeoutError(
                f"Tesseract OCR timed out after {resolved_timeout} seconds"
            ) from exc

    def _raise_process_error(self, message: str) -> None:
        normalized = message.strip() or "Tesseract OCR failed"
        lower_message = normalized.lower()
        if "failed loading language" in lower_message or "error opening data file" in lower_message:
            raise OCRProviderUnavailableError(normalized)
        if any(
            fragment in lower_message
            for fragment in (
                "cannot read",
                "error in pixread",
                "image file is truncated",
                "unsupported image",
                "unable to read",
                "read of file failed",
                "image too large",
            )
        ):
            raise OCRUnreadableImageError(normalized)
        raise OCRProviderResponseError(normalized)

    def _split_pages(self, text_output: str) -> list[str]:
        raw_pages = [page.strip() for page in text_output.split("\f")]
        pages = [page for page in raw_pages if page]
        return pages or [text_output.strip()]

    def _parse_tsv_confidences(self, tsv_output: str) -> dict[int, float | None]:
        if not tsv_output.strip():
            return {}

        reader = csv.DictReader(io.StringIO(tsv_output), delimiter="\t")
        fieldnames = set(reader.fieldnames or [])
        if not {"page_num", "conf"}.issubset(fieldnames):
            raise OCRProviderResponseError("Tesseract returned malformed TSV output")

        totals: dict[int, tuple[float, int]] = {}
        for row in reader:
            raw_page = (row.get("page_num") or "").strip()
            raw_conf = (row.get("conf") or "").strip()
            if not raw_page or not raw_conf:
                continue
            try:
                page_number = int(raw_page)
                confidence = float(raw_conf)
            except ValueError as exc:
                raise OCRProviderResponseError(
                    "Tesseract returned invalid TSV confidence data"
                ) from exc
            if confidence < 0:
                continue
            total, count = totals.get(page_number, (0.0, 0))
            totals[page_number] = (total + confidence, count + 1)

        return {
            page_number: (total / count if count else None)
            for page_number, (total, count) in totals.items()
        }


def _assert_supported_mimes() -> set[str]:
    return set(SUPPORTED_IMAGE_MIME_TYPES)


def _supports_ocr_mime(mime_type: str | None) -> bool:
    return mime_type_requires_ocr(mime_type)
