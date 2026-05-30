"""Tests for OCR provider contracts and the Tesseract adapter."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from app.adapters.ocr import (
    OCRProvider,
    OCRProviderResponseError,
    OCRResult,
    OCRTimeoutError,
    OCRUnreadableImageError,
    OCRUnsupportedMimeTypeError,
    TesseractOCRProvider,
)


def _completed_process(
    args: list[str],
    *,
    stdout: str = "",
    stderr: str = "",
    returncode: int = 0,
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=args,
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


def test_tesseract_provider_satisfies_protocol() -> None:
    assert isinstance(TesseractOCRProvider(), OCRProvider)


async def test_tesseract_provider_rejects_unsupported_mime_type(tmp_path: Path) -> None:
    file_path = tmp_path / "document.txt"
    file_path.write_text("not an image", encoding="utf-8")

    with pytest.raises(OCRUnsupportedMimeTypeError):
        await TesseractOCRProvider().extract_text(file_path, mime_type="text/plain")


async def test_tesseract_provider_maps_cli_output_to_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "image.png"
    file_path.write_bytes(b"fake png")

    calls: list[tuple[list[str], int | None]] = []

    def fake_run(
        args: list[str],
        *,
        capture_output: bool,
        text: bool,
        check: bool,
        timeout: int | None,
    ) -> subprocess.CompletedProcess[str]:
        del capture_output, text, check
        calls.append((args, timeout))
        if "--version" in args:
            return _completed_process(args, stdout="tesseract 5.4.0\n")
        if args[-1] == "tsv":
            return _completed_process(
                args,
                stdout=(
                    "level\tpage_num\tconf\ttext\n"
                    "5\t1\t90\tHello\n"
                    "5\t1\t80\tworld\n"
                ),
            )
        return _completed_process(args, stdout="Hello world\n")

    monkeypatch.setattr(subprocess, "run", fake_run)

    provider = TesseractOCRProvider(
        command="/usr/local/bin/tesseract",
        default_languages=["eng"],
        default_timeout_seconds=33,
        psm=4,
        oem=1,
    )

    result = await provider.extract_text(
        file_path,
        mime_type="image/png",
        languages=["eng", "deu"],
        timeout_seconds=17,
    )

    assert isinstance(result, OCRResult)
    assert result.provider == "tesseract"
    assert result.engine_version == "tesseract 5.4.0"
    assert result.languages == ["eng", "deu"]
    assert len(result.pages) == 1
    assert result.pages[0].text == "Hello world"
    assert result.pages[0].confidence == pytest.approx(85.0)

    commands = [args for args, _ in calls]
    assert any(command[:2] == ["/usr/local/bin/tesseract", "--version"] for command in commands)
    assert any(
        command[:3] == ["/usr/local/bin/tesseract", str(file_path), "stdout"]
        and "-l" in command
        and "eng+deu" in command
        and "--psm" in command
        and "--oem" in command
        for command in commands
    )
    assert all(timeout == 17 or timeout == 33 for _, timeout in calls)


async def test_tesseract_provider_maps_corrupt_image_to_domain_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "corrupt.png"
    file_path.write_bytes(b"not really an image")

    def fake_run(
        args: list[str],
        *,
        capture_output: bool,
        text: bool,
        check: bool,
        timeout: int | None,
    ) -> subprocess.CompletedProcess[str]:
        del capture_output, text, check, timeout
        if "--version" in args:
            return _completed_process(args, stdout="tesseract 5.4.0\n")
        return _completed_process(
            args,
            stderr="Error in pixRead: image file is truncated",
            returncode=1,
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(OCRUnreadableImageError, match="pixRead"):
        await TesseractOCRProvider().extract_text(file_path, mime_type="image/png")


async def test_tesseract_provider_applies_timeout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "slow.png"
    file_path.write_bytes(b"fake png")

    def fake_run(
        args: list[str],
        *,
        capture_output: bool,
        text: bool,
        check: bool,
        timeout: int | None,
    ) -> subprocess.CompletedProcess[str]:
        del args, capture_output, text, check, timeout
        raise subprocess.TimeoutExpired(cmd="tesseract", timeout=9)

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(OCRTimeoutError, match="9 seconds"):
        await TesseractOCRProvider(default_timeout_seconds=9).extract_text(
            file_path,
            mime_type="image/png",
        )


async def test_tesseract_provider_rejects_malformed_tsv_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "image.png"
    file_path.write_bytes(b"fake png")

    def fake_run(
        args: list[str],
        *,
        capture_output: bool,
        text: bool,
        check: bool,
        timeout: int | None,
    ) -> subprocess.CompletedProcess[str]:
        del capture_output, text, check, timeout
        if "--version" in args:
            return _completed_process(args, stdout="tesseract 5.4.0\n")
        if args[-1] == "tsv":
            return _completed_process(args, stdout="not\treal\ttsv\n")
        return _completed_process(args, stdout="Recognized text\n")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(OCRProviderResponseError, match="malformed TSV"):
        await TesseractOCRProvider().extract_text(file_path, mime_type="image/png")
