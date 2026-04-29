from functools import lru_cache
from pathlib import Path

from pydantic import ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "PDA API"
    app_version: str = "0.1.0"
    api_prefix: str = "/"

    database_url: str = "sqlite+aiosqlite:///./pda.db"
    storage_path: Path = Path("./storage")

    allowed_file_types: tuple[str, ...] = (
        "application/pdf",
        "text/plain",
        "image/png",
        "image/jpeg",
    )
    max_file_size_bytes: int = 10 * 1024 * 1024

    model_provider: str = "local"
    model_name: str = "llama3.1:8b-instruct"

    ocr_provider: str = "tesseract"
    ocr_language: str = "eng"
    ocr_dpi: int = 300

    @field_validator("allowed_file_types", mode="before")
    @classmethod
    def normalize_allowed_file_types(cls, value: object) -> tuple[str, ...]:
        if isinstance(value, str):
            parsed = tuple(item.strip() for item in value.split(",") if item.strip())
        elif isinstance(value, (list, tuple, set)):
            parsed = tuple(str(item).strip() for item in value if str(item).strip())
        else:
            raise ValueError("allowed_file_types must be a list or comma-separated string")

        if not parsed:
            raise ValueError("allowed_file_types must include at least one MIME type")

        return parsed

    @field_validator("max_file_size_bytes", "ocr_dpi")
    @classmethod
    def must_be_positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("must be greater than 0")
        return value

    model_config = SettingsConfigDict(
        env_prefix="PDA_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def _format_validation_error(exc: ValidationError) -> str:
    details: list[str] = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error.get("loc", ()))
        message = error.get("msg", "invalid value")
        if location:
            details.append(f"{location}: {message}")
        else:
            details.append(str(message))
    return "; ".join(details) if details else str(exc)


def validate_settings() -> Settings:
    try:
        return get_settings()
    except ValidationError as exc:
        message = _format_validation_error(exc)
        raise RuntimeError(f"Invalid PDA configuration: {message}") from exc
