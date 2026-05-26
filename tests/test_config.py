import pytest
from pydantic import ValidationError

from app.core.config import Settings


def test_settings_defaults() -> None:
    settings = Settings()
    assert settings.database_url
    assert settings.storage_path.as_posix().endswith("storage")
    assert "application/pdf" in settings.allowed_file_types
    assert settings.max_file_size_bytes > 0
    assert settings.model_provider == "local"
    assert settings.ocr_provider == "tesseract"
    assert settings.embedding_provider == "ollama"
    assert settings.embedding_model == "all-minilm"
    assert settings.embedding_dimensions == 1536
    assert settings.embedding_batch_size == 16
    assert settings.embedding_timeout_seconds == 60
    assert settings.embedding_truncate is True
    assert settings.ollama_base_url == "http://localhost:11434"


def test_settings_reads_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PDA_ALLOWED_FILE_TYPES", "[\"application/pdf\", \"text/markdown\"]")
    monkeypatch.setenv("PDA_MAX_FILE_SIZE_BYTES", "512")
    monkeypatch.setenv("PDA_MODEL_NAME", "custom-local-model")
    monkeypatch.setenv("PDA_EMBEDDING_PROVIDER", "fake")

    settings = Settings()

    assert settings.allowed_file_types == ("application/pdf", "text/markdown")
    assert settings.max_file_size_bytes == 512
    assert settings.model_name == "custom-local-model"
    assert settings.embedding_provider == "fake"


def test_invalid_settings_fail_clearly() -> None:
    with pytest.raises(ValidationError, match="max_file_size_bytes"):
        Settings(max_file_size_bytes=0)


def test_embedding_dimensions_must_match_chunk_vector_dimensions() -> None:
    from app.models.document_chunk import EMBEDDING_DIMENSIONS

    with pytest.raises(ValidationError, match="embedding_dimensions"):
        Settings(embedding_dimensions=EMBEDDING_DIMENSIONS + 1)
