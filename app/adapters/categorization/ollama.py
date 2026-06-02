"""Ollama-backed categorization provider.

Uses the local Ollama `/api/generate` endpoint to classify documents into the
allowed category vocabulary.  The model is prompted to return constrained JSON
only – raw model output is never persisted without validation.

Privacy
-------
Document text is sent **only** to a local Ollama instance (never to public
cloud APIs).  Raw text is not logged; only document id, model, category, and
confidence appear in logs.
"""

from __future__ import annotations

import asyncio
import json
import logging

import httpx

from app.adapters.categorization.base import (
    ALLOWED_CATEGORIES,
    CategorizationProviderResponseError,
    CategorizationProviderUnavailableError,
    CategorizationResult,
    CategorizationSource,
    DocumentCategory,
)

logger = logging.getLogger(__name__)

_CATEGORIZATION_PROMPT_TEMPLATE = """\
You classify a private personal document for a local document assistant.
Use only the provided text, summary, filename, and metadata.
Choose exactly one category from this allowed list: {allowed_categories}.
Return valid JSON only:
{{
  "category": "contract|agreement|offer|invoice|official_document|documentation|note|other",
  "confidence": 0.0,
  "reason": "short factual reason"
}}
Do not invent document facts.
If unsure, use "other".

Filename: {filename}
Summary: {summary}
Document text:
{text}"""


class OllamaCategorizationProvider:
    """Classify documents using a local Ollama model with constrained JSON output."""

    name: str = CategorizationSource.local_model.value

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:11434",
        timeout_seconds: int = 120,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._transport = transport
        self._client: httpx.AsyncClient | None = None
        self._client_lock: asyncio.Lock | None = None

    def _get_client_lock(self) -> asyncio.Lock:
        if self._client_lock is None:
            self._client_lock = asyncio.Lock()
        return self._client_lock

    async def _get_client(self) -> httpx.AsyncClient:
        async with self._get_client_lock():
            if self._client is None or self._client.is_closed:
                kwargs: dict = {
                    "base_url": self._base_url,
                    "timeout": float(self._timeout_seconds),
                }
                if self._transport is not None:
                    kwargs["transport"] = self._transport
                self._client = httpx.AsyncClient(**kwargs)
        return self._client

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()

    async def categorize(
        self,
        text: str,
        *,
        filename: str | None = None,
        summary: str | None = None,
        metadata: dict | None = None,
        model: str = "llama3.2:3b",
    ) -> CategorizationResult:
        """Classify the document text using a local Ollama model."""
        allowed_str = ", ".join(sorted(ALLOWED_CATEGORIES))
        prompt = _CATEGORIZATION_PROMPT_TEMPLATE.format(
            allowed_categories=allowed_str,
            filename=filename or "unknown",
            summary=summary or "not available",
            text=text,
        )
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "num_predict": 128,
                "temperature": 0.0,
            },
        }

        client = await self._get_client()
        try:
            response = await client.post("/api/generate", json=payload)
        except httpx.TimeoutException as exc:
            raise CategorizationProviderUnavailableError(
                f"Ollama categorization timed out after {self._timeout_seconds}s"
            ) from exc
        except httpx.ConnectError as exc:
            raise CategorizationProviderUnavailableError(
                f"Ollama is not reachable at {self._base_url}"
            ) from exc
        except httpx.HTTPError as exc:
            raise CategorizationProviderUnavailableError(
                f"Ollama HTTP error during categorization: {exc}"
            ) from exc

        if response.status_code != 200:
            raise CategorizationProviderResponseError(
                f"Ollama returned HTTP {response.status_code} during categorization"
            )

        try:
            data = response.json()
        except Exception as exc:
            raise CategorizationProviderResponseError(
                "Ollama returned a non-JSON response envelope during categorization"
            ) from exc

        raw_response = data.get("response", "").strip()
        if not raw_response:
            raise CategorizationProviderResponseError(
                "Ollama returned an empty categorization response"
            )

        try:
            parsed = json.loads(raw_response)
        except json.JSONDecodeError as exc:
            raise CategorizationProviderResponseError(
                "Ollama categorization response is not valid JSON"
            ) from exc

        category = str(parsed.get("category", "")).strip().lower()
        if category not in ALLOWED_CATEGORIES:
            logger.warning(
                "ollama_categorize: model returned unknown category %r; falling back to other",
                category,
            )
            category = DocumentCategory.other.value

        raw_confidence = parsed.get("confidence", 0.0)
        try:
            confidence = float(raw_confidence)
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))

        reason = str(parsed.get("reason", "")).strip()[:500] or "no_reason_provided"

        return CategorizationResult(
            category=category,
            confidence=confidence,
            reason=reason,
            source=CategorizationSource.local_model.value,
            model=data.get("model", model),
        )

    async def healthcheck(self) -> bool:
        """Return True when Ollama responds to a GET /api/tags request."""
        client = await self._get_client()
        try:
            response = await client.get("/api/tags")
            return response.status_code == 200
        except httpx.HTTPError:
            return False
