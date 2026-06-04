"""HTTP endpoints for grounded document chat."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.db.session import get_db
from app.schemas.chat import ChatAskRequest, ChatAskResponse
from app.services.chat_service import (
    ChatConfigurationError,
    ChatProviderNotAvailableError,
    ChatService,
    ChatServiceError,
)

router = APIRouter(prefix="/chat", tags=["chat"])
logger = logging.getLogger(__name__)


def get_chat_service(
    db: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> ChatService:
    return ChatService(db, settings=settings)


@router.post(
    "/ask",
    response_model=ChatAskResponse,
    summary="Ask a question grounded in indexed document chunks",
    description=(
        "Retrieve relevant document chunks, assemble prompt context, call the "
        "configured local chat model, and return the answer with source citations."
    ),
)
async def ask_question(
    request: ChatAskRequest,
    service: ChatService = Depends(get_chat_service),
) -> ChatAskResponse:
    """Answer a one-shot question using retrieved document context."""
    try:
        return await service.ask_question(request)
    except ChatProviderNotAvailableError as exc:
        logger.warning("Chat provider unavailable: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chat model provider is currently unavailable. Please try again later.",
        ) from exc
    except ChatConfigurationError as exc:
        logger.error("Chat configuration error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chat is not configured on this server.",
        ) from exc
    except ChatServiceError as exc:
        logger.error("Chat service error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred during chat generation.",
        ) from exc
