from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.router import api_router
from app.api.routers.search import close_search_providers
from app.core.config import validate_settings


@asynccontextmanager
async def _app_lifespan(_: FastAPI) -> AsyncIterator[None]:
    try:
        yield
    finally:
        await close_search_providers()


def create_app() -> FastAPI:
    settings = validate_settings()
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        lifespan=_app_lifespan,
    )
    app.include_router(api_router, prefix=settings.api_prefix)
    return app


app = create_app()
