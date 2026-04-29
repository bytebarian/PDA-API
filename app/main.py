from fastapi import FastAPI

from app.api.router import api_router
from app.core.config import validate_settings


def create_app() -> FastAPI:
    settings = validate_settings()
    app = FastAPI(title=settings.app_name, version=settings.app_version)
    app.include_router(api_router, prefix=settings.api_prefix)
    return app


app = create_app()
