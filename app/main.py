from fastapi import FastAPI

from app.api.router import api_router
from app.core.config import validate_settings
from app.core.logging import configure_logging
from app.core.middleware import request_logging_middleware


def create_app() -> FastAPI:
    configure_logging()
    settings = validate_settings()
    app = FastAPI(title=settings.app_name, version=settings.app_version)
    app.middleware("http")(request_logging_middleware)
    app.include_router(api_router, prefix=settings.api_prefix)
    return app


app = create_app()
