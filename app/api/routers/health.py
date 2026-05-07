from fastapi import APIRouter, HTTPException
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from app.core.config import get_settings
from app.db.session import get_engine

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/live")
async def live() -> dict[str, str]:
    """Liveness probe – returns 200 if the API process is running."""
    settings = get_settings()
    return {"status": "ok", "service": "pda-api", "version": settings.app_version}


@router.get("/ready")
async def ready() -> dict[str, object]:
    """Readiness probe – returns 200 only when required dependencies are available."""
    engine = get_engine()
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        db_status = "ok"
    except SQLAlchemyError:
        raise HTTPException(
            status_code=503,
            detail={"status": "not ready", "dependencies": {"database": "unavailable"}},
        )

    return {"status": "ready", "dependencies": {"database": db_status}}
