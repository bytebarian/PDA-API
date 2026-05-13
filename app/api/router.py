from fastapi import APIRouter

from app.api.routers.documents import router as documents_router
from app.api.routers.health import router as health_router
from app.api.routers.root import router as root_router

api_router = APIRouter()
api_router.include_router(root_router)
api_router.include_router(health_router)
api_router.include_router(documents_router)
