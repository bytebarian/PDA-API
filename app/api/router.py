from fastapi import APIRouter

from app.api.routers.chat import router as chat_router
from app.api.routers.citations import router as citations_router
from app.api.routers.documents import router as documents_router
from app.api.routers.health import router as health_router
from app.api.routers.jobs import router as jobs_router
from app.api.routers.reports import router as reports_router
from app.api.routers.root import router as root_router
from app.api.routers.search import router as search_router

api_router = APIRouter()
api_router.include_router(root_router)
api_router.include_router(health_router)
api_router.include_router(chat_router)
api_router.include_router(citations_router)
api_router.include_router(documents_router)
api_router.include_router(jobs_router)
api_router.include_router(reports_router)
api_router.include_router(search_router)
