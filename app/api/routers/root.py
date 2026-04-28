from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/", tags=["root"])
async def root(request: Request) -> dict[str, str]:
    return {"name": request.app.title, "status": "ok"}
