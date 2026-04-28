from fastapi import APIRouter

router = APIRouter()


@router.get("/", tags=["root"])
async def root() -> dict[str, str]:
    return {"name": "PDA API", "status": "ok"}
