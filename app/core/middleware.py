import json
import logging
import time
from datetime import UTC, datetime
from uuid import uuid4

from fastapi import Request

from app.core.request_context import request_id_ctx

request_logger = logging.getLogger("pda.request")


async def request_logging_middleware(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or request.headers.get("x-correlation-id") or str(uuid4())
    token = request_id_ctx.set(request_id)
    start = time.perf_counter()

    try:
        response = await call_next(request)

        duration_ms = round((time.perf_counter() - start) * 1000, 3)
        response.headers["x-request-id"] = request_id

        request_logger.info(
            json.dumps(
                {
                    "timestamp": datetime.now(UTC).isoformat(),
                    "level": "INFO",
                    "logger": "pda.request",
                    "message": "request.completed",
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status": response.status_code,
                    "duration_ms": duration_ms,
                }
            )
        )

        return response
    finally:
        request_id_ctx.reset(token)
