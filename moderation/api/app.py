"""Builds the FastAPI app: settings, policy, store, error handlers and the two routers."""

from contextlib import asynccontextmanager

import psycopg
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from moderation.api.routes import health, moderations
from moderation.api.schemas import Rejected
from moderation.config import get_settings
from moderation.policy import D1_POLICY
from moderation.store import Store

DESCRIPTION = """
The backend sends a post (caption and image URLs). The checker answers **202** at once, checks the caption and
every image in the background, and sends the decision by callback.

| Label | What the backend does |
|---|---|
| `allow` | publish |
| `hold` | keep hidden |
| `block` | do not publish |

**Flow:** `POST /v1/moderations` waits and answers `200` with the decision. If the check takes longer than `API_WAIT_S`
it answers `202 {job_id}`: read the result with `GET /v1/moderations/{job_id}` or from the optional callback
(`moderation.decided` or `moderation.failed`).
"""

TAGS = [
    {"name": "Moderation", "description": "Submit a post and read the decision."},
    {"name": "Health", "description": "For the orchestrator: is the process up, can it reach the database."},
]


def create_app(*, settings=None, store=None):
    cfg = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app):
        if store is not None:
            yield
            return
        app.state.store = Store(cfg)
        await app.state.store.open()
        yield
        await app.state.store.close()

    app = FastAPI(
        title="Halal SNS moderation: Check door",
        version="0.2.0",
        description=DESCRIPTION,
        openapi_tags=TAGS,
        lifespan=lifespan,
    )
    app.state.cfg = cfg
    app.state.policy = D1_POLICY
    if store is not None:
        app.state.store = store  # set here: test clients do not run the lifespan

    app.include_router(moderations)
    app.include_router(health)

    @app.middleware("http")
    async def limit_body(request: Request, call_next):
        # The upload endpoint carries raw image bytes, not JSON, so it gets its own (larger) cap.
        is_upload = request.url.path == "/v1/moderations/uploads"
        cap = cfg.image_max_bytes + 4096 if is_upload else cfg.api_max_body_bytes
        declared = request.headers.get("content-length", "")
        if declared.isdigit() and int(declared) > cap:
            return JSONResponse(
                status_code=413,
                content={"error": "body_too_large", "detail": f"body is larger than {cap} bytes"},
            )
        return await call_next(request)

    @app.exception_handler(Rejected)
    async def handle_rejected(_: Request, exc: Rejected):
        return JSONResponse(status_code=exc.status, content={"error": exc.error, "detail": exc.detail})

    @app.exception_handler(psycopg.Error)
    async def handle_db_error(_: Request, exc: psycopg.Error):
        # The backend retries on 503; the post stays "processing".
        return JSONResponse(status_code=503, content={"error": "unavailable", "detail": type(exc).__name__})

    return app
