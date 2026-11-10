"""The endpoints. Each does one thing; limits live in the helpers below them."""

import asyncio
import secrets
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, Header, Request, UploadFile
from fastapi.responses import JSONResponse

from moderation.api.schemas import (
    FINISHED,
    AcceptedResponse,
    ErrorResponse,
    JobStatusResponse,
    Rejected,
    UploadResponse,
)
from moderation.config import Settings
from moderation.domain import ModerationRequest
from moderation.errors import PermanentError
from moderation.fetch import _check_type, guess_extension
from moderation.policy import Policy
from moderation.store import Store


# What a route needs comes from app.state, which create_app fills.
def get_store(request: Request):
    return request.app.state.store


def get_cfg(request: Request):
    return request.app.state.cfg


def get_policy(request: Request):
    return request.app.state.policy


StoreDep = Annotated[Store, Depends(get_store)]
CfgDep = Annotated[Settings, Depends(get_cfg)]
PolicyDep = Annotated[Policy, Depends(get_policy)]


async def require_api_key(cfg: CfgDep, x_api_key: Annotated[str | None, Header()] = None) -> None:
    """Checked against Settings.api_key_names (X-API-Key: name:key pairs in API_KEYS). An empty
    API_KEYS disables this - local/CPU dev only, never for a tunnel reachable from the internet."""
    keys = cfg.api_key_names
    if not keys:
        return
    if x_api_key is None or x_api_key not in keys:
        raise Rejected(401, "unauthorized", "missing or invalid X-API-Key")


moderations = APIRouter(prefix="/v1/moderations", tags=["Moderation"], dependencies=[Depends(require_api_key)])
health = APIRouter(tags=["Health"])

ERRORS = {
    400: {"model": ErrorResponse, "description": "unknown_surface: D1 accepts 'post' only"},
    401: {"model": ErrorResponse, "description": "unauthorized: missing or invalid X-API-Key"},
    404: {"model": ErrorResponse, "description": "not_found: no job with this id"},
    422: {"model": ErrorResponse, "description": "limit_exceeded, or a malformed body (FastAPI's own 'detail' list)"},
    503: {"model": ErrorResponse, "description": "unavailable: the database is down, retry"},
}


@moderations.post(
    "",
    status_code=200,
    response_model=None,
    summary="Submit a post and get the decision",
    responses={
        200: {"model": JobStatusResponse, "description": "Checked. The body holds the decision (or the failure)"},
        202: {"model": AcceptedResponse, "description": "Saved but not finished within the wait limit. Poll GET, or wait for the callback"},
        400: ERRORS[400],
        401: ERRORS[401],
        422: ERRORS[422],
        503: ERRORS[503],
    },
)
async def submit_moderation(
    body: ModerationRequest,
    store: StoreDep,
    cfg: CfgDep,
    policy: PolicyDep,
):
    """Save the post as a job, check it, and answer with the decision. The caption and every image are checked and
    the worst result decides the label.

    **200** carries the decision when the check finishes within the wait limit (`API_WAIT_S`, 10 s by default).
    **202** means the check is still running: read the result later with `GET /v1/moderations/{job_id}`, or from
    the callback if you sent `callback_url`. Publish only on `allow`.

    Sending the same `surface` + `content_id` + `content_version` again returns the same job and never re-runs it.
    An edit must send a higher `content_version`.
    """
    if not policy.accepts(body.surface):
        raise Rejected(400, "unknown_surface", body.surface)
    check_limits(body, cfg)
    job_id, status = await store.create_job(body)
    row = await wait_for_decision(store, cfg, job_id)
    if row is not None:
        return JSONResponse(status_code=200, content=job_response(row).model_dump())
    return JSONResponse(status_code=202, content=AcceptedResponse(job_id=str(job_id), status=status).model_dump())


@moderations.post(
    "/uploads",
    status_code=201,
    summary="Upload a raw image instead of hosting it yourself",
    responses={
        401: ERRORS[401],
        422: ERRORS[422],
        503: {"model": ErrorResponse, "description": "unavailable: raw uploads are disabled on this deployment"},
    },
)
async def upload_image(
    cfg: CfgDep,
    file: Annotated[UploadFile, File(description="The raw image bytes")],
) -> UploadResponse:
    """Stores the file and returns a `url` to put in `images[].url` on the next `POST /v1/moderations` - an
    alternative to hosting the image yourself and sending its link. Disabled unless the deployment sets
    `IMAGE_ALLOWED_ROOT`. Accepts JPEG, PNG, WebP and GIF, judged by file content, up to `IMAGE_MAX_BYTES`.
    """
    root = cfg.image_allowed_root.strip()
    if not root:
        raise Rejected(503, "unavailable", "raw image uploads are disabled on this deployment")
    data = await file.read(cfg.image_max_bytes + 1)
    if len(data) > cfg.image_max_bytes:
        raise Rejected(422, "limit_exceeded", f"image larger than {cfg.image_max_bytes} bytes")
    try:
        _check_type(data)
    except PermanentError as exc:
        raise Rejected(422, "limit_exceeded", str(exc)) from exc
    uploads_dir = Path(root) / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    media_id = secrets.token_hex(8)
    filename = media_id + guess_extension(data)
    (uploads_dir / filename).write_bytes(data)
    return UploadResponse(media_id=media_id, url=f"uploads/{filename}")


@moderations.get(
    "/{job_id}",
    response_model=JobStatusResponse,
    summary="Get the state of a job",
    responses={401: ERRORS[401], 404: ERRORS[404], 503: ERRORS[503]},
)
async def get_moderation(job_id: uuid.UUID, store: StoreDep):
    """Poll a job. Stop when `status` is `done`, `failed` or `dead`. A malformed id is a 422."""
    row = await store.get_job_status(job_id)
    if row is None:
        raise Rejected(404, "not_found", "job not found")
    return job_response(row)


@health.get("/healthz", summary="Liveness: the process is up")
async def healthz():
    return {"status": "ok"}


@health.get("/readyz", summary="Readiness: the database answers", responses={503: ERRORS[503]})
async def readyz(store: StoreDep):
    await store.ping()
    return {"status": "ready"}


def check_limits(body, cfg):
    if len(body.images) > cfg.api_max_images:
        raise Rejected(422, "limit_exceeded", f"too many images: {len(body.images)} > {cfg.api_max_images}")
    if body.text is not None and len(body.text) > cfg.api_max_text_chars:
        raise Rejected(422, "limit_exceeded", f"text longer than {cfg.api_max_text_chars} characters")
    if body.callback_url is not None and len(str(body.callback_url)) > cfg.api_max_callback_url_chars:
        raise Rejected(422, "limit_exceeded", f"callback URL longer than {cfg.api_max_callback_url_chars} characters")
    if any(len(image.url) > cfg.api_max_url_chars for image in body.images):
        raise Rejected(422, "limit_exceeded", f"image URL longer than {cfg.api_max_url_chars} characters")


async def wait_for_decision(store, cfg, job_id):
    """Check the job until it is finished or API_WAIT_S runs out. None: still not finished."""
    if cfg.api_wait_s <= 0:
        return None
    loop = asyncio.get_running_loop()
    deadline = loop.time() + cfg.api_wait_s
    while True:
        row = await store.get_job_status(job_id)
        if row is not None and row.status in FINISHED:
            return row
        remaining = deadline - loop.time()
        if remaining <= 0:
            return None
        await asyncio.sleep(min(cfg.api_wait_poll_s, remaining))


def job_response(row):
    return JobStatusResponse(
        job_id=str(row.job_id),
        status=row.status,
        label=row.label,
        category=row.category,
        reason=row.reason,
        error_code=row.error_code,
        latency_ms=row.latency_ms,
    )

