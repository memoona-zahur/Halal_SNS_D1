"""Response bodies of the Check door, and the one error shape every refusal uses."""

from typing import Literal

from pydantic import BaseModel, Field

JobStatusName = Literal["queued", "running", "done", "failed", "dead"]
LabelName = Literal["allow", "hold", "block"]
FINISHED = frozenset({"done", "failed", "dead"})


class AcceptedResponse(BaseModel):
    """The check is still running. Read the decision with GET /v1/moderations/{job_id}, or from the callback."""

    job_id: str = Field(description="Id to poll, or to match the callback")
    status: JobStatusName = Field(description="'queued' for a new job, the current state for a repeated content version")


class JobStatusResponse(BaseModel):
    """Where a job is. label, category and reason are set once status is 'done'."""

    job_id: str = Field(description="Job id")
    status: JobStatusName = Field(
        description="queued, running, done (decided), failed (could not be checked) or dead (gave up after retries)"
    )
    label: LabelName | None = Field(default=None, description="allow: publish. hold: keep hidden. block: do not publish")
    category: str | None = Field(default=None, description="Harm category of the worst item, e.g. G-03. Null when allowed")
    reason: str | None = Field(default=None, description="One line naming the item and category")
    error_code: str | None = Field(default=None, description="Why a failed or dead job could not be checked")
    latency_ms: int | None = Field(default=None, description="Accepted to decided, queue wait included. Null until decided")


class UploadResponse(BaseModel):
    """A raw image was stored. Put `url` in `images[].url` on the next POST /v1/moderations."""

    media_id: str = Field(description="A suggested media_id for this image; you may use your own instead")
    url: str = Field(description="Put this in images[].url when you send the post")


class ErrorResponse(BaseModel):
    """Every refusal the door makes itself. A malformed body is FastAPI's own 422 with a 'detail' list."""

    error: str = Field(description="unknown_surface, limit_exceeded, not_found or unavailable")
    detail: str = Field(description="What was wrong")


class Rejected(Exception):
    """Raise it anywhere in a route. One handler in app.py turns it into an ErrorResponse."""

    def __init__(self, status, error, detail):
        super().__init__(detail)
        self.status = status
        self.error = error
        self.detail = detail
