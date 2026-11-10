"""The two ways a job can go wrong.

PermanentError  retrying cannot help (broken image, forbidden URL). The job fails at once.
RetryableError  it may work later (image not there yet, timeout). The job retries with backoff.
Anything else is a bug or an outage: the job retries, then fails closed.

`translate` turns a library exception into one of the two, where a library raises its own.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

# error_code values the backend can receive in a failure callback.
MEDIA_NOT_FOUND = "media_not_found"
MEDIA_TOO_LARGE = "media_too_large"
MEDIA_UNSUPPORTED = "media_unsupported"
MEDIA_FORBIDDEN = "media_forbidden"
MEDIA_UNDECODABLE = "media_undecodable"
MODEL_OOM = "model_oom"
ATTEMPTS_EXHAUSTED = "attempts_exhausted"
INTERNAL_ERROR = "internal_error"


class JobError(Exception):
    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code


class PermanentError(JobError):
    """Retrying cannot help. The job fails now with `code`."""


class RetryableError(JobError):
    """May succeed later. The job retries; `code` is reported if the retries run out."""


@contextmanager
def translate(kind: type[JobError], code: str, *catch: type[BaseException]) -> Iterator[None]:
    """Inside the block, an exception of a type in `catch` leaves as kind(code)."""
    try:
        yield
    except catch as exc:
        raise kind(code, f"{type(exc).__name__}: {exc}") from exc
