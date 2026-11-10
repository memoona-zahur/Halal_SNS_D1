"""Postgres store: jobs, decisions and the outbox (callbacks waiting to go out).

Claims use FOR UPDATE SKIP LOCKED with a lease, so several worker processes never take the
same job, and a crashed worker's jobs go back to the queue when their lease ends.
A new job sends a notification on NOTIFY_CHANNEL, so workers wait on LISTEN instead of polling.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from psycopg import AsyncConnection, sql
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

from moderation.config import Settings
from moderation.domain import Decision, ModerationRequest
from moderation.errors import ATTEMPTS_EXHAUSTED

NOTIFY_CHANNEL = "moderation_job"


@dataclass(frozen=True)
class ClaimedJob:
    job_id: uuid.UUID
    attempts: int  # tries including this one: the claim counts it
    request: ModerationRequest


@dataclass(frozen=True)
class OutboxItem:
    outbox_id: int
    callback_url: str
    body: dict[str, Any]
    attempts: int  # tries including the one about to be made


@dataclass(frozen=True)
class SaveResult:
    """A decision or failure that was written. `latency_ms` (decisions only): job accepted -> saved, by the database clock."""

    callback: OutboxItem | None  # None without a callback_url
    latency_ms: int | None = None


@dataclass(frozen=True)
class JobSnapshot:
    job_id: uuid.UUID
    status: str
    error_code: str | None
    label: str | None
    category: str | None
    reason: str | None
    latency_ms: int | None = None  # set once decided: how long the post showed "processing"


def decided_body(job_id: uuid.UUID, request: ModerationRequest, decision: Decision) -> dict[str, Any]:
    """Callback for a finished check: label, category and reason, plus rejected_images when any image could not be checked."""
    body: dict[str, Any] = {
        "event_type": "moderation.decided",
        "job_id": str(job_id),
        "content_id": request.content_id,
        "content_version": request.content_version,
        "label": decision.label.value,
        "category": decision.category,
        "reason": decision.reason,
    }
    if decision.rejected_images:
        body["rejected_images"] = [
            {"media_id": r.media_id, "error_code": r.error_code, "detail": r.detail}
            for r in decision.rejected_images
        ]
    return body


def failed_body(job_id: uuid.UUID, content_id: str, content_version: int, error_code: str) -> dict[str, Any]:
    return {
        "event_type": "moderation.failed",
        "job_id": str(job_id),
        "content_id": content_id,
        "content_version": content_version,
        "error_code": error_code,
    }


# One statement that always returns the job: a duplicate key hits the no-op DO UPDATE, which
# returns the existing row (even one a concurrent insert just committed). Only a new row, whose
# id is the one we generated, sends the notification.
CREATE_JOB_SQL = """
WITH up AS (
    INSERT INTO job (id, request_id, surface, content_id, content_version, status, payload)
    VALUES (%(id)s, %(request_id)s, %(surface)s, %(content_id)s, %(content_version)s, 'queued', %(payload)s)
    ON CONFLICT (surface, content_id, content_version) DO UPDATE SET updated_at = job.updated_at
    RETURNING id, status
)
SELECT up.id, up.status
  FROM up LEFT JOIN LATERAL (SELECT pg_notify(%(channel)s, '') WHERE up.id = %(id)s) AS notify ON true
"""

CLAIM_JOB_SQL = """
UPDATE job
   SET status = 'running',
       lease_until = now() + make_interval(secs => %(lease_s)s),
       attempts = attempts + 1,
       updated_at = now()
 WHERE id = (
       SELECT id FROM job
        WHERE status = 'queued' AND next_attempt_at <= now()
        ORDER BY created_at
        FOR UPDATE SKIP LOCKED
        LIMIT 1)
RETURNING id, surface, content_id, content_version, attempts, payload
"""

# Lease fencing: each write by a worker matches the attempts value its claim returned. A worker
# whose lease ran out, and whose job another worker re-claimed (attempts + 1), matches nothing.

# One statement: the job leaves running/queued for done, the decision and its image embeddings
# are written, and the callback row is reserved for this worker (due only after the lease) so it
# can post it now. No row comes back when the job is no longer ours (lease lost).
# %(vectors)s is a 2-D array, one row per media id; [i:i][:] slices out row i.
SAVE_DECISION_SQL = """
WITH done AS (
    UPDATE job SET status = 'done', lease_until = NULL, error_code = NULL, updated_at = now()
     WHERE id = %(job_id)s AND attempts = %(attempts)s AND status IN ('queued', 'running')
    RETURNING id, created_at
), dec AS (
    INSERT INTO decision (job_id, label, category, reason, items, policy_version, model_versions)
    SELECT id, %(label)s, %(category)s, %(reason)s, %(items)s, %(policy_version)s, %(model_versions)s FROM done
    RETURNING job_id
), emb AS (
    INSERT INTO image_embedding (job_id, media_id, model, embedding)
    SELECT dec.job_id, e.media_id, %(image_model)s, ARRAY(SELECT unnest((%(vectors)s::real[])[e.i:e.i][:]))
      FROM dec, unnest(%(media_ids)s::text[]) WITH ORDINALITY AS e(media_id, i)
), ob AS (
    INSERT INTO outbox (job_id, callback_url, body, status, attempts, next_attempt_at)
    SELECT job_id, %(callback_url)s, %(body)s, 'queued', 1, now() + make_interval(secs => %(lease_s)s)
      FROM dec WHERE %(callback_url)s::text IS NOT NULL
    RETURNING id, callback_url, body, attempts
)
SELECT done.id, ob.id AS outbox_id, ob.callback_url, ob.body, ob.attempts,
       (extract(epoch FROM now() - done.created_at) * 1000)::int AS latency_ms
  FROM done LEFT JOIN ob ON true
"""

MARK_FAILED_SQL = """
WITH failed AS (
    UPDATE job SET status = 'failed', error_code = %(error_code)s, lease_until = NULL, updated_at = now()
     WHERE id = %(job_id)s AND attempts = %(attempts)s AND status IN ('queued', 'running')
    RETURNING id
), ob AS (
    INSERT INTO outbox (job_id, callback_url, body, status, attempts, next_attempt_at)
    SELECT id, %(callback_url)s, %(body)s, 'queued', 1, now() + make_interval(secs => %(lease_s)s)
      FROM failed WHERE %(callback_url)s::text IS NOT NULL
    RETURNING id, callback_url, body, attempts
)
SELECT failed.id, ob.id AS outbox_id, ob.callback_url, ob.body, ob.attempts
  FROM failed LEFT JOIN ob ON true
"""

MARK_RETRY_SQL = """
UPDATE job
   SET status = 'queued',
       error_code = %(error_code)s,
       lease_until = NULL,
       next_attempt_at = now() + make_interval(secs => %(delay_s)s),
       updated_at = now()
 WHERE id = %(job_id)s AND attempts = %(attempts)s AND status = 'running'
RETURNING id
"""

EXTEND_LEASE_SQL = """
UPDATE job
   SET lease_until = now() + make_interval(secs => %(lease_s)s)
 WHERE id = %(job_id)s AND attempts = %(attempts)s AND status = 'running'
RETURNING id
"""

RECLAIM_SQL = """
UPDATE job
   SET status = CASE WHEN attempts >= %(max_attempts)s THEN 'dead' ELSE 'queued' END,
       error_code = CASE WHEN attempts >= %(max_attempts)s THEN %(exhausted)s ELSE error_code END,
       lease_until = NULL,
       next_attempt_at = now(),
       updated_at = now()
 WHERE status = 'running' AND lease_until < now()
RETURNING id, status, content_id, content_version, payload ->> 'callback_url' AS callback_url
"""

INSERT_DUE_OUTBOX_SQL = """
INSERT INTO outbox (job_id, callback_url, body, status, attempts, next_attempt_at)
VALUES (%(job_id)s, %(callback_url)s, %(body)s, 'queued', 0, now())
"""

CLAIM_OUTBOX_SQL = """
UPDATE outbox
   SET attempts = attempts + 1,
       next_attempt_at = now() + make_interval(secs => %(lease_s)s)
 WHERE id IN (
       SELECT id FROM outbox
        WHERE status = 'queued' AND next_attempt_at <= now()
        ORDER BY id
        FOR UPDATE SKIP LOCKED
        LIMIT %(limit)s)
RETURNING id, callback_url, body, attempts
"""

MARK_OUTBOX_SENT_SQL = "UPDATE outbox SET status = 'sent' WHERE id = %(id)s"
MARK_OUTBOX_DEAD_SQL = "UPDATE outbox SET status = 'dead' WHERE id = %(id)s"
MARK_OUTBOX_RETRY_SQL = "UPDATE outbox SET next_attempt_at = now() + make_interval(secs => %(delay_s)s) WHERE id = %(id)s"

GET_STATUS_SQL = """
SELECT j.id AS job_id, j.status, j.error_code, d.label, d.category, d.reason,
       (extract(epoch FROM d.created_at - j.created_at) * 1000)::int AS latency_ms
  FROM job j
  LEFT JOIN decision d ON d.job_id = j.id
 WHERE j.id = %(job_id)s
"""


def _outbox_item(row: dict[str, Any]) -> OutboxItem | None:
    if row["outbox_id"] is None:
        return None
    return OutboxItem(row["outbox_id"], row["callback_url"], row["body"], row["attempts"])


class Store:
    def __init__(self, cfg: Settings) -> None:
        self._cfg = cfg
        # min_size=1: an idle API or worker holds one connection, not psycopg's default of four.
        self._pool: AsyncConnectionPool[AsyncConnection[dict[str, Any]]] = AsyncConnectionPool(
            conninfo=cfg.database_url,
            open=False,
            min_size=1,
            max_size=cfg.db_pool_max_size,
            timeout=5,  # database down: answer 503 after 5 s, not psycopg's default 30 s
            kwargs={"row_factory": dict_row},
            check=AsyncConnectionPool.check_connection,  # replaces a connection that went stale, e.g. after a Postgres restart
        )

    async def open(self) -> None:
        await self._pool.open(wait=True)

    async def close(self) -> None:
        await self._pool.close()

    async def _fetch_one(self, query: str, params: dict[str, Any]) -> dict[str, Any] | None:
        async with self._pool.connection() as conn:
            cur = await conn.execute(query, params)
            return await cur.fetchone()

    async def _fetch_all(self, query: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        async with self._pool.connection() as conn:
            cur = await conn.execute(query, params)
            return await cur.fetchall()

    async def ping(self) -> None:
        await self._fetch_one("SELECT 1", {})

    async def create_job(self, request: ModerationRequest) -> tuple[uuid.UUID, str]:
        """Queue a job and wake the workers. The same surface + content id + version returns the existing job."""
        params = {
            "id": uuid.uuid4(),
            "request_id": request.request_id,
            "surface": request.surface,
            "content_id": request.content_id,
            "content_version": request.content_version,
            "payload": Jsonb(request.model_dump(mode="json")),
            "channel": NOTIFY_CHANNEL,
        }
        [row] = await self._fetch_all(CREATE_JOB_SQL, params)
        return row["id"], row["status"]

    async def claim_job(self) -> ClaimedJob | None:
        """Lease the oldest due job. None when there is none."""
        row = await self._fetch_one(CLAIM_JOB_SQL, {"lease_s": self._cfg.lease_s})
        if row is None:
            return None
        return ClaimedJob(row["id"], row["attempts"], ModerationRequest.model_validate(row["payload"]))

    async def save_decision(self, job: ClaimedJob, decision: Decision) -> SaveResult | None:
        """Record the decision, its model versions and image embeddings. Returns the callback to post now.

        None when the job is no longer ours (its lease ran out and another worker re-claimed it,
        or it was failed meanwhile); nothing is written then.
        """
        items = [
            {
                "source": item.source.value,
                "media_id": item.media_id,
                "label": item.label.value,
                "category": item.category,
                "score": item.score,
            }
            for item in decision.items
        ]
        row = await self._fetch_one(
            SAVE_DECISION_SQL,
            {
                "job_id": job.job_id,
                "attempts": job.attempts,
                "label": decision.label.value,
                "category": decision.category,
                "reason": decision.reason,
                "items": Jsonb(items),
                "policy_version": decision.policy_version,
                "model_versions": Jsonb(decision.model_versions),
                "image_model": decision.model_versions.get("image"),
                "media_ids": list(decision.embeddings),
                "vectors": [list(vector) for vector in decision.embeddings.values()],
                "callback_url": _url(job.request),
                "body": Jsonb(decided_body(job.job_id, job.request, decision)),
                "lease_s": self._cfg.lease_s,
            },
        )
        return None if row is None else SaveResult(_outbox_item(row), row["latency_ms"])

    async def fail_job(self, job: ClaimedJob, error_code: str) -> SaveResult | None:
        """Fail the job and reserve its moderation.failed callback to post now.
        None when the job is no longer ours; nothing is written then."""
        row = await self._fetch_one(
            MARK_FAILED_SQL,
            {
                "job_id": job.job_id,
                "attempts": job.attempts,
                "error_code": error_code,
                "callback_url": _url(job.request),
                "body": Jsonb(failed_body(job.job_id, job.request.content_id, job.request.content_version, error_code)),
                "lease_s": self._cfg.lease_s,
            },
        )
        return None if row is None else SaveResult(_outbox_item(row))

    async def requeue_job(self, job: ClaimedJob, error_code: str, delay_s: float) -> bool:
        """Put a running job back in the queue after `delay_s`. False when the job is no longer ours."""
        params = {"job_id": job.job_id, "attempts": job.attempts, "error_code": error_code, "delay_s": delay_s}
        row = await self._fetch_one(MARK_RETRY_SQL, params)
        return row is not None

    async def extend_lease(self, job: ClaimedJob) -> bool:
        """Push the lease of a running job out by lease_s. False when the job is no longer ours."""
        params = {"job_id": job.job_id, "attempts": job.attempts, "lease_s": self._cfg.lease_s}
        return await self._fetch_one(EXTEND_LEASE_SQL, params) is not None

    async def requeue_expired_jobs(self, max_attempts: int) -> int:
        """Requeue jobs whose worker vanished. Jobs out of attempts become dead and get a failure callback."""
        async with self._pool.connection() as conn, conn.transaction():
            cur = await conn.execute(RECLAIM_SQL, {"max_attempts": max_attempts, "exhausted": ATTEMPTS_EXHAUSTED})
            rows = await cur.fetchall()
            for row in rows:
                if row["status"] == "dead" and row["callback_url"] is not None:
                    body = failed_body(row["id"], row["content_id"], row["content_version"], ATTEMPTS_EXHAUSTED)
                    await conn.execute(
                        INSERT_DUE_OUTBOX_SQL,
                        {"job_id": row["id"], "callback_url": row["callback_url"], "body": Jsonb(body)},
                    )
        return len(rows)

    async def claim_outbox(self, limit: int) -> list[OutboxItem]:
        """Lease up to `limit` due callbacks: retries, and rows whose worker vanished before posting."""
        rows = await self._fetch_all(CLAIM_OUTBOX_SQL, {"lease_s": self._cfg.lease_s, "limit": limit})
        return [OutboxItem(r["id"], r["callback_url"], r["body"], r["attempts"]) for r in rows]

    async def mark_outbox_sent(self, outbox_id: int) -> None:
        await self._execute(MARK_OUTBOX_SENT_SQL, {"id": outbox_id})

    async def mark_outbox_retry(self, outbox_id: int, delay_s: float) -> None:
        await self._execute(MARK_OUTBOX_RETRY_SQL, {"id": outbox_id, "delay_s": delay_s})

    async def mark_outbox_dead(self, outbox_id: int) -> None:
        await self._execute(MARK_OUTBOX_DEAD_SQL, {"id": outbox_id})

    async def _execute(self, query: str, params: dict[str, Any]) -> None:
        async with self._pool.connection() as conn:
            await conn.execute(query, params)

    async def get_job_status(self, job_id: uuid.UUID) -> JobSnapshot | None:
        row = await self._fetch_one(GET_STATUS_SQL, {"job_id": job_id})
        return None if row is None else JobSnapshot(**row)

    async def listen_for_jobs(self) -> AsyncIterator[None]:
        """Yield once per new job, from a dedicated connection. Ends only on error."""
        async with await AsyncConnection.connect(self._cfg.database_url, autocommit=True) as conn:
            await conn.execute(sql.SQL("LISTEN {}").format(sql.Identifier(NOTIFY_CHANNEL)))
            async for _ in conn.notifies():
                yield


def _url(request: ModerationRequest) -> str | None:
    return None if request.callback_url is None else str(request.callback_url)
