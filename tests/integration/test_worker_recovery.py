"""Plan §7 "Checker crashes": a worker dies mid-job, its lease runs out, another worker finishes it."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx

from moderation.config import Settings
from moderation.domain import Decision, Label, ModerationRequest
from moderation.store import Store
from moderation.worker import Worker
from tests.integration.conftest import make_decision, make_request, run_sql


class ScriptedModerator:
    def __init__(self, decision: Decision) -> None:
        self.decision = decision

    async def moderate(self, request: ModerationRequest) -> Decision:
        return self.decision


async def test_job_of_a_crashed_worker_is_finished_by_another_worker(store: Store, db_url: str) -> None:
    await store.create_job(make_request())
    crashed = await store.claim_job()  # worker A claims the job, then its process dies
    assert crashed is not None
    await run_sql(
        db_url, "UPDATE job SET lease_until = now() - interval '1 second' WHERE id = %(id)s", {"id": crashed.job_id}
    )

    received: list[dict[str, Any]] = []

    def backend(request: httpx.Request) -> httpx.Response:
        received.append(json.loads(request.read()))
        return httpx.Response(200)

    survivor = Worker(
        Settings(database_url=db_url, poll_s=0.05),
        store,
        ScriptedModerator(make_decision(Label.ALLOW)),  # type: ignore[arg-type]
        httpx.AsyncClient(transport=httpx.MockTransport(backend)),
    )
    task = asyncio.create_task(survivor.run())  # reclaims at start, then claims on the next wake-up
    async with asyncio.timeout(5):
        while not received:
            await asyncio.sleep(0.02)
    survivor.stop()
    await task

    status = await store.get_job_status(crashed.job_id)
    assert status is not None and (status.status, status.label) == ("done", "allow")
    assert [(body["event_type"], body["job_id"]) for body in received] == [("moderation.decided", str(crashed.job_id))]
    # Worker A wakes up late: fencing drops its write, so the post is decided once.
    assert await store.save_decision(crashed, make_decision(Label.BLOCK)) is None
