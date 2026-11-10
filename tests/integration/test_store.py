"""Store tests against a real Postgres (fixtures in conftest.py)."""

from __future__ import annotations

import asyncio
import uuid

import psycopg
import pytest

from moderation.domain import Decision, Label
from moderation.store import ClaimedJob, Store
from tests.integration.conftest import (
    EMBEDDING,
    VERSIONS,
    claimed,
    make_decision,
    make_request,
    run_sql,
)


async def test_create_is_idempotent_and_notifies_only_new_jobs(store: Store) -> None:
    heard = 0

    async def listen() -> None:
        nonlocal heard
        async for _ in store.listen_for_jobs():
            heard += 1

    listener = asyncio.create_task(listen())
    await asyncio.sleep(0.3)  # let LISTEN register
    request = make_request()
    first = await store.create_job(request)
    second = await store.create_job(request)
    await asyncio.sleep(0.3)
    listener.cancel()

    assert first == second
    assert first[1] == "queued"
    assert heard == 1


async def test_claim_rebuilds_the_request(store: Store) -> None:
    request = make_request()
    await store.create_job(request)

    job = await store.claim_job()

    assert job is not None
    assert job.attempts == 1
    assert job.request == request


async def test_concurrent_claims_get_different_jobs(store: Store) -> None:
    await store.create_job(make_request("post_a"))
    await store.create_job(make_request("post_b"))

    first, second = await asyncio.gather(store.claim_job(), store.claim_job())

    assert first is not None and second is not None
    assert first.job_id != second.job_id
    assert await store.claim_job() is None


async def test_expired_lease_is_requeued(store: Store, db_url: str) -> None:
    job = await claimed(store, make_request())
    job_id = job.job_id
    await run_sql(db_url, "UPDATE job SET lease_until = now() - interval '1 second' WHERE id = %(id)s", {"id": job_id})

    assert await store.requeue_expired_jobs(max_attempts=5) == 1
    again = await store.claim_job()
    assert again is not None and again.job_id == job_id and again.attempts == 2


async def test_exhausted_lease_becomes_dead_and_queues_failure(store: Store, db_url: str) -> None:
    job = await claimed(store, make_request())
    job_id = job.job_id
    await run_sql(
        db_url,
        "UPDATE job SET lease_until = now() - interval '1 second', attempts = 5 WHERE id = %(id)s",
        {"id": job_id},
    )

    assert await store.requeue_expired_jobs(max_attempts=5) == 1

    status = await store.get_job_status(job_id)
    assert status is not None and status.status == "dead" and status.error_code == "attempts_exhausted"
    [item] = await store.claim_outbox(10)
    assert item.body == {
        "event_type": "moderation.failed",
        "job_id": str(job_id),
        "content_id": "post_1",
        "content_version": 1,
        "error_code": "attempts_exhausted",
    }


async def test_save_decision_marks_done_and_reserves_the_callback(store: Store) -> None:
    request = make_request()
    job = await claimed(store, request)
    job_id = job.job_id

    saved = await store.save_decision(job, make_decision())

    assert saved is not None
    callback = saved.callback
    assert callback is not None and callback.attempts == 1
    assert callback.callback_url == "http://cb.local/hook"
    assert callback.body["label"] == "block" and callback.body["job_id"] == str(job_id)
    status = await store.get_job_status(job_id)
    assert status is not None and (status.status, status.label, status.category) == ("done", "block", "G-03")
    assert await store.claim_outbox(10) == []  # reserved for the worker that saved it


async def test_save_decision_without_callback_url(store: Store) -> None:
    request = make_request(callback=None)
    job = await claimed(store, request)

    saved = await store.save_decision(job, make_decision(Label.ALLOW))
    assert saved is not None and saved.callback is None


async def test_save_decision_on_a_job_that_is_not_ours_writes_nothing(store: Store) -> None:
    request = make_request()
    job = await claimed(store, request)
    job_id = job.job_id
    await store.fail_job(job, "media_not_found")

    assert await store.save_decision(job, make_decision()) is None
    status = await store.get_job_status(job_id)
    assert status is not None and status.status == "failed" and status.label is None


async def test_save_decision_is_atomic(store: Store, db_url: str) -> None:
    request = make_request(callback="http://cb.local/" + "a" * 2050)  # over the outbox 2048-char check
    job = await claimed(store, request)
    job_id = job.job_id

    with pytest.raises(psycopg.errors.CheckViolation):
        await store.save_decision(job, make_decision())

    status = await store.get_job_status(job_id)
    assert status is not None and status.status == "running" and status.label is None
    assert await run_sql(db_url, "SELECT 1 FROM image_embedding WHERE job_id = %(id)s", {"id": job_id}) == []


async def test_outbox_retry_makes_it_due_and_sent_stops_it(store: Store) -> None:
    request = make_request()
    job = await claimed(store, request)
    saved = await store.save_decision(job, make_decision())
    callback = saved.callback if saved is not None else None
    assert callback is not None

    await store.mark_outbox_retry(callback.outbox_id, delay_s=0)
    [due] = await store.claim_outbox(10)
    assert due.outbox_id == callback.outbox_id and due.attempts == 2
    assert await store.claim_outbox(10) == []  # leased by the claim

    await store.mark_outbox_retry(callback.outbox_id, delay_s=0)
    await store.mark_outbox_sent(callback.outbox_id)
    assert await store.claim_outbox(10) == []


async def test_dead_outbox_row_is_not_claimed(store: Store) -> None:
    request = make_request()
    job = await claimed(store, request)
    saved = await store.save_decision(job, make_decision())
    callback = saved.callback if saved is not None else None
    assert callback is not None

    await store.mark_outbox_retry(callback.outbox_id, delay_s=0)
    await store.mark_outbox_dead(callback.outbox_id)
    assert await store.claim_outbox(10) == []


async def test_fail_job_reserves_the_failure_callback(store: Store) -> None:
    request = make_request()
    job = await claimed(store, request)
    job_id = job.job_id

    failed = await store.fail_job(job, "media_not_found")

    assert failed is not None and failed.callback is not None
    assert failed.callback.body == {
        "event_type": "moderation.failed",
        "job_id": str(job_id),
        "content_id": "post_1",
        "content_version": 1,
        "error_code": "media_not_found",
    }
    status = await store.get_job_status(job_id)
    assert status is not None and (status.status, status.error_code) == ("failed", "media_not_found")


async def test_requeue_job_needs_a_running_job(store: Store) -> None:
    request = make_request()
    job_id, _ = await store.create_job(request)
    not_claimed = ClaimedJob(job_id=job_id, attempts=0, request=request)

    assert await store.requeue_job(not_claimed, "media_not_found", delay_s=5) is False
    job = await store.claim_job()
    assert job is not None
    assert await store.requeue_job(job, "media_not_found", delay_s=0) is True
    status = await store.get_job_status(job_id)
    assert status is not None and (status.status, status.error_code) == ("queued", "media_not_found")


async def test_ping_and_unknown_status(store: Store) -> None:
    await store.ping()
    assert await store.get_job_status(uuid.uuid4()) is None


async def test_save_decision_stores_embeddings_and_model_versions_in_the_same_statement(
    store: Store, db_url: str
) -> None:
    request = make_request()
    job = await claimed(store, request)
    job_id = job.job_id

    assert await store.save_decision(job, make_decision()) is not None

    [(media_id, model, embedding)] = await run_sql(
        db_url, "SELECT media_id, model, embedding FROM image_embedding WHERE job_id = %(id)s", {"id": job_id}
    )
    assert (media_id, model) == ("m1", VERSIONS["image"])
    assert embedding == pytest.approx(list(EMBEDDING))
    [(versions,)] = await run_sql(db_url, "SELECT model_versions FROM decision WHERE job_id = %(id)s", {"id": job_id})
    assert versions == VERSIONS


async def test_decision_without_images_stores_no_embedding(store: Store, db_url: str) -> None:
    request = make_request()
    job = await claimed(store, request)
    job_id = job.job_id
    text_only = Decision(Label.ALLOW, None, "Allowed", (), "d1.1.0")

    assert await store.save_decision(job, text_only) is not None
    assert await run_sql(db_url, "SELECT 1 FROM image_embedding WHERE job_id = %(id)s", {"id": job_id}) == []


async def test_latency_is_reported_once_decided(store: Store, db_url: str) -> None:
    request = make_request()
    job = await claimed(store, request)
    job_id = job.job_id
    await run_sql(db_url, "UPDATE job SET created_at = now() - interval '1.5 seconds' WHERE id = %(id)s", {"id": job_id})
    before = await store.get_job_status(job_id)

    saved = await store.save_decision(job, make_decision())

    after = await store.get_job_status(job_id)
    assert before is not None and before.latency_ms is None
    assert saved is not None and after is not None
    assert saved.latency_ms is not None and 1500 <= saved.latency_ms < 10_000
    assert after.latency_ms == saved.latency_ms


async def test_extend_lease_only_works_for_the_owner_attempt(store: Store, db_url: str) -> None:
    stale = await claimed(store, make_request())
    await _expire(db_url, stale)
    assert await store.extend_lease(stale) is True  # still ours and running: the lease moves out
    await _expire(db_url, stale)
    await store.requeue_expired_jobs(max_attempts=5)
    fresh = await store.claim_job()
    assert fresh is not None

    assert await store.extend_lease(stale) is False  # another attempt holds it now
    assert await store.extend_lease(fresh) is True


# --- lease fencing: a worker whose lease ran out cannot overwrite the job --------------------


async def _expire(db_url: str, job: ClaimedJob) -> None:
    await run_sql(db_url, "UPDATE job SET lease_until = now() - interval '1 second' WHERE id = %(id)s", {"id": job.job_id})


async def test_a_stale_worker_cannot_write_after_another_worker_reclaimed_the_job(store: Store, db_url: str) -> None:
    stale = await claimed(store, make_request())
    await _expire(db_url, stale)
    assert await store.requeue_expired_jobs(max_attempts=5) == 1
    fresh = await store.claim_job()
    assert fresh is not None and fresh.job_id == stale.job_id and fresh.attempts == stale.attempts + 1

    assert await store.save_decision(stale, make_decision()) is None
    assert await store.requeue_job(stale, "media_not_found", delay_s=0) is False
    assert await store.fail_job(stale, "media_not_found") is None
    status = await store.get_job_status(stale.job_id)
    assert status is not None and status.status == "running" and status.label is None

    assert await store.save_decision(fresh, make_decision()) is not None  # the current holder still can


async def test_a_stale_worker_may_finish_a_requeued_job_nobody_claimed_yet(store: Store, db_url: str) -> None:
    stale = await claimed(store, make_request())
    await _expire(db_url, stale)
    await store.requeue_expired_jobs(max_attempts=5)

    assert await store.save_decision(stale, make_decision()) is not None  # no other worker holds it
    assert await store.claim_job() is None  # done: it is not checked twice
