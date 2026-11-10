"""Worker paths from plan §7, with an in-memory store, a scripted moderator and a mock callback server."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import psycopg
import pytest
import torch

from moderation.config import MODEL_PINS, Settings
from moderation.domain import Decision, Label, ModerationRequest
from moderation.errors import PermanentError, RetryableError
from moderation.prompts import PROMPT_VERSION
from moderation.store import (
    ClaimedJob,
    OutboxItem,
    SaveResult,
    decided_body,
    failed_body,
)
from moderation.worker import (
    Worker,
    backoff_delay,
    is_gpu_crash,
    pick_device,
    read_model_versions,
)

REQUEST = ModerationRequest.model_validate(
    {
        "request_id": "r1",
        "surface": "post",
        "content_id": "post_1",
        "content_version": 1,
        "text": "hello",
        "callback_url": "http://backend.local/hook",
    }
)
DECISION = Decision(Label.ALLOW, None, "Allowed", (), "d1.1.0")


class FakeStore:
    def __init__(self, *, ours: bool = True, drop_first_listen: bool = False) -> None:
        self.ours = ours  # False: the lease was lost, every guarded write matches nothing
        self.calls: list[tuple[Any, ...]] = []
        self.next_outbox = 1
        self.queued: list[ClaimedJob] = []  # what claim_job hands out, oldest first
        self.notify_on_connect = 0  # wake-ups the listener announces right after connecting
        self.drop_first_listen = drop_first_listen
        self.listens = 0

    def _callback(self, body: dict[str, Any]) -> OutboxItem:
        item = OutboxItem(self.next_outbox, str(REQUEST.callback_url), body, attempts=1)
        self.next_outbox += 1
        return item

    async def claim_job(self) -> ClaimedJob | None:
        return self.queued.pop(0) if self.queued else None

    async def save_decision(self, job: ClaimedJob, decision: Decision) -> SaveResult | None:
        self.calls.append(("save", decision.label))
        if not self.ours:
            return None
        return SaveResult(self._callback(decided_body(job.job_id, job.request, decision)), latency_ms=420)

    async def fail_job(self, job: ClaimedJob, error_code: str) -> SaveResult | None:
        self.calls.append(("failed", error_code))
        if not self.ours:
            return None
        body = failed_body(job.job_id, job.request.content_id, job.request.content_version, error_code)
        return SaveResult(self._callback(body))

    async def requeue_job(self, job: ClaimedJob, error_code: str, delay_s: float) -> bool:
        self.calls.append(("retry", delay_s, error_code))
        return self.ours

    async def mark_outbox_sent(self, outbox_id: int) -> None:
        self.calls.append(("sent", outbox_id))

    async def mark_outbox_retry(self, outbox_id: int, delay_s: float) -> None:
        self.calls.append(("outbox_retry", delay_s))

    async def mark_outbox_dead(self, outbox_id: int) -> None:
        self.calls.append(("outbox_dead", outbox_id))

    async def extend_lease(self, job: ClaimedJob) -> bool:
        self.calls.append(("extend",))
        return True

    async def requeue_expired_jobs(self, max_attempts: int) -> int:
        return 0

    async def claim_outbox(self, limit: int) -> list[OutboxItem]:
        return []

    async def listen_for_jobs(self) -> AsyncIterator[None]:
        self.listens += 1
        if self.drop_first_listen and self.listens == 1:
            raise psycopg.OperationalError("server closed the connection unexpectedly")
        for _ in range(self.notify_on_connect):
            yield
        await asyncio.Event().wait()  # then a quiet connection: no more notifications


class ScriptedModerator:
    def __init__(self, outcome: Decision | Exception) -> None:
        self.outcome = outcome

    async def moderate(self, request: ModerationRequest) -> Decision:
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


class CallbackServer:
    def __init__(self, status: int = 200) -> None:
        self.status = status
        self.bodies: list[Any] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.bodies.append(request.read())
        return httpx.Response(self.status)


def make_worker(
    outcome: Decision | Exception, *, ours: bool = True, status: int = 200, **cfg: Any
) -> tuple[Worker, FakeStore, CallbackServer]:
    store = FakeStore(ours=ours)
    server = CallbackServer(status)
    client = httpx.AsyncClient(transport=httpx.MockTransport(server))
    settings = Settings(database_url="postgresql://unused", **cfg)
    worker = Worker(settings, store, ScriptedModerator(outcome), client)  # type: ignore[arg-type]
    return worker, store, server


def job(attempts: int = 1) -> ClaimedJob:
    return ClaimedJob(job_id=uuid.uuid4(), attempts=attempts, request=REQUEST)


async def test_decided_job_posts_the_callback_and_marks_it_sent() -> None:
    worker, store, server = make_worker(DECISION)

    await worker.process(job())

    assert store.calls == [("save", Label.ALLOW), ("sent", 1)]
    assert len(server.bodies) == 1 and b'"moderation.decided"' in server.bodies[0]


async def test_callback_failure_retries_then_goes_dead() -> None:
    worker, store, _ = make_worker(DECISION, status=503, callback_max_attempts=2)

    await worker.process(job())
    assert store.calls == [("save", Label.ALLOW), ("outbox_retry", 1.0)]

    await worker.deliver_callback(OutboxItem(1, "http://backend.local/hook", {}, attempts=2))
    assert store.calls[-1] == ("outbox_dead", 1)


async def test_unreachable_backend_counts_as_not_delivered(caplog: pytest.LogCaptureFixture) -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    worker, store, _ = make_worker(DECISION)
    worker._callbacks = httpx.AsyncClient(transport=httpx.MockTransport(refuse))

    with caplog.at_level(logging.WARNING, logger="moderation"):
        await worker.process(job())

    assert store.calls == [("save", Label.ALLOW), ("outbox_retry", 1.0)]
    [record] = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert record.getMessage() == "callback will retry"
    assert record.status is None and record.exc_info is None  # type: ignore[attr-defined]


async def test_permanent_error_fails_the_job_with_its_code() -> None:
    worker, store, server = make_worker(PermanentError("media_unsupported", "not an image"))

    await worker.process(job())

    assert store.calls == [("failed", "media_unsupported"), ("sent", 1)]
    assert b'"media_unsupported"' in server.bodies[0]
    assert b'"job_id"' in server.bodies[0]


@pytest.mark.parametrize(("attempt", "delay"), [(1, 5.0), (2, 15.0), (3, 60.0), (4, 300.0)])
async def test_retryable_error_backs_off(attempt: int, delay: float) -> None:
    worker, store, _ = make_worker(RetryableError("media_not_found"))

    await worker.process(job(attempts=attempt))

    assert store.calls == [("retry", delay, "media_not_found")]


async def test_retryable_error_on_the_last_attempt_reports_its_own_code() -> None:
    worker, store, _ = make_worker(RetryableError("media_not_found"), max_attempts=5)

    await worker.process(job(attempts=5))

    assert store.calls[0] == ("failed", "media_not_found")


async def test_unknown_error_retries_then_fails_as_internal_error() -> None:
    worker, store, _ = make_worker(ValueError("bug"), max_attempts=5)

    await worker.process(job(attempts=1))
    await worker.process(job(attempts=5))

    assert store.calls[0] == ("retry", 5.0, "internal_error")
    assert store.calls[1] == ("failed", "internal_error")


async def test_lost_lease_writes_nothing_and_posts_nothing() -> None:
    worker, store, server = make_worker(DECISION, ours=False)

    await worker.process(job())

    assert store.calls == [("save", Label.ALLOW)]
    assert server.bodies == []


async def test_lease_is_renewed_while_a_job_runs_longer_than_a_third_of_it() -> None:
    class SlowModerator:
        async def moderate(self, request: ModerationRequest) -> Decision:
            await asyncio.sleep(0.5)
            return DECISION

    worker, store, _ = make_worker(DECISION, lease_s=5)
    worker._cfg = worker._cfg.model_copy(update={"lease_s": 0.3})  # renews every 0.1 s
    worker._moderator = SlowModerator()  # type: ignore[assignment]

    await worker.process(job())

    assert store.calls.count(("extend",)) >= 2
    count = store.calls.count(("extend",))
    await asyncio.sleep(0.2)
    assert store.calls.count(("extend",)) == count  # the renew task is cancelled when the job ends


def test_backoff_repeats_the_last_step() -> None:
    assert [backoff_delay((1.0, 5.0), n) for n in (1, 2, 3, 9)] == [1.0, 5.0, 5.0, 5.0]


@pytest.mark.parametrize(
    ("requested", "cuda", "expected"),
    [("auto", True, "cuda"), ("auto", False, "cpu"), ("cuda", True, "cuda"), ("cpu", True, "cpu")],
)
def test_device_table(requested: Any, cuda: bool, expected: str) -> None:
    assert pick_device(requested, cuda) == expected


def test_cuda_required_without_a_gpu_refuses_to_start() -> None:
    with pytest.raises(RuntimeError, match="DEVICE=cuda"):
        pick_device("cuda", cuda_works=False)


def test_model_versions_come_from_the_pins_the_head_and_the_ocr_files(tmp_path: Path) -> None:
    (tmp_path / "head").mkdir()
    (tmp_path / "head" / "head.json").write_text(
        json.dumps({"kind": "zero-shot", "created_at": "2026-10-21T10:00:00+00:00"}), encoding="utf-8"
    )
    cfg = Settings(
        database_url="postgresql://unused",
        image_head_dir=str(tmp_path / "head"),
        ocr_model_dir=str(tmp_path / "ocr"),
    )

    versions = read_model_versions(cfg)

    assert versions["text"] == "{}@{}+{}".format(*MODEL_PINS["text"], PROMPT_VERSION)
    assert versions["image"] == "{}@{}".format(*MODEL_PINS["image"])
    assert versions["image_head"] == "zero-shot@2026-10-21T10:00:00+00:00"
    assert versions["ocr"] == "ch_PP-OCRv5_det_mobile.onnx+ch_PP-OCRv5_rec_mobile.onnx"


# --- GPU crash (plan §7: "GPU crashes while running: restarts itself") -------------------------


async def test_gpu_crash_exits_with_70_and_writes_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_exit(code: int) -> None:
        raise SystemExit(code)

    monkeypatch.setattr("moderation.worker.os._exit", fake_exit)
    worker, store, server = make_worker(RuntimeError("CUDA error: an illegal memory access was encountered"))

    with pytest.raises(SystemExit) as info:
        await worker.process(job())

    assert info.value.code == 70
    assert store.calls == [] and server.bodies == []  # the lease puts the job back in the queue


def test_gpu_crash_detection() -> None:
    assert is_gpu_crash(RuntimeError("CUDA error: device-side assert triggered"))
    assert is_gpu_crash(torch.AcceleratorError("CUDA error: unspecified launch failure"))
    assert not is_gpu_crash(torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB"))
    assert not is_gpu_crash(RuntimeError("shape mismatch"))
    assert not is_gpu_crash(ValueError("CUDA error in a message is not enough without RuntimeError"))


# --- wake-ups (plan §7: missed notification, listener drop) ---------------------------------


async def _run_until(worker: Worker, done: Any, timeout_s: float = 3.0) -> None:
    task = asyncio.create_task(worker.run())
    async with asyncio.timeout(timeout_s):
        while not done():
            await asyncio.sleep(0.01)
    worker.stop()
    await task


async def test_poller_picks_up_a_job_that_was_never_announced() -> None:
    worker, store, _ = make_worker(DECISION, poll_s=0.05)

    async def enqueue_silently() -> None:
        await asyncio.sleep(0.15)  # after the start-up poll, and with no NOTIFY
        store.queued.append(job())

    feeder = asyncio.create_task(enqueue_silently())
    await _run_until(worker, lambda: ("sent", 1) in store.calls)
    await feeder

    assert store.calls == [("save", Label.ALLOW), ("sent", 1)]
    assert store.listens == 1  # the listener stayed connected and heard nothing


async def test_listener_reconnects_after_the_connection_drops(caplog: pytest.LogCaptureFixture) -> None:
    worker, store, _ = make_worker(DECISION, poll_s=0.05)
    store.drop_first_listen = True
    store.notify_on_connect = 1
    store.queued.append(job())

    with caplog.at_level(logging.ERROR, logger="moderation"):
        await _run_until(worker, lambda: store.listens >= 2 and ("sent", 1) in store.calls)

    assert store.listens == 2
    [dropped] = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert "listen_for_jobs failed" in dropped.getMessage() and dropped.exc_info is not None
