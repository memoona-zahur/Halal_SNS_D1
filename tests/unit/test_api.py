from __future__ import annotations

import uuid
from typing import Any

import httpx
import psycopg

from moderation.api import create_app
from moderation.config import Settings
from moderation.domain import ModerationRequest
from moderation.store import JobSnapshot


def make_settings(**overrides: Any) -> Settings:
    overrides.setdefault("api_wait_s", 0.0)  # most tests do not want to wait for a decision
    overrides.setdefault("api_keys", "")  # tests must not depend on a real .env's API_KEYS
    return Settings(database_url="postgresql://test:test@localhost:5432/test", **overrides)


class FakeStore:
    """In-memory stand-in for the three store calls the API makes."""

    def __init__(self) -> None:
        self.rows: dict[uuid.UUID, JobSnapshot] = {}
        self.by_key: dict[tuple[str, str, int], uuid.UUID] = {}
        self.down = False

    async def create_job(self, request: ModerationRequest) -> tuple[uuid.UUID, str]:
        self._check()
        key = (request.surface, request.content_id, request.content_version)
        if key not in self.by_key:
            job_id = uuid.uuid4()
            self.by_key[key] = job_id
            self.rows[job_id] = JobSnapshot(job_id, "queued", None, None, None, None)
        job_id = self.by_key[key]
        return job_id, self.rows[job_id].status

    async def get_job_status(self, job_id: uuid.UUID) -> JobSnapshot | None:
        self._check()
        return self.rows.get(job_id)

    async def ping(self) -> None:
        self._check()

    def _check(self) -> None:
        if self.down:
            raise psycopg.OperationalError("connection refused")


def post_body(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "request_id": "req-1",
        "surface": "post",
        "content_id": "post_123",
        "content_version": 1,
        "text": "a caption",
        "images": [{"media_id": "m1", "url": "https://cdn.example/m1.jpg"}],
        "callback_url": "http://backend.local/moderation-result",
    }
    body.update(overrides)
    return body


def client_for(store: FakeStore, **settings: Any) -> httpx.AsyncClient:
    app = create_app(settings=make_settings(**settings), store=store)  # type: ignore[arg-type]
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


async def test_post_returns_202_queued() -> None:
    store = FakeStore()
    async with client_for(store) as client:
        resp = await client.post("/v1/moderations", json=post_body())
    assert resp.status_code == 202
    assert resp.json()["status"] == "queued"
    uuid.UUID(resp.json()["job_id"])


async def test_same_content_and_version_returns_the_existing_job() -> None:
    async with client_for(FakeStore()) as client:
        first = await client.post("/v1/moderations", json=post_body())
        second = await client.post("/v1/moderations", json=post_body(request_id="req-2"))
    assert first.json()["job_id"] == second.json()["job_id"]


async def test_new_version_is_a_new_job() -> None:
    async with client_for(FakeStore()) as client:
        first = await client.post("/v1/moderations", json=post_body())
        second = await client.post("/v1/moderations", json=post_body(content_version=2))
    assert first.json()["job_id"] != second.json()["job_id"]


async def test_surface_outside_d1_is_400() -> None:
    async with client_for(FakeStore()) as client:
        resp = await client.post("/v1/moderations", json=post_body(surface="comment"))
    assert resp.status_code == 400
    assert resp.json() == {"error": "unknown_surface", "detail": "comment"}


async def test_limits_come_from_settings() -> None:
    images = [{"media_id": f"m{i}", "url": f"https://cdn.example/{i}.jpg"} for i in range(3)]
    async with client_for(FakeStore(), api_max_images=2, api_max_text_chars=5, api_max_url_chars=30) as client:
        too_many = await client.post("/v1/moderations", json=post_body(images=images))
        too_long = await client.post("/v1/moderations", json=post_body(text="x" * 6, images=[]))
        long_url = await client.post(
            "/v1/moderations", json=post_body(text=None, images=[{"media_id": "m1", "url": "https://cdn.example/" + "a" * 20}])
        )
    for resp in (too_many, too_long, long_url):
        assert resp.status_code == 422
        assert resp.json()["error"] == "limit_exceeded"


async def test_callback_url_limit_comes_from_settings() -> None:
    async with client_for(FakeStore(), api_max_callback_url_chars=30) as client:
        resp = await client.post("/v1/moderations", json=post_body(callback_url="http://backend.local/" + "a" * 20))
        ok = await client.post("/v1/moderations", json=post_body(callback_url="http://backend.local/hook"))
    assert resp.status_code == 422
    assert resp.json()["error"] == "limit_exceeded"
    assert ok.status_code == 202


async def test_no_content_and_extra_fields_are_422() -> None:
    async with client_for(FakeStore()) as client:
        empty = await client.post("/v1/moderations", json=post_body(text=None, images=[]))
        extra = await client.post("/v1/moderations", json=post_body(sha256="ab" * 32))
        videos = await client.post("/v1/moderations", json=post_body(videos=["https://cdn.example/v.mp4"]))
    assert empty.status_code == 422
    assert extra.status_code == 422
    assert videos.status_code == 422


async def test_empty_api_keys_disables_auth() -> None:
    """The default: no API_KEYS configured means the header is never checked."""
    async with client_for(FakeStore(), api_keys="") as client:
        resp = await client.post("/v1/moderations", json=post_body())
    assert resp.status_code == 202


async def test_missing_or_wrong_api_key_is_401() -> None:
    async with client_for(FakeStore(), api_keys="alice:secret123") as client:
        missing = await client.post("/v1/moderations", json=post_body())
        wrong = await client.post("/v1/moderations", json=post_body(), headers={"X-API-Key": "nope"})
    for resp in (missing, wrong):
        assert resp.status_code == 401
        assert resp.json() == {"error": "unauthorized", "detail": "missing or invalid X-API-Key"}


async def test_correct_api_key_is_accepted() -> None:
    async with client_for(FakeStore(), api_keys="alice:secret123") as client:
        resp = await client.post("/v1/moderations", json=post_body(), headers={"X-API-Key": "secret123"})
    assert resp.status_code == 202


async def test_api_key_also_guards_the_get_endpoint() -> None:
    async with client_for(FakeStore(), api_keys="alice:secret123") as client:
        resp = await client.get(f"/v1/moderations/{uuid.uuid4()}")
    assert resp.status_code == 401


async def test_health_endpoints_are_not_guarded_by_api_key() -> None:
    """/healthz has no store dependency and no auth; it must stay reachable even with API_KEYS set."""
    async with client_for(FakeStore(), api_keys="alice:secret123") as client:
        resp = await client.get("/healthz")
    assert resp.status_code == 200


async def test_get_status_and_unknown_job() -> None:
    store = FakeStore()
    async with client_for(store) as client:
        job_id = (await client.post("/v1/moderations", json=post_body())).json()["job_id"]
        queued = await client.get(f"/v1/moderations/{job_id}")
        store.rows[uuid.UUID(job_id)] = JobSnapshot(
            uuid.UUID(job_id), "done", None, "block", "G-03", "Image m1: Sexual content and nudity", latency_ms=812
        )
        done = await client.get(f"/v1/moderations/{job_id}")
        unknown = await client.get(f"/v1/moderations/{uuid.uuid4()}")
    assert queued.json()["status"] == "queued" and queued.json()["label"] is None
    assert queued.json()["latency_ms"] is None
    assert done.json()["label"] == "block" and done.json()["category"] == "G-03"
    assert done.json()["latency_ms"] == 812
    assert unknown.status_code == 404


async def test_health_and_readiness() -> None:
    store = FakeStore()
    async with client_for(store) as client:
        assert (await client.get("/healthz")).status_code == 200
        assert (await client.get("/readyz")).status_code == 200
        store.down = True
        assert (await client.get("/readyz")).status_code == 503
        assert (await client.post("/v1/moderations", json=post_body())).status_code == 503


async def test_duplicate_media_ids_are_refused() -> None:
    images = [{"media_id": "m1", "url": "https://cdn.example/a.jpg"}, {"media_id": "m1", "url": "https://cdn.example/b.jpg"}]
    async with client_for(FakeStore()) as client:
        resp = await client.post("/v1/moderations", json=post_body(images=images))
    assert resp.status_code == 422


class FinishingStore(FakeStore):
    """A job that is decided on its third status check."""

    def __init__(self) -> None:
        super().__init__()
        self.polls = 0

    async def get_job_status(self, job_id: uuid.UUID) -> JobSnapshot | None:
        self.polls += 1
        if self.polls >= 3 and job_id in self.rows:
            self.rows[job_id] = JobSnapshot(job_id, "done", None, "allow", None, "Allowed", 120)
        return await super().get_job_status(job_id)


async def test_post_waits_and_answers_200_with_the_decision() -> None:
    async with client_for(FinishingStore(), api_wait_s=2.0, api_wait_poll_s=0.01) as client:
        resp = await client.post("/v1/moderations", json=post_body())
    assert resp.status_code == 200
    assert resp.json()["status"] == "done"
    assert resp.json()["label"] == "allow"
    assert resp.json()["reason"] == "Allowed"


async def test_post_answers_202_when_the_check_is_not_finished_in_time() -> None:
    async with client_for(FakeStore(), api_wait_s=0.05, api_wait_poll_s=0.01) as client:
        resp = await client.post("/v1/moderations", json=post_body())
    assert resp.status_code == 202
    assert resp.json()["status"] == "queued"
    assert "job_id" in resp.json()


async def test_a_repeated_content_version_that_is_already_decided_answers_200_at_once() -> None:
    store = FakeStore()
    async with client_for(store, api_wait_s=2.0, api_wait_poll_s=0.01) as client:
        first = await client.post("/v1/moderations", json=post_body())
        job_id = uuid.UUID(first.json()["job_id"])
        store.rows[job_id] = JobSnapshot(job_id, "done", None, "block", "G-01", "Caption: Violence and gore", 90)
        again = await client.post("/v1/moderations", json=post_body())
    assert again.status_code == 200 and again.json()["label"] == "block"


async def test_body_larger_than_the_limit_is_refused_before_it_is_read() -> None:
    async with client_for(FakeStore(), api_max_body_bytes=1024) as client:
        resp = await client.post("/v1/moderations", content=b"x" * 2048, headers={"content-type": "application/json"})
    assert resp.status_code == 413
    assert resp.json()["error"] == "body_too_large"
