"""Shared fixtures for tests against a real Postgres. Skipped unless TEST_DATABASE_URL points at a *_test database."""

from __future__ import annotations

import asyncio
import os
import sys
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import psycopg
import pytest

from moderation.config import Settings
from moderation.domain import (
    Decision,
    ItemResult,
    ItemSource,
    Label,
    ModerationRequest,
)
from moderation.store import ClaimedJob, Store

if sys.platform == "win32":
    # psycopg async cannot run on the Proactor loop that Windows defaults to.
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

MIGRATION = Path(__file__).resolve().parents[2] / "migrations" / "001_init.sql"


@pytest.fixture
def db_url() -> str:
    """The suite TRUNCATEs every table, so it must never point at the app database."""
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL not set")
    if not urlsplit(url).path.rstrip("/").endswith("_test"):
        pytest.fail(f"TEST_DATABASE_URL must name a database ending in _test, got {urlsplit(url).path!r}")
    return url


@pytest.fixture
async def store(db_url: str) -> AsyncIterator[Store]:
    async with await psycopg.AsyncConnection.connect(db_url, autocommit=True) as conn:
        await conn.execute(MIGRATION.read_text(encoding="utf-8"))
        await conn.execute("TRUNCATE job, decision, outbox, image_embedding RESTART IDENTITY CASCADE")
    job_store = Store(Settings(database_url=db_url, lease_s=60))
    await job_store.open()
    yield job_store
    await job_store.close()


async def run_sql(db_url: str, query: str, params: dict[str, Any] | None = None) -> list[tuple[Any, ...]]:
    async with await psycopg.AsyncConnection.connect(db_url, autocommit=True) as conn:
        cur = await conn.execute(query, params)
        return await cur.fetchall() if cur.description else []


def make_request(content_id: str = "post_1", version: int = 1, callback: str | None = "http://cb.local/hook") -> ModerationRequest:
    return ModerationRequest.model_validate(
        {
            "request_id": str(uuid.uuid4()),
            "surface": "post",
            "content_id": content_id,
            "content_version": version,
            "text": "hello",
            "images": [{"media_id": "m1", "url": "https://img.local/m1.jpg"}],
            "callback_url": callback,
        }
    )


EMBEDDING = (0.6, 0.0, 0.8)
VERSIONS = {"text": "Qwen/Qwen3Guard-Gen-0.6B@abc", "image": "google/siglip2-base-patch16-384@def"}


def make_decision(label: Label = Label.BLOCK) -> Decision:
    return Decision(
        label=label,
        category="G-03",
        reason="Image m1: Sexual content and nudity",
        items=(ItemResult(ItemSource.IMAGE, "m1", label, "G-03", 0.91),),
        policy_version="d1.1.0",
        embeddings={"m1": EMBEDDING},
        model_versions=VERSIONS,
    )


async def claimed(store: Store, request: ModerationRequest) -> ClaimedJob:
    job_id, _ = await store.create_job(request)
    job = await store.claim_job()
    assert job is not None and job.job_id == job_id
    return job
