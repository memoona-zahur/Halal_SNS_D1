from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from pathlib import Path

import httpx
import pytest

from moderation.config import Settings
from moderation.errors import (
    MEDIA_FORBIDDEN,
    MEDIA_NOT_FOUND,
    MEDIA_TOO_LARGE,
    MEDIA_UNSUPPORTED,
    PermanentError,
    RetryableError,
)
from moderation.fetch import ImageFetcher, is_animated

JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 64
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
WEBP = b"RIFF" + b"\x00\x00\x00\x00" + b"WEBP" + b"\x00" * 64
GIF = b"GIF89a" + b"\x00" * 64
HTML = b"<html><body>not an image</body></html>"

ALLOWED_HOST = "media.example.com"


def make_settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "database_url": "postgresql://unused",
        "image_allowed_hosts": ALLOWED_HOST,
        "image_max_bytes": 1_000_000,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)  # type: ignore[call-arg]


def make_fetcher(
    handler: Callable[[httpx.Request], httpx.Response], settings: Settings | None = None
) -> ImageFetcher:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=False, trust_env=False)
    return ImageFetcher(settings or make_settings(), client)


class ChunkStream(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk


# --- HTTP ------------------------------------------------------------------------


async def test_allowed_jpeg_is_returned() -> None:
    fetcher = make_fetcher(lambda req: httpx.Response(200, content=JPEG))
    assert await fetcher.fetch(f"https://{ALLOWED_HOST}/m1.jpg") == JPEG


@pytest.mark.parametrize("data", [JPEG, PNG, WEBP])
async def test_supported_formats_pass_magic_check(data: bytes) -> None:
    fetcher = make_fetcher(lambda req: httpx.Response(200, content=data))
    assert await fetcher.fetch(f"https://{ALLOWED_HOST}/x") == data


async def test_gif_is_returned_and_marked_animated() -> None:
    fetcher = make_fetcher(lambda req: httpx.Response(200, content=GIF))
    data = await fetcher.fetch(f"https://{ALLOWED_HOST}/anim.gif")
    assert is_animated(data)
    assert not is_animated(JPEG)


async def test_host_not_allowlisted_refused_before_any_request() -> None:
    calls: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(req)
        return httpx.Response(200, content=JPEG)

    with pytest.raises(PermanentError) as exc:
        await make_fetcher(handler).fetch("https://evil.example.org/m1.jpg")
    assert exc.value.code == MEDIA_FORBIDDEN
    assert calls == []


async def test_empty_allowlist_refuses_everything() -> None:
    calls: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(req)
        return httpx.Response(200, content=JPEG)

    fetcher = make_fetcher(handler, make_settings(image_allowed_hosts=""))
    with pytest.raises(PermanentError) as exc:
        await fetcher.fetch(f"https://{ALLOWED_HOST}/m1.jpg")
    assert exc.value.code == MEDIA_FORBIDDEN
    assert calls == []


async def test_plain_http_refused_for_allowlisted_host() -> None:
    fetcher = make_fetcher(lambda req: httpx.Response(200, content=JPEG))
    with pytest.raises(PermanentError) as exc:
        await fetcher.fetch(f"http://{ALLOWED_HOST}/m1.jpg")
    assert exc.value.code == MEDIA_FORBIDDEN


async def test_plain_http_localhost_allowed_only_with_flag() -> None:
    allowed = make_fetcher(
        lambda req: httpx.Response(200, content=PNG),
        make_settings(image_allowed_hosts="localhost", allow_http_localhost=True),
    )
    assert await allowed.fetch("http://localhost:8000/m1.png") == PNG

    refused = make_fetcher(
        lambda req: httpx.Response(200, content=PNG),
        make_settings(image_allowed_hosts="localhost", allow_http_localhost=False),
    )
    with pytest.raises(PermanentError) as exc:
        await refused.fetch("http://localhost:8000/m1.png")
    assert exc.value.code == MEDIA_FORBIDDEN


async def test_plain_http_docker_host_allowed_only_with_flag() -> None:
    allowed = make_fetcher(
        lambda req: httpx.Response(200, content=PNG),
        make_settings(image_allowed_hosts="host.docker.internal", allow_http_localhost=True),
    )
    assert await allowed.fetch("http://host.docker.internal:8090/examples/a.png") == PNG

    refused = make_fetcher(
        lambda req: httpx.Response(200, content=PNG),
        make_settings(image_allowed_hosts="host.docker.internal", allow_http_localhost=False),
    )
    with pytest.raises(PermanentError) as exc:
        await refused.fetch("http://host.docker.internal:8090/examples/a.png")
    assert exc.value.code == MEDIA_FORBIDDEN


async def test_credentials_in_url_refused() -> None:
    fetcher = make_fetcher(lambda req: httpx.Response(200, content=JPEG))
    with pytest.raises(PermanentError) as exc:
        await fetcher.fetch(f"https://user:pw@{ALLOWED_HOST}/m1.jpg")
    assert exc.value.code == MEDIA_FORBIDDEN


async def test_unparseable_url_refused() -> None:
    fetcher = make_fetcher(lambda req: httpx.Response(200, content=JPEG))
    with pytest.raises(PermanentError) as exc:
        await fetcher.fetch("https://[::1/m1.jpg")
    assert exc.value.code == MEDIA_FORBIDDEN


async def test_redirect_is_refused_and_not_followed() -> None:
    calls: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(req)
        return httpx.Response(302, headers={"location": "https://evil.example.org/x.jpg"})

    with pytest.raises(PermanentError) as exc:
        await make_fetcher(handler).fetch(f"https://{ALLOWED_HOST}/m1.jpg")
    assert exc.value.code == MEDIA_FORBIDDEN
    assert len(calls) == 1


async def test_declared_size_over_cap_refused_before_reading() -> None:
    fetcher = make_fetcher(
        lambda req: httpx.Response(200, headers={"content-length": "999999"}, content=JPEG),
        make_settings(image_max_bytes=2048),
    )
    with pytest.raises(PermanentError) as exc:
        await fetcher.fetch(f"https://{ALLOWED_HOST}/big.jpg")
    assert exc.value.code == MEDIA_TOO_LARGE


async def test_streamed_body_over_cap_is_cut_off() -> None:
    chunks = [b"\xff\xd8\xff" + b"\x00" * 999] + [b"\x00" * 1000] * 49
    fetcher = make_fetcher(
        lambda req: httpx.Response(200, stream=ChunkStream(chunks)), make_settings(image_max_bytes=5000)
    )
    with pytest.raises(PermanentError) as exc:
        await fetcher.fetch(f"https://{ALLOWED_HOST}/stream.jpg")
    assert exc.value.code == MEDIA_TOO_LARGE


async def test_bad_magic_bytes_are_unsupported() -> None:
    fetcher = make_fetcher(lambda req: httpx.Response(200, content=HTML))
    with pytest.raises(PermanentError) as exc:
        await fetcher.fetch(f"https://{ALLOWED_HOST}/page.jpg")
    assert exc.value.code == MEDIA_UNSUPPORTED


async def test_404_is_retryable_not_found() -> None:
    fetcher = make_fetcher(lambda req: httpx.Response(404))
    with pytest.raises(RetryableError) as exc:
        await fetcher.fetch(f"https://{ALLOWED_HOST}/late.jpg")
    assert exc.value.code == MEDIA_NOT_FOUND


async def test_403_is_permanent_forbidden() -> None:
    fetcher = make_fetcher(lambda req: httpx.Response(403))
    with pytest.raises(PermanentError) as exc:
        await fetcher.fetch(f"https://{ALLOWED_HOST}/private.jpg")
    assert exc.value.code == MEDIA_FORBIDDEN


@pytest.mark.parametrize("status", [429, 500, 503])
async def test_throttling_and_server_errors_are_retryable(status: int) -> None:
    fetcher = make_fetcher(lambda req: httpx.Response(status))
    with pytest.raises(RetryableError):
        await fetcher.fetch(f"https://{ALLOWED_HOST}/x.jpg")


async def test_other_client_error_is_permanent_not_found() -> None:
    fetcher = make_fetcher(lambda req: httpx.Response(410))
    with pytest.raises(PermanentError) as exc:
        await fetcher.fetch(f"https://{ALLOWED_HOST}/gone.jpg")
    assert exc.value.code == MEDIA_NOT_FOUND


@pytest.mark.parametrize(
    "exc_factory",
    [lambda: httpx.ReadTimeout("slow"), lambda: httpx.ConnectError("refused")],
)
async def test_timeout_and_connect_errors_are_retryable(exc_factory: Callable[[], Exception]) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        raise exc_factory()

    with pytest.raises(RetryableError) as exc:
        await make_fetcher(handler).fetch(f"https://{ALLOWED_HOST}/x.jpg")
    assert exc.value.code == MEDIA_NOT_FOUND


# --- local paths (tests only) ---------------------------------------------------------


def _no_http(req: httpx.Request) -> httpx.Response:
    raise AssertionError("a local path must not reach HTTP")


@pytest.fixture
def media_root(tmp_path: Path) -> Path:
    root = tmp_path / "media"
    root.mkdir()
    (root / "m1.jpg").write_bytes(JPEG)
    (tmp_path / "secret.jpg").write_bytes(JPEG)
    return root


def make_local(root: Path, **overrides: object) -> ImageFetcher:
    return make_fetcher(_no_http, make_settings(image_allowed_root=str(root), **overrides))


async def test_local_relative_path_under_root_is_read(media_root: Path) -> None:
    assert await make_local(media_root).fetch("m1.jpg") == JPEG


async def test_local_absolute_path_inside_root_is_read(media_root: Path) -> None:
    assert await make_local(media_root).fetch(str(media_root / "m1.jpg")) == JPEG


async def test_local_traversal_refused(media_root: Path) -> None:
    with pytest.raises(PermanentError) as exc:
        await make_local(media_root).fetch("../secret.jpg")
    assert exc.value.code == MEDIA_FORBIDDEN


async def test_local_absolute_path_outside_root_refused(media_root: Path) -> None:
    with pytest.raises(PermanentError) as exc:
        await make_local(media_root).fetch(str(media_root.parent / "secret.jpg"))
    assert exc.value.code == MEDIA_FORBIDDEN


async def test_local_missing_file_is_retryable(media_root: Path) -> None:
    with pytest.raises(RetryableError) as exc:
        await make_local(media_root).fetch("not-yet.jpg")
    assert exc.value.code == MEDIA_NOT_FOUND


async def test_local_oversized_file_refused(media_root: Path) -> None:
    (media_root / "big.jpg").write_bytes(JPEG + b"\x00" * 4096)
    with pytest.raises(PermanentError) as exc:
        await make_local(media_root, image_max_bytes=2048).fetch("big.jpg")
    assert exc.value.code == MEDIA_TOO_LARGE


async def test_local_bad_magic_bytes_unsupported(media_root: Path) -> None:
    (media_root / "fake.jpg").write_bytes(HTML)
    with pytest.raises(PermanentError) as exc:
        await make_local(media_root).fetch("fake.jpg")
    assert exc.value.code == MEDIA_UNSUPPORTED


async def test_local_paths_refused_when_root_is_unset(media_root: Path) -> None:
    fetcher = make_fetcher(_no_http, make_settings(image_allowed_root=""))
    with pytest.raises(PermanentError) as exc:
        await fetcher.fetch(str(media_root / "m1.jpg"))
    assert exc.value.code == MEDIA_FORBIDDEN


class SlowStream(httpx.AsyncByteStream):
    """Sends a small chunk every `delay_s`: each read is quick, the whole download is not."""

    def __init__(self, chunks: int, delay_s: float) -> None:
        self._chunks = chunks
        self._delay_s = delay_s

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield JPEG
        for _ in range(self._chunks):
            await asyncio.sleep(self._delay_s)
            yield b"\x00" * 16


async def test_whole_download_is_bounded_by_fetch_timeout() -> None:
    fetcher = make_fetcher(
        lambda req: httpx.Response(200, stream=SlowStream(chunks=20, delay_s=0.05)),
        make_settings(fetch_timeout_s=0.2),
    )
    with pytest.raises(RetryableError) as info:
        await fetcher.fetch(f"https://{ALLOWED_HOST}/slow.jpg")
    assert info.value.code == MEDIA_NOT_FOUND


def test_animated_webp_and_apng_are_detected() -> None:
    png = b"\x89PNG\r\n\x1a\n"
    still_webp = b"RIFF\x00\x00\x00\x00WEBPVP8 " + b"\x00" * 40
    anim_webp = b"RIFF\x00\x00\x00\x00WEBPVP8X\x0a\x00\x00\x00\x02\x00\x00\x00ANIM" + b"\x00" * 14
    assert is_animated(anim_webp)
    assert not is_animated(still_webp)
    assert is_animated(png + b"\x00\x00\x00\x08acTL" + b"\x00" * 8 + b"\x00\x00\x00\x00IDAT")
    assert not is_animated(png + b"\x00\x00\x00\x00IDAT" + b"\x00" * 8 + b"acTL")  # acTL after IDAT does not count
    assert not is_animated(png + b"\x00\x00\x00\x00IDAT")
