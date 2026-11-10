"""Image download.

Production: http(s) URLs from allowlisted hosts. Tests and local runs: a path under
IMAGE_ALLOWED_ROOT. Both return the raw bytes after a size cap and a file-type check.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from moderation.config import Settings
from moderation.errors import (
    MEDIA_FORBIDDEN,
    MEDIA_NOT_FOUND,
    MEDIA_TOO_LARGE,
    MEDIA_UNSUPPORTED,
    PermanentError,
    RetryableError,
    translate,
)

_LOCALHOST = frozenset({"localhost", "127.0.0.1", "host.docker.internal"})  # dev only, with ALLOW_HTTP_LOCALHOST
_CHUNK_BYTES = 8192
_GIF_MAGIC = (b"GIF87a", b"GIF89a")
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def is_animated(data: bytes) -> bool:
    """GIF, animated WebP (ANIM chunk) or APNG (acTL before the first IDAT). D1 cannot check motion, so the caller holds it."""
    if data[:6] in _GIF_MAGIC:
        return True
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return b"ANIM" in data[12:64]
    if data[:8] == _PNG_MAGIC:
        idat = data.find(b"IDAT")
        return 0 < data.find(b"acTL") < (idat if idat != -1 else len(data))
    return False


def guess_extension(data: bytes) -> str:
    """File extension by content, for files we write ourselves (so they open in an image viewer).
    Call after `_check_type` has already accepted the data."""
    if data[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if data[:8] == _PNG_MAGIC:
        return ".png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    return ".gif"  # the only other type _check_type accepts (via is_animated's GIF branch)


def _check_type(data: bytes) -> bytes:
    """Accept JPEG, PNG, WebP and GIF by magic bytes. The extension and Content-Type are ignored."""
    if (
        data[:3] == b"\xff\xd8\xff"
        or data[:8] == _PNG_MAGIC
        or (data[:4] == b"RIFF" and data[8:12] == b"WEBP")
        or is_animated(data)
    ):
        return data
    raise PermanentError(MEDIA_UNSUPPORTED, "not a JPEG, PNG, WebP or GIF image")


def _check_status(status: int) -> None:
    if 200 <= status < 300:
        return
    if 300 <= status < 400 or status in (401, 403):
        raise PermanentError(MEDIA_FORBIDDEN, f"status {status}")  # redirects are never followed
    if status in (404, 429) or status >= 500:
        raise RetryableError(MEDIA_NOT_FOUND, f"status {status}")  # 404: not replicated yet
    raise PermanentError(MEDIA_NOT_FOUND, f"status {status}")


class ImageFetcher:
    """The caller owns `client` (follow_redirects=False, trust_env=False, timeout set)."""

    def __init__(self, cfg: Settings, client: httpx.AsyncClient) -> None:
        self._client = client
        self._hosts = cfg.allowed_hosts
        self._allow_http_localhost = cfg.allow_http_localhost
        self._max_bytes = cfg.image_max_bytes
        self._timeout_s = cfg.fetch_timeout_s
        self._root = Path(cfg.image_allowed_root).resolve() if cfg.image_allowed_root.strip() else None

    async def fetch(self, url: str) -> bytes:
        if url.startswith(("http://", "https://")):
            return _check_type(await self._download(url))
        return _check_type(await asyncio.to_thread(self._read_local, url))

    async def _download(self, url: str) -> bytes:
        """FETCH_TIMEOUT_S is the budget for the whole download, so a slow-drip server cannot hold a job."""
        self._check_url(url)
        body = bytearray()
        with translate(RetryableError, MEDIA_NOT_FOUND, httpx.TransportError, TimeoutError):
            async with asyncio.timeout(self._timeout_s), self._client.stream("GET", url) as response:
                _check_status(response.status_code)
                declared = response.headers.get("content-length")
                if declared is not None and declared.isdigit() and int(declared) > self._max_bytes:
                    raise PermanentError(MEDIA_TOO_LARGE, f"declared {declared} bytes exceeds {self._max_bytes}")
                async for chunk in response.aiter_bytes(_CHUNK_BYTES):
                    body.extend(chunk)
                    if len(body) > self._max_bytes:
                        raise PermanentError(MEDIA_TOO_LARGE, f"body exceeds {self._max_bytes} bytes")
        return bytes(body)

    def _check_url(self, url: str) -> None:
        with translate(PermanentError, MEDIA_FORBIDDEN, ValueError):
            parts = urlsplit(url)
            host = (parts.hostname or "").lower()  # a malformed netloc raises ValueError here
        if parts.username is not None or parts.password is not None:
            raise PermanentError(MEDIA_FORBIDDEN, "credentials in URL are not allowed")
        if host not in self._hosts:
            raise PermanentError(MEDIA_FORBIDDEN, f"host {host!r} is not allowlisted")
        if parts.scheme == "https":
            return
        if parts.scheme == "http" and self._allow_http_localhost and host in _LOCALHOST:
            return
        raise PermanentError(MEDIA_FORBIDDEN, f"scheme {parts.scheme!r} is not allowed for this host")

    def _read_local(self, url: str) -> bytes:
        """`url` is a path relative to IMAGE_ALLOWED_ROOT, or an absolute path inside it."""
        if self._root is None:
            raise PermanentError(MEDIA_FORBIDDEN, "local image paths are disabled (IMAGE_ALLOWED_ROOT is empty)")
        path = (self._root / url).resolve()
        if not path.is_relative_to(self._root):
            raise PermanentError(MEDIA_FORBIDDEN, "path is outside IMAGE_ALLOWED_ROOT")
        with translate(RetryableError, MEDIA_NOT_FOUND, FileNotFoundError):
            if path.stat().st_size > self._max_bytes:
                raise PermanentError(MEDIA_TOO_LARGE, f"file exceeds {self._max_bytes} bytes")
            return path.read_bytes()
