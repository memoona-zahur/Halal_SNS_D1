"""Test doubles for the models and fetcher that moderation.moderate.Moderator takes.

Keyword and byte-marker rules, not moderation models. They prove the pipeline works end to
end and say nothing about moderation quality.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence

from moderation.domain import CATEGORIES, ImageScore, OcrResult, TextScore, TextTier

JPEG_MAGIC = b"\xff\xd8\xff\xe0"
GIF = b"GIF89a" + b"\x00" * 16
FAKE_EMBEDDING = (0.6, 0.8)  # unit length, like a real normalised embedding


def jpeg(marker: bytes = b"") -> bytes:
    """Bytes that pass the JPEG magic check and carry a FakeImage marker, e.g. jpeg(b"FAKE:GUN")."""
    return JPEG_MAGIC + b"\x00" * 8 + marker


_TEXT_RULES: tuple[tuple[re.Pattern[str], TextTier, str, float], ...] = (
    (re.compile(r"\b(gun|kill|bomb|knife)\b", re.IGNORECASE), "Unsafe", "Violent", 0.95),
    (re.compile(r"\bnude\b", re.IGNORECASE), "Unsafe", "Sexual Content or Sexual Acts", 0.92),
    (re.compile(r"\bhate\b", re.IGNORECASE), "Unsafe", "Unethical Acts", 0.90),
    (re.compile(r"\bphone\b", re.IGNORECASE), "Unsafe", "PII", 0.90),
    (re.compile(r"\bdebate\b", re.IGNORECASE), "Controversial", "Politically Sensitive Topics", 0.60),
)

# Image bytes containing a marker get a high score for that G-code.
_IMAGE_MARKERS: tuple[tuple[bytes, str, float], ...] = (
    (b"FAKE:BLOOD", "G-01", 0.92),
    (b"FAKE:GUN", "G-02", 0.90),
    (b"FAKE:NUDE", "G-03", 0.95),
    (b"FAKE:SOFT", "G-03", 0.85),
    (b"FAKE:HATE", "G-04", 0.90),
)
_NAN_MARKER = b"FAKE:NAN"
_BASELINE = 0.02


class FakeText:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def score(self, texts: Sequence[str]) -> list[TextScore]:
        self.calls.append(list(texts))
        return [self._score_one(text) for text in texts]

    @staticmethod
    def _score_one(text: str) -> TextScore:
        for pattern, tier, category, value in _TEXT_RULES:
            if pattern.search(text):
                return TextScore(tier=tier, category=category, score=value)
        return TextScore(tier="Safe", category=None, score=0.99)


class FakeImage:
    def __init__(self) -> None:
        self.calls = 0

    async def score(self, images: Sequence[bytes]) -> list[ImageScore]:
        self.calls += 1
        return [ImageScore(self._score_one(data), FAKE_EMBEDDING) for data in images]

    @staticmethod
    def _score_one(data: bytes) -> dict[str, float]:
        if _NAN_MARKER in data:
            return {code: math.nan for code in CATEGORIES}
        scores = {code: _BASELINE for code in CATEGORIES}
        for marker, code, value in _IMAGE_MARKERS:
            if marker in data:
                scores[code] = max(scores[code], value)
        return scores


class FakeOcr:
    """A fixed OcrResult per image bytes; no text boxes for anything else."""

    def __init__(self, results: Mapping[bytes, OcrResult] | None = None) -> None:
        self._results = dict(results or {})
        self.calls = 0

    async def read(self, images: Sequence[bytes]) -> list[OcrResult]:
        self.calls += 1
        return [self._results.get(blob, OcrResult("", None, 0)) for blob in images]


class FakeFetcher:
    """Serves bytes by URL. A value that is an exception is raised instead."""

    def __init__(self, files: Mapping[str, bytes | Exception]) -> None:
        self._files = dict(files)

    async def fetch(self, url: str) -> bytes:
        value = self._files[url]
        if isinstance(value, Exception):
            raise value
        return value
