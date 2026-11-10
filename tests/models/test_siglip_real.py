"""SigLIP2 with the real downloaded weights. Skipped unless the model runtime and files exist.

Run:  pytest tests/models/test_siglip_real.py -s   (prints timing and VRAM)
"""

from __future__ import annotations

import io
import json
import statistics
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
from PIL import Image

from model_download import build_head
from moderation.classifiers.image_siglip import (
    HEAD_META_FILE,
    SiglipImageClassifier,
    load_head,
)
from moderation.domain import CATEGORIES, Label
from moderation.policy import D1_POLICY
from tests.models.conftest import (
    HEAD_DIR,
    IMAGE_DIR,
    pick_device,
)

DEVICE, DTYPE = pick_device()

pytestmark = pytest.mark.skipif(
    not (IMAGE_DIR / "config.json").is_file(), reason=f"model not downloaded to {IMAGE_DIR}"
)


def _png(color: tuple[int, int, int], size: tuple[int, int] = (384, 384)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, "PNG")
    return buffer.getvalue()


@pytest.fixture(scope="module")
def head_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if (HEAD_DIR / HEAD_META_FILE).is_file():
        return HEAD_DIR
    return build_head(IMAGE_DIR, tmp_path_factory.mktemp("siglip2-head"), DEVICE)


@pytest.fixture(scope="module")
def loaded(head_dir: Path) -> Iterator[tuple[SiglipImageClassifier, dict[str, Any]]]:
    stats: dict[str, Any] = {"device": DEVICE, "dtype": DTYPE}
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
    start = time.perf_counter()
    clf = SiglipImageClassifier.load(IMAGE_DIR, head_dir, DEVICE, DTYPE)
    stats["load_s"] = time.perf_counter() - start
    if DEVICE == "cuda":
        stats["vram_after_load_mb"] = (torch.cuda.memory_allocated() - base) / 2**20
    clf.warmup()
    try:
        yield clf, stats
    finally:
        clf.close()


def test_plain_colour_image_is_allowed(loaded: tuple[SiglipImageClassifier, dict[str, Any]]) -> None:
    clf, _ = loaded
    [result] = clf.score_batch([_png((200, 200, 200))])
    scores = result.categories
    assert set(scores) == set(CATEGORIES)
    assert all(0.0 <= v <= 1.0 for v in scores.values())
    item = D1_POLICY.judge_image("m1", scores)
    assert item.label is Label.ALLOW, scores


def test_embedding_is_unit_length_and_768_wide(loaded: tuple[SiglipImageClassifier, dict[str, Any]]) -> None:
    clf, _ = loaded
    [result] = clf.score_batch([_png((90, 160, 40))])
    assert len(result.embedding) == 768
    assert sum(x * x for x in result.embedding) == pytest.approx(1.0, abs=1e-4)


async def test_async_score_matches_sync(loaded: tuple[SiglipImageClassifier, dict[str, Any]]) -> None:
    clf, _ = loaded
    data = [_png((30, 120, 200))]
    [sync] = clf.score_batch(data)
    [async_result] = await clf.score(data)
    assert async_result.categories == pytest.approx(sync.categories)
    assert async_result.embedding == pytest.approx(sync.embedding)


def test_timing_and_vram(loaded: tuple[SiglipImageClassifier, dict[str, Any]]) -> None:
    clf, stats = loaded
    single = [_png((i * 20 % 255, 90, 160)) for i in range(10)]
    timings = []
    for data in single:
        start = time.perf_counter()
        clf.score_batch([data])
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        timings.append((time.perf_counter() - start) * 1000)
    start = time.perf_counter()
    clf.score_batch(single[:4])
    batch4_ms = (time.perf_counter() - start) * 1000
    stats["single_ms_median"] = statistics.median(timings)
    stats["batch4_ms"] = batch4_ms
    if DEVICE == "cuda":
        stats["vram_peak_mb"] = torch.cuda.max_memory_allocated() / 2**20
    print("\nSIGLIP_STATS", {k: round(v, 1) if isinstance(v, float) else v for k, v in stats.items()})
    assert stats["single_ms_median"] > 0


def test_head_builder_reproduces_the_shipped_head(tmp_path: Path) -> None:
    """The builder (model_download.build_head) gives the same head as the one on disk. Built on CPU."""
    if not (HEAD_DIR / HEAD_META_FILE).is_file():
        pytest.skip(f"no shipped head at {HEAD_DIR}")
    shipped = load_head(HEAD_DIR)
    built = load_head(build_head(IMAGE_DIR, tmp_path, "cpu"))
    assert built.classes == shipped.classes
    assert json.loads((tmp_path / HEAD_META_FILE).read_text(encoding="utf-8"))["scoring"] == "softmax_v2"
    for got, want in zip(built.weight, shipped.weight, strict=True):
        assert got == pytest.approx(want, rel=1e-3, abs=1e-3)
