"""Real OCR models. Skipped unless the model extras are installed and the models were pulled."""

from __future__ import annotations

import io
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

pytest.importorskip("rapidocr", reason="model extras not installed: pip install -e .[models]")
from PIL import Image, ImageDraw, ImageFont

from moderation.classifiers.ocr_rapidocr import RapidOcrEngine, ocr_model_paths
from moderation.errors import MEDIA_UNDECODABLE, PermanentError
from tests.models.conftest import OCR_DIR

ARIAL = Path(r"C:\Windows\Fonts\arial.ttf")
YU_GOTHIC = Path(r"C:\Windows\Fonts\YuGothM.ttc")

pytestmark = pytest.mark.skipif(
    not all(p.is_file() for p in ocr_model_paths(OCR_DIR)),
    reason=f"OCR models not pulled into {OCR_DIR}",
)


def _render(text: str, font: Path, size: tuple[int, int] = (960, 540), pt: int = 56) -> bytes:
    img = Image.new("RGB", size, "white")
    ImageDraw.Draw(img).text((60, size[1] // 2 - pt), text, fill="black", font=ImageFont.truetype(str(font), pt))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=92)
    return buf.getvalue()


@pytest.fixture(scope="module")
def engine() -> Iterator[RapidOcrEngine]:
    ocr = RapidOcrEngine(OCR_DIR)
    yield ocr
    ocr.close()


@pytest.mark.skipif(not ARIAL.is_file(), reason="Arial font not available")
def test_reads_english(engine: RapidOcrEngine) -> None:
    [result] = engine.read_batch([_render("hello world, buy a gun now", ARIAL)])
    assert result.text == "hello world, buy a gun now"
    assert result.box_count == 1
    assert result.mean_confidence is not None and result.mean_confidence > 0.9


@pytest.mark.skipif(not YU_GOTHIC.is_file(), reason="Japanese font not available")
def test_reads_japanese(engine: RapidOcrEngine) -> None:
    [result] = engine.read_batch([_render("こんにちは世界", YU_GOTHIC)])
    assert result.text == "こんにちは世界"
    assert result.mean_confidence is not None and result.mean_confidence > 0.9


@pytest.mark.skipif(not ARIAL.is_file(), reason="Arial font not available")
def test_blank_image_has_no_boxes(engine: RapidOcrEngine) -> None:
    [result] = engine.read_batch([_render("", ARIAL)])
    assert result.box_count == 0
    assert result.mean_confidence is None
    assert result.text == ""


@pytest.mark.skipif(not ARIAL.is_file(), reason="Arial font not available")
def test_cpu_latency_per_960px_image_is_reported(engine: RapidOcrEngine) -> None:
    blob = _render("latency probe text", ARIAL)
    engine.read_batch([blob])
    runs = 5
    start = time.perf_counter()
    for _ in range(runs):
        engine.read_batch([blob])
    per_image_ms = (time.perf_counter() - start) / runs * 1000
    print(f"OCR CPU latency: {per_image_ms:.0f} ms per 960x540 image")
    # Loose bound: catches a regression to upscaling (~530 ms measured) or a hang, not jitter.
    assert per_image_ms < 2000


def test_undecodable_bytes_are_a_permanent_error(engine: RapidOcrEngine) -> None:
    with pytest.raises(PermanentError) as info:
        engine.read_batch([b"not an image at all"])
    assert info.value.code == MEDIA_UNDECODABLE


@pytest.mark.skipif(not ARIAL.is_file(), reason="Arial font not available")
async def test_async_read_matches_sync(engine: RapidOcrEngine) -> None:
    blob = _render("async read", ARIAL)
    assert await engine.read([blob]) == engine.read_batch([blob])
