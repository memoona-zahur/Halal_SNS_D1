"""Text inside images: RapidOCR with PP-OCRv5 mobile det + rec on ONNX Runtime (CPU).

RapidOcrEngine loads the two local model files at worker start. It never downloads: a missing
file stops the worker at start-up instead of fetching at runtime. `model_download.py pull`
downloads them (see `ocr_model_urls`).

The "ch" PP-OCRv5 recogniser covers Chinese, English and Japanese, which is the D1 language set.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Final, cast

import numpy as np
from PIL import UnidentifiedImageError
from rapidocr import EngineType, LangDet, LangRec, ModelType, OCRVersion, RapidOCR
from rapidocr.inference_engine.base import FileInfo, InferSession
from rapidocr.utils.load_image import LoadImageError
from rapidocr.utils.output import RapidOCROutput
from rapidocr.utils.typings import TaskType

from moderation.classifiers import SingleThread
from moderation.domain import OcrResult
from moderation.errors import MEDIA_UNDECODABLE, PermanentError

log = logging.getLogger(__name__)

_DET: Final = FileInfo(
    engine_type=EngineType.ONNXRUNTIME,
    ocr_version=OCRVersion.PPOCRV5,
    task_type=TaskType.DET,
    lang_type=LangDet.CH,
    model_type=ModelType.MOBILE,
)
_REC: Final = FileInfo(
    engine_type=EngineType.ONNXRUNTIME,
    ocr_version=OCRVersion.PPOCRV5,
    task_type=TaskType.REC,
    lang_type=LangRec.CH,
    model_type=ModelType.MOBILE,
)


def ocr_model_urls() -> tuple[str, str]:
    """Download URLs of the (det, rec) models, from rapidocr's own model table."""
    det, rec = (str(InferSession.get_model_url(info)["model_dir"]) for info in (_DET, _REC))
    return det, rec


def ocr_model_paths(model_dir: Path) -> tuple[Path, Path]:
    det_url, rec_url = ocr_model_urls()
    return model_dir / Path(det_url).name, model_dir / Path(rec_url).name


class RapidOcrEngine:
    """OCR on CPU. Build once at worker start; `read` runs on this engine's own single thread."""

    def __init__(self, model_dir: Path, *, threads: int = 2, max_side: int = 960) -> None:
        det_path, rec_path = ocr_model_paths(model_dir)
        for path in (det_path, rec_path):
            if not path.is_file():
                # Checked here because RapidOCR would otherwise download the file at runtime.
                raise FileNotFoundError(f"OCR model missing: {path}; run python model_download.py pull")
        self._engine = RapidOCR(
            params={
                "Global.use_cls": False,
                # Keep every recognised line; mean_confidence decides readability, not RapidOCR.
                "Global.text_score": 0.0,
                "Global.max_side_len": max_side,
                # "warning" logs one line per image without text, i.e. most photos.
                "Global.log_level": "error",
                "Global.model_root_dir": str(model_dir),
                "EngineConfig.onnxruntime.intra_op_num_threads": threads,
                "EngineConfig.onnxruntime.inter_op_num_threads": 1,
                "Det.engine_type": _DET.engine_type,
                "Det.ocr_version": _DET.ocr_version,
                "Det.lang_type": _DET.lang_type,
                "Det.model_type": _DET.model_type,
                "Det.model_path": str(det_path),
                # Default is limit_type=min/736, which upscales small images before detection.
                # Cap the longest side instead: never upscale, downscale large images to max_side.
                "Det.limit_type": "max",
                "Det.limit_side_len": max_side,
                "Rec.engine_type": _REC.engine_type,
                "Rec.ocr_version": _REC.ocr_version,
                "Rec.lang_type": _REC.lang_type,
                "Rec.model_type": _REC.model_type,
                "Rec.model_path": str(rec_path),
            }
        )
        # RapidOCR loads models on first use and has no public eager-load; the private loaders make the first job fast.
        self._engine._load_det_model()  # type: ignore[no-untyped-call]
        self._engine._load_rec_model()  # type: ignore[no-untyped-call]
        # RapidOCR mutates per-call settings on the instance, so calls must not overlap: one thread.
        self._thread = SingleThread("ocr")
        log.info("OCR ready", extra={"det": det_path.name, "rec": rec_path.name, "max_side": max_side})

    async def read(self, images: Sequence[bytes]) -> list[OcrResult]:
        return await self._thread.run(self.read_batch, list(images))

    def read_batch(self, images: Sequence[bytes]) -> list[OcrResult]:
        return [self._read_one(blob) for blob in images]

    def close(self) -> None:
        self._thread.close()

    def _read_one(self, blob: bytes) -> OcrResult:
        try:
            out = cast(RapidOCROutput, self._engine(blob))
        except (LoadImageError, UnidentifiedImageError) as exc:  # RapidOCR decodes with Pillow
            raise PermanentError(MEDIA_UNDECODABLE, f"{type(exc).__name__}: {exc}") from exc
        if out.boxes is None or len(out.boxes) == 0:
            return OcrResult(text="", mean_confidence=None, box_count=0)
        confidences = [float(s) for s in out.scores or ()]
        text = "\n".join(str(t) for t in out.txts or () if str(t).strip())
        return OcrResult(
            text=text,
            mean_confidence=float(np.mean(confidences)) if confidences else None,
            box_count=len(out.boxes),
        )
