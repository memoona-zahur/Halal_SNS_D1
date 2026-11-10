from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("rapidocr", reason="model extras not installed: pip install -e .[models]")

from moderation.classifiers.ocr_rapidocr import RapidOcrEngine, ocr_model_paths


def test_model_paths_are_pp_ocrv5_mobile_inside_model_dir(tmp_path: Path) -> None:
    det, rec = ocr_model_paths(tmp_path)
    assert det == tmp_path / "ch_PP-OCRv5_det_mobile.onnx"
    assert rec == tmp_path / "ch_PP-OCRv5_rec_mobile.onnx"


def test_missing_models_stop_start_up_instead_of_downloading(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="model_download"):
        RapidOcrEngine(tmp_path)
    assert not any(tmp_path.iterdir())
