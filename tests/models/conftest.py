"""Real-model tests. Skipped unless torch, transformers and the downloaded weights are present.

Model folders follow the Settings defaults (TEXT_MODEL_DIR, IMAGE_MODEL_DIR, IMAGE_HEAD_DIR,
OCR_MODEL_DIR); an environment variable of the same name overrides each one.
"""

from __future__ import annotations

import importlib.util
import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from moderation.classifiers.text_qwen3guard import Qwen3GuardClassifier
from moderation.config import Settings

REPO_ROOT = Path(__file__).resolve().parents[2]


def model_dir(field: str) -> Path:
    """A model folder from its env var or the Settings default, without needing DATABASE_URL."""
    raw = os.environ.get(field.upper(), str(Settings.model_fields[field].default))
    path = Path(raw)
    return path if path.is_absolute() else REPO_ROOT / path


TEXT_DIR = model_dir("text_model_dir")
IMAGE_DIR = model_dir("image_model_dir")
HEAD_DIR = model_dir("image_head_dir")
OCR_DIR = model_dir("ocr_model_dir")


def pick_device() -> tuple[str, str]:
    import torch

    if torch.cuda.is_available():
        return "cuda", "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    return "cpu", "fp32"


@pytest.fixture(scope="session")
def qwen() -> Iterator[Qwen3GuardClassifier]:
    for module in ("torch", "transformers"):
        if importlib.util.find_spec(module) is None:
            pytest.skip(f"{module} is not installed")
    if not (TEXT_DIR / "config.json").is_file():
        pytest.skip(f"no Qwen3Guard weights at {TEXT_DIR}")
    device, dtype = pick_device()
    clf = Qwen3GuardClassifier.load(TEXT_DIR, device, dtype)
    yield clf
    clf.close()
