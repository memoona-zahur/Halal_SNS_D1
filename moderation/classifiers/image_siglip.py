"""SigLIP2 image classifier: vision tower + linear head.

At runtime only the vision tower and the head file are loaded. The text tower is used once,
offline, by ``model_download.py build-head`` to build the head.

Scoring rule ``softmax_v2`` (recorded in head.json)::

    logit[r] = weight[r] . normalize(image_embedding)      for every row r
    p        = softmax(logit) over all rows: G-01..G-07 and every ``safe:<slug>`` row

The 7 G-code probabilities are the per-category scores. ``safe:*`` rows are never
reported. Each safe prompt has its own row (food, landscape, ...), so a clean image only has
to match one of them. The head has no bias: the zero-shot bias is the same for every row, so it
cancels inside the softmax.

A broken embedding (NaN or inf) gives NaN scores, which the policy holds.
"""

from __future__ import annotations

import io
import json
import math
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

from PIL import Image, ImageOps

from moderation.classifiers import SingleThread, split_on_oom
from moderation.domain import CATEGORIES, ImageScore
from moderation.errors import MEDIA_UNDECODABLE, PermanentError

HEAD_WEIGHTS_FILE: Final = "head.safetensors"
HEAD_META_FILE: Final = "head.json"
HEAD_SCORING: Final = "softmax_v2"
# Non-harm rows. The softmax needs at least one, or a clean image is forced onto a category.
SAFE_PREFIX: Final = "safe:"
_SAFE_SLUG: Final = re.compile(r"[a-z0-9_]+")
WARMUP_IMAGE_SIZE: Final = 64

def is_safe_row(name: str) -> bool:
    """True for a non-reported row name: ``safe:`` followed by a lowercase slug."""
    return name.startswith(SAFE_PREFIX) and _SAFE_SLUG.fullmatch(name[len(SAFE_PREFIX) :]) is not None


def _softmax(logits: Sequence[float]) -> list[float]:
    top = max(logits)  # subtract the max so exp never overflows
    exps = [math.exp(x - top) for x in logits]
    total = math.fsum(exps)
    return [e / total for e in exps]


MAX_PIXELS = 25_000_000  # a small file can hold a huge bitmap; refuse before it is decoded


def decode_image(data: bytes) -> Image.Image:
    """Bytes -> upright RGB image. Undecodable bytes fail the job (media_undecodable), never retry."""
    try:
        with Image.open(io.BytesIO(data)) as img:
            if img.width * img.height > MAX_PIXELS:
                raise ValueError(f"image has {img.width * img.height} pixels, the limit is {MAX_PIXELS}")
            img.load()
            return ImageOps.exif_transpose(img).convert("RGB")
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        raise PermanentError(MEDIA_UNDECODABLE, f"{type(exc).__name__}: {exc}") from exc


class LinearHead:
    """Softmax over all rows of an L2-normalised embedding. Pure Python: 17 x 768 is cheap."""

    def __init__(self, classes: Sequence[str], weight: Sequence[Sequence[float]]) -> None:
        missing = [code for code in CATEGORIES if code not in classes]
        if missing:
            raise ValueError(f"head is missing categories {missing}; rebuild it with model_download.py build-head")
        self.classes = tuple(classes)
        self.weight = tuple(tuple(row) for row in weight)
        self._category_rows = [self.classes.index(code) for code in CATEGORIES]
        if not any(is_safe_row(name) for name in self.classes):
            raise ValueError(f"head has no {SAFE_PREFIX}<slug> rows; rebuild it with model_download.py build-head")

    @property
    def dim(self) -> int:
        return len(self.weight[0])

    def score(self, embeddings: Sequence[Sequence[float]]) -> list[ImageScore]:
        results: list[ImageScore] = []
        for embedding in embeddings:
            norm = math.sqrt(sum(x * x for x in embedding))
            unit = tuple(x / norm for x in embedding)
            probs = _softmax([sum(w * x for w, x in zip(row, unit, strict=True)) for row in self.weight])
            categories = {code: probs[i] for code, i in zip(CATEGORIES, self._category_rows, strict=True)}
            results.append(ImageScore(categories, unit))
        return results


def load_head(head_dir: Path) -> LinearHead:
    meta = json.loads((head_dir / HEAD_META_FILE).read_text(encoding="utf-8"))
    if meta.get("scoring") != HEAD_SCORING:
        raise ValueError(
            f"{head_dir / HEAD_META_FILE}: scoring {meta.get('scoring')!r} is not {HEAD_SCORING!r}; rebuild the head"
        )
    from safetensors.torch import load_file

    weight: list[list[float]] = load_file(str(head_dir / HEAD_WEIGHTS_FILE))["weight"].float().tolist()
    return LinearHead(meta["classes"], weight)


class SiglipVisionEmbedder:
    """SigLIP / SigLIP2 vision tower only. Loads from a local folder, never from the network."""

    def __init__(self, model_dir: Path, device: str, dtype: str) -> None:
        import torch
        import transformers

        self._torch = torch
        self._device = device
        self._torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]
        self._processor = transformers.AutoImageProcessor.from_pretrained(  # type: ignore[no-untyped-call]
            str(model_dir), local_files_only=True
        )
        model: Any = transformers.SiglipVisionModel.from_pretrained(
            str(model_dir), local_files_only=True, dtype=self._torch_dtype
        )
        self._model = model.to(device).eval()
        self._dim = int(model.config.hidden_size)

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, images: Sequence[bytes]) -> list[list[float]]:
        inputs = self._processor(images=[decode_image(data) for data in images], return_tensors="pt")
        batch: dict[str, Any] = {}
        for key, value in inputs.items():
            moved = value.to(self._device)
            batch[key] = moved.to(self._torch_dtype) if moved.is_floating_point() else moved
        with self._torch.inference_mode():
            outputs = self._model(**batch)
        embeddings: list[list[float]] = outputs.pooler_output.float().cpu().tolist()
        return embeddings


class SiglipImageClassifier:
    """Image classifier: embedder + head. Build once with `load()`; every job reuses it.

    ``score_batch`` is synchronous; ``score`` runs it on this model's own single thread,
    so calls to the model never overlap.
    """

    def __init__(self, embedder: SiglipVisionEmbedder, head: LinearHead) -> None:
        if embedder.dim != head.dim:
            raise ValueError(f"embedder dim {embedder.dim} != head dim {head.dim}; rebuild the head")
        self._embedder = embedder
        self._head = head
        self._thread = SingleThread("image-model")

    @classmethod
    def load(cls, model_dir: Path, head_dir: Path, device: str, dtype: str) -> SiglipImageClassifier:
        """Read the head first, so a stale head fails before the weights are loaded."""
        head = load_head(head_dir)
        return cls(SiglipVisionEmbedder(model_dir, device, dtype), head)

    @split_on_oom
    def score_batch(self, images: Sequence[bytes]) -> list[ImageScore]:
        """Category scores and the normalised embedding per image (the embedding is saved for D5)."""
        return self._head.score(self._embedder.embed(images))

    async def score(self, images: Sequence[bytes]) -> list[ImageScore]:
        return await self._thread.run(self.score_batch, list(images))

    def warmup(self) -> None:
        buffer = io.BytesIO()
        Image.new("RGB", (WARMUP_IMAGE_SIZE, WARMUP_IMAGE_SIZE), (128, 128, 128)).save(buffer, "PNG")
        self.score_batch([buffer.getvalue()])

    def close(self) -> None:
        self._thread.close()
