"""Offline model tools, run from the project root: download the pinned models and build the image head.

    python model_download.py pull              # download everything, then build the head
    python model_download.py pull --skip-head  # download only
    python model_download.py build-head        # rebuild the head after editing IMAGE_PROMPTS

The head is a softmax_v2 head: rows are the 7 G-codes then one "safe:<slug>" row per safe prompt;
weight[r] = logit_scale * L2-normalised mean prompt embedding. See moderation/classifiers/image_siglip.py.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

ROOT: Final = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))  # so `moderation` imports when run as a script

from huggingface_hub import snapshot_download

from moderation.classifiers.image_siglip import (
    HEAD_META_FILE,
    HEAD_SCORING,
    HEAD_WEIGHTS_FILE,
)
from moderation.config import MODEL_PINS, get_settings
from moderation.logging_setup import configure_logging
from moderation.prompts import IMAGE_PROMPTS

log = logging.getLogger("model_download")

# SigLIP text towers are trained with padding="max_length", max_length=64.
TEXT_MAX_LENGTH: Final = 64


def _resolve(path: str) -> Path:
    """A configured folder, relative to the project root unless absolute."""
    return ROOT / path


def model_revision(model_dir: Path) -> str:
    """Commit hash recorded by huggingface_hub when it downloaded into model_dir, if any."""
    meta = model_dir / ".cache" / "huggingface" / "download" / "config.json.metadata"
    try:
        first_line = meta.read_text(encoding="utf-8").splitlines()[0].strip()
    except (OSError, IndexError):
        return "unknown"
    return first_line or "unknown"


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def build_head(model_dir: Path, head_dir: Path, device: str) -> Path:
    import torch
    import transformers
    from safetensors.torch import save_file

    model = transformers.AutoModel.from_pretrained(str(model_dir), local_files_only=True).to(device).eval()
    processor = transformers.AutoProcessor.from_pretrained(  # type: ignore[no-untyped-call]
        str(model_dir), local_files_only=True
    )

    class_rows: list[Any] = []
    with torch.inference_mode():
        for row_prompts in IMAGE_PROMPTS.values():
            inputs = processor(
                text=[p.strip().lower() for p in row_prompts],
                padding="max_length",
                max_length=TEXT_MAX_LENGTH,
                truncation=True,
                return_tensors="pt",
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            outputs = model.get_text_features(**inputs)
            # A tensor in transformers 4.x; an output object in later versions.
            features = getattr(outputs, "pooler_output", outputs).float()
            features = torch.nn.functional.normalize(features, dim=-1)
            mean = features.mean(dim=0)
            class_rows.append(mean / mean.norm())
        logit_scale = float(model.logit_scale.exp().item())

    weight = (torch.stack(class_rows) * logit_scale).float().cpu().contiguous()
    rows = list(IMAGE_PROMPTS)

    meta: dict[str, Any] = {
        "kind": "zero-shot",
        "classes": rows,
        "embedding_dim": int(weight.shape[1]),
        "scoring": HEAD_SCORING,
        "logit_scale": logit_scale,
        "prompt_file": "moderation/prompts.py:IMAGE_PROMPTS",
        "model_dir": model_dir.name,
        "model_revision": model_revision(model_dir),
        "created_at": datetime.now(UTC).isoformat(),
    }

    head_dir.mkdir(parents=True, exist_ok=True)
    weights_tmp = head_dir / f"{HEAD_WEIGHTS_FILE}.tmp"
    meta_tmp = head_dir / f"{HEAD_META_FILE}.tmp"
    save_file({"weight": weight}, str(weights_tmp))
    meta_tmp.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    # Replace both only after both are written, so a crash never leaves a half-new head.
    os.replace(weights_tmp, head_dir / HEAD_WEIGHTS_FILE)
    os.replace(meta_tmp, head_dir / HEAD_META_FILE)
    return head_dir


def pull_ocr_models(model_dir: Path) -> list[Path]:
    """Download the OCR det + rec models once. Files already present are kept."""
    from rapidocr.utils.download_file import DownloadFile, DownloadFileInput

    from moderation.classifiers.ocr_rapidocr import ocr_model_paths, ocr_model_urls

    paths = ocr_model_paths(model_dir)
    for url, path in zip(ocr_model_urls(), paths, strict=True):
        DownloadFile.run(DownloadFileInput(file_url=url, save_path=path, logger=log))
    return list(paths)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    pull = commands.add_parser("pull", help="download the models, then build the image head")
    pull.add_argument("--skip-head", action="store_true", help="do not build the image head")
    head = commands.add_parser("build-head", help="rebuild the image head from IMAGE_PROMPTS")
    for command in (pull, head):
        command.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args(argv)
    configure_logging("INFO")
    cfg = get_settings()
    image_dir, head_dir = _resolve(cfg.image_model_dir), _resolve(cfg.image_head_dir)

    if args.command == "pull":
        for key, target in (("text", _resolve(cfg.text_model_dir)), ("image", image_dir)):
            pin = MODEL_PINS[key]
            log.info("downloading", extra={"model": key, "repo": pin.repo, "revision": pin.revision, "dir": str(target)})
            snapshot_download(repo_id=pin.repo, revision=pin.revision, local_dir=target)
        pulled = pull_ocr_models(_resolve(cfg.ocr_model_dir))
        log.info("pulled", extra={"model": "rapidocr", "files": len(pulled), "dir": cfg.ocr_model_dir})
        if args.skip_head:
            return 0
    out = build_head(image_dir, head_dir, resolve_device(args.device))
    log.info("image head built", extra={"dir": str(out)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
