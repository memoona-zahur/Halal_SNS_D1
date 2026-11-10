"""Head maths (softmax_v2), head loading rules and classifier wiring. No model weights needed."""

from __future__ import annotations

import io
import json
import math
from collections.abc import Sequence
from pathlib import Path

import pytest
from PIL import Image

from moderation.classifiers.image_siglip import (
    HEAD_META_FILE,
    LinearHead,
    SiglipImageClassifier,
    decode_image,
    is_safe_row,
    load_head,
)
from moderation.domain import CATEGORIES
from moderation.errors import MEDIA_UNDECODABLE, PermanentError
from moderation.prompts import IMAGE_PROMPTS

CODES = tuple(CATEGORIES)
SAFE = ("safe:food", "safe:landscape")
CLASSES = (*CODES, *SAFE)
DIM = len(CLASSES)  # one axis per row


def _one_hot(index: int, dim: int = DIM) -> tuple[float, ...]:
    return tuple(1.0 if i == index else 0.0 for i in range(dim))


def _head(scale: float = 10.0, classes: tuple[str, ...] = CLASSES) -> LinearHead:
    weight = tuple(tuple(scale * v for v in _one_hot(i, len(classes))) for i in range(len(classes)))
    return LinearHead(classes=classes, weight=weight)


def _softmax(logits: list[float]) -> list[float]:
    top = max(logits)
    exps = [math.exp(x - top) for x in logits]
    return [e / sum(exps) for e in exps]


class StubEmbedder:
    def __init__(self, vectors: Sequence[Sequence[float]], dim: int = DIM) -> None:
        self._vectors = [list(v) for v in vectors]
        self._dim = dim
        self.calls = 0

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, images: Sequence[bytes]) -> list[list[float]]:
        self.calls += 1
        return self._vectors[: len(images)]


# --- maths -------------------------------------------------------------------


def test_aligned_embedding_scores_its_category_high_and_others_low() -> None:
    [scores] = [r.categories for r in _head().score([_one_hot(1)])]  # points at G-02
    assert set(scores) == set(CODES)  # safe rows are never reported
    expected = _softmax([10.0] + [0.0] * (DIM - 1))
    assert scores["G-02"] == pytest.approx(expected[0])
    for code in CODES:
        if code != "G-02":
            assert scores[code] == pytest.approx(expected[1])


def test_one_matching_safe_row_is_enough_to_keep_categories_low() -> None:
    [result] = _head().score([_one_hot(CLASSES.index("safe:landscape"))])
    assert max(result.categories.values()) < 0.01


def test_scores_ignore_embedding_length() -> None:
    head = _head()
    [small] = [r.categories for r in head.score([[0.1 * v for v in _one_hot(3)]])]
    [large] = [r.categories for r in head.score([[50.0 * v for v in _one_hot(3)]])]
    assert small == pytest.approx(large)


def test_extreme_logits_do_not_overflow() -> None:
    [scores] = [r.categories for r in _head(scale=1e4).score([_one_hot(0)])]
    assert scores["G-01"] == pytest.approx(1.0)
    assert all(0.0 <= v <= 1.0 and math.isfinite(v) for v in scores.values())


@pytest.mark.parametrize("bad", [[math.nan] * DIM, [math.inf] + [0.0] * (DIM - 1)])
def test_broken_embedding_gives_nan_scores_for_the_policy_to_hold(bad: list[float]) -> None:
    [scores] = [r.categories for r in _head().score([bad])]
    assert all(math.isnan(v) for v in scores.values())


# --- row naming and head validation -------------------------------------------


@pytest.mark.parametrize("name", ["safe:food", "safe:text_message", "safe:a1"])
def test_valid_safe_row_names(name: str) -> None:
    assert is_safe_row(name)


@pytest.mark.parametrize("name", ["safe", "safe:", "safe:Food", "safe:two words", "food", "G-08"])
def test_invalid_safe_row_names(name: str) -> None:
    assert not is_safe_row(name)


def test_head_without_safe_rows_is_rejected() -> None:
    with pytest.raises(ValueError, match="no safe"):
        _head(classes=CODES)


def test_head_missing_a_category_is_rejected() -> None:
    with pytest.raises(ValueError, match="missing"):
        _head(classes=(*CODES[:-1], *SAFE))


@pytest.mark.parametrize("scoring", ["softmax_v1", None])
def test_loader_rejects_heads_that_are_not_softmax_v2(tmp_path: Path, scoring: str | None) -> None:
    meta: dict[str, object] = {"classes": [*CODES, "safe:food"]}
    if scoring is not None:
        meta["scoring"] = scoring
    (tmp_path / HEAD_META_FILE).write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError, match="softmax_v2"):
        load_head(tmp_path)  # rejected before the weights are read


# --- decoding ------------------------------------------------------------------


def test_undecodable_bytes_are_a_permanent_error() -> None:
    with pytest.raises(PermanentError) as info:
        decode_image(b"\xff\xd8\xff not really a jpeg")
    assert info.value.code == MEDIA_UNDECODABLE


def test_decode_returns_rgb() -> None:
    buffer = io.BytesIO()
    Image.new("L", (8, 8), 200).save(buffer, "PNG")
    assert decode_image(buffer.getvalue()).mode == "RGB"


# --- classifier wiring ---------------------------------------------------------


def test_classifier_scores_each_image() -> None:
    clf = SiglipImageClassifier(StubEmbedder([_one_hot(0), _one_hot(6)]), _head())
    try:
        first, second = clf.score_batch([b"img-a", b"img-b"])
    finally:
        clf.close()
    assert set(first.categories) == set(CODES)
    assert max(first.categories, key=first.categories.__getitem__) == "G-01"
    assert max(second.categories, key=second.categories.__getitem__) == "G-07"


def test_classifier_returns_the_normalised_embedding_it_scored() -> None:
    raw = [3.0 * v for v in _one_hot(2)]
    raw[5] = 4.0  # length 5 before normalising
    clf = SiglipImageClassifier(StubEmbedder([raw]), _head())
    try:
        [result] = clf.score_batch([b"img"])
    finally:
        clf.close()
    assert len(result.embedding) == DIM
    assert math.isclose(math.fsum(x * x for x in result.embedding), 1.0)
    assert result.embedding[2] == pytest.approx(0.6) and result.embedding[5] == pytest.approx(0.8)


def test_classifier_rejects_dim_mismatch() -> None:
    with pytest.raises(ValueError, match="dim"):
        SiglipImageClassifier(StubEmbedder([], dim=DIM + 1), _head())


async def test_async_score_matches_sync() -> None:
    clf = SiglipImageClassifier(StubEmbedder([_one_hot(2)]), _head())
    try:
        assert await clf.score([b"x"]) == clf.score_batch([b"x"])
    finally:
        clf.close()


# --- prompts -------------------------------------------------------------------


def test_prompts_give_categories_then_safe_rows() -> None:
    rows = tuple(IMAGE_PROMPTS)
    assert rows[: len(CODES)] == CODES
    safe_rows = rows[len(CODES) :]
    assert len(safe_rows) >= 2 and all(is_safe_row(r) for r in safe_rows)
    assert "safe:food" in safe_rows and "safe:landscape" in safe_rows
    assert all(prompts and all(p.strip() for p in prompts) for prompts in IMAGE_PROMPTS.values())
