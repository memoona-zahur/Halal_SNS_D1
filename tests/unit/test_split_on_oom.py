"""split_on_oom (plan §7, GPU out of memory): split the batch and retry; one item that still does not fit retries later."""

from __future__ import annotations

from collections.abc import Sequence

import pytest
import torch

from moderation.classifiers import split_on_oom
from moderation.errors import MODEL_OOM, RetryableError


class FakeModel:
    """Runs out of memory on any batch larger than `fits`."""

    def __init__(self, fits: int) -> None:
        self.fits = fits
        self.batches: list[list[int]] = []

    @split_on_oom
    def score_batch(self, items: Sequence[int]) -> list[int]:
        if len(items) > self.fits:
            raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")
        self.batches.append(list(items))
        return [item * 10 for item in items]


def test_batch_that_fits_runs_once() -> None:
    model = FakeModel(fits=8)
    assert model.score_batch([1, 2, 3]) == [10, 20, 30]
    assert model.batches == [[1, 2, 3]]


def test_oversized_batch_is_split_and_keeps_input_order() -> None:
    model = FakeModel(fits=2)
    assert model.score_batch([1, 2, 3, 4, 5]) == [10, 20, 30, 40, 50]
    assert all(len(batch) <= 2 for batch in model.batches)
    assert [item for batch in model.batches for item in batch] == [1, 2, 3, 4, 5]


def test_single_item_that_does_not_fit_retries_the_job_later() -> None:
    model = FakeModel(fits=0)
    with pytest.raises(RetryableError) as info:
        model.score_batch([1, 2])
    assert info.value.code == MODEL_OOM
    assert isinstance(info.value.__cause__, torch.OutOfMemoryError)


def test_other_errors_are_not_split() -> None:
    class Broken:
        calls = 0

        @split_on_oom
        def score_batch(self, items: Sequence[int]) -> list[int]:
            Broken.calls += 1
            raise RuntimeError("shape mismatch")

    with pytest.raises(RuntimeError, match="shape mismatch"):
        Broken().score_batch([1, 2, 3, 4])
    assert Broken.calls == 1
