"""The three models, loaded once by the worker: Qwen3Guard (text), SigLIP2 + head (images), RapidOCR (text in images)."""

from __future__ import annotations

import asyncio
import functools
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor

from moderation.errors import MODEL_OOM, RetryableError


class SingleThread:
    """One worker thread per model: blocking model calls never overlap and never block the event loop."""

    def __init__(self, name: str) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=name)

    async def run[**P, R](self, fn: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
        return await asyncio.get_running_loop().run_in_executor(self._executor, functools.partial(fn, *args, **kwargs))

    def close(self) -> None:
        self._executor.shutdown(wait=True)


def split_on_oom[S, T, R](fn: Callable[[S, Sequence[T]], list[R]]) -> Callable[[S, Sequence[T]], list[R]]:
    """Decorator for a model's sync batch method (GPU out of memory): free the cache and
    run the two halves; a single item that still does not fit retries the job later."""

    @functools.wraps(fn)
    def wrapper(self: S, items: Sequence[T]) -> list[R]:
        import torch

        try:
            return fn(self, items)
        except torch.OutOfMemoryError as exc:
            if len(items) <= 1:
                raise RetryableError(MODEL_OOM, "one item does not fit in GPU memory") from exc
            torch.cuda.empty_cache()
            half = len(items) // 2
            return wrapper(self, items[:half]) + wrapper(self, items[half:])

    return wrapper
