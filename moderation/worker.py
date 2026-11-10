"""Worker: loads the models once, then runs the job loops, the callback sender and lease requeue.

    python -m moderation.worker

A new job wakes the job loops through a Postgres notification. A poller also wakes them each
POLL_S, which picks up retries that came due, requeued jobs and anything sent while the
listener was reconnecting; it sends due callbacks and requeues expired leases too.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import shutil
import signal
import sys
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import httpx

from moderation.config import MODEL_PINS, DeviceRequest, Settings, get_settings
from moderation.domain import CATEGORIES, UNCLASSIFIED, Decision
from moderation.errors import INTERNAL_ERROR, PermanentError, RetryableError
from moderation.fetch import ImageFetcher
from moderation.logging_setup import configure_logging
from moderation.moderate import Moderator
from moderation.policy import D1_POLICY, worst_item
from moderation.store import ClaimedJob, OutboxItem, Store

if TYPE_CHECKING:
    from moderation.classifiers.image_siglip import SiglipImageClassifier
    from moderation.classifiers.ocr_rapidocr import RapidOcrEngine
    from moderation.classifiers.text_qwen3guard import Qwen3GuardClassifier

log = logging.getLogger("moderation.worker")

ARCHIVE_SUBDIR = "by_category"  # under IMAGE_ALLOWED_ROOT: by_category/<label>/<category>/<file>


def _category_folder_name(category: str | None) -> str:
    """'G-01' -> 'G-01_Violence_and_gore', so the folder name is readable without a lookup table."""
    if category is None:
        return UNCLASSIFIED
    name = CATEGORIES.get(category)
    if name is None:
        return category
    return f"{category}_{re.sub(r'[^A-Za-z0-9]+', '_', name).strip('_')}"

JOB_BACKOFF_S: tuple[float, ...] = (5.0, 15.0, 60.0, 300.0)
CALLBACK_BACKOFF_S: tuple[float, ...] = (1.0, 5.0, 30.0, 120.0, 600.0)
OUTBOX_BATCH = 10
WARMUP_ROUNDS = 3
EXIT_GPU_CRASH = 70  # non-zero, so compose restarts the worker


def backoff_delay(steps: tuple[float, ...], attempt: int) -> float:
    """Delay after `attempt` tries (1-based); the last step repeats."""
    return steps[min(attempt, len(steps)) - 1]


def is_gpu_crash(exc: BaseException) -> bool:
    """A broken CUDA context, which only a new process recovers from. Out of memory is not one:
    split_on_oom handles it inside the model call."""
    torch = sys.modules.get("torch")  # loaded with the models; never imported just for this check
    if torch is not None:
        if isinstance(exc, torch.OutOfMemoryError):
            return False
        if isinstance(exc, torch.AcceleratorError):
            return True
    return isinstance(exc, RuntimeError) and "CUDA error" in str(exc)


class Worker:
    def __init__(
        self,
        cfg: Settings,
        store: Store,
        moderator: Moderator,
        callbacks: httpx.AsyncClient,
    ) -> None:
        self._cfg = cfg
        self._store = store
        self._moderator = moderator
        self._callbacks = callbacks
        self._wake = asyncio.Event()
        self._stop = asyncio.Event()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()

    async def run(self) -> None:
        jobs = [asyncio.create_task(self._job_loop(), name=f"job-{i}") for i in range(self._cfg.fast_workers)]
        background = [
            asyncio.create_task(self.run_periodically(self._cfg.poll_s, step), name=step.__name__)
            for step in (self.listen_for_jobs, self._requeue_expired, self.send_due_callbacks, self._wake_jobs)
        ]
        log.info("worker started", extra={"job_loops": len(jobs)})
        await self._stop.wait()
        log.info("shutdown requested; finishing running jobs")
        for task in background:
            task.cancel()  # the listener blocks on its socket until cancelled
        await asyncio.gather(*jobs, *background, return_exceptions=True)

    # --- jobs ---------------------------------------------------------------

    async def _job_loop(self) -> None:
        while not self._stop.is_set():
            await self._wake.wait()
            self._wake.clear()
            try:
                while not self._stop.is_set() and (job := await self._store.claim_job()) is not None:
                    await self.process(job)
            except Exception:
                log.exception("job loop failed")

    async def process(self, job: ClaimedJob) -> None:
        """Check one job, save the decision, post the callback.
        A PermanentError fails the job; any other error retries it, or fails it out of attempts."""
        try:
            renew = asyncio.create_task(self._renew_lease(job))
            try:
                decision = await self._moderator.moderate(job.request)
            finally:
                renew.cancel()
            saved = await self._store.save_decision(job, decision)
        except PermanentError as exc:
            await self._fail(job, exc.code)
            return
        except Exception as exc:  # noqa: BLE001 - _retry logs it, then retries or fails the job
            await self._retry(job, exc)
            return
        if saved is None:
            log.warning("lease lost; decision dropped", extra={"job_id": str(job.job_id)})
            return
        await asyncio.to_thread(self._archive_uploaded_images, job, decision)
        log.info(
            "job decided",
            extra={"job_id": str(job.job_id), "label": decision.label.value, "latency_ms": saved.latency_ms},
        )
        if saved.callback is not None:
            await self.deliver_callback(saved.callback)

    def _archive_uploaded_images(self, job: ClaimedJob, decision: Decision) -> None:
        """Keeps a copy of every raw-uploaded image (POST /v1/moderations/uploads), sorted by its own
        decided label and category. The original stays in place too, so the same upload can still be
        referenced by another request. Images fetched from an external URL are not copied here."""
        root = self._cfg.image_allowed_root.strip()
        if not root:
            return
        base = Path(root).resolve()
        for image in job.request.images:
            if image.url.startswith(("http://", "https://")) or not image.url.startswith("uploads/"):
                continue
            source = (base / image.url).resolve()
            if not source.is_relative_to(base) or not source.is_file():
                continue
            per_image = [item for item in decision.items if item.media_id == image.media_id]
            worst = worst_item(per_image) if per_image else None
            label = worst.label.value if worst is not None else "unknown"
            category_folder = _category_folder_name(worst.category if worst is not None else None)
            dest_dir = base / ARCHIVE_SUBDIR / label / category_folder
            try:
                dest_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, dest_dir / source.name)
            except OSError:
                log.exception(
                    "archiving uploaded image failed",
                    extra={"job_id": str(job.job_id), "media_id": image.media_id},
                )

    async def _renew_lease(self, job: ClaimedJob) -> None:
        """Keep the lease while moderation runs, so a long job is not reclaimed and run twice."""
        while True:
            await asyncio.sleep(self._cfg.lease_s / 3)
            try:
                if not await self._store.extend_lease(job):
                    log.warning("lease lost while running", extra={"job_id": str(job.job_id)})
                    return
            except Exception:
                log.exception("lease renewal failed", extra={"job_id": str(job.job_id)})

    async def _retry(self, job: ClaimedJob, exc: Exception) -> None:
        extra = {"job_id": str(job.job_id), "attempt": job.attempts, "detail": str(exc)}
        if is_gpu_crash(exc):
            # Only a new process recovers a broken CUDA context. Compose restarts the
            # worker (restart: unless-stopped); the job's lease runs out and it goes back to the queue.
            log.critical("GPU failure; exiting so the worker restarts", extra=extra, exc_info=exc)
            os._exit(EXIT_GPU_CRASH)
        if isinstance(exc, RetryableError):
            log.warning("job will retry", extra=extra)
        else:
            log.error("job error", extra=extra, exc_info=exc)
        code = exc.code if isinstance(exc, RetryableError) else INTERNAL_ERROR
        if job.attempts >= self._cfg.max_attempts:
            await self._fail(job, code)
        elif not await self._store.requeue_job(job, code, backoff_delay(JOB_BACKOFF_S, job.attempts)):
            log.warning("lease lost; retry not recorded", extra={"job_id": str(job.job_id)})

    async def _fail(self, job: ClaimedJob, code: str) -> None:
        saved = await self._store.fail_job(job, code)
        if saved is None:
            log.warning("lease lost; failure not recorded", extra={"job_id": str(job.job_id)})
            return
        log.info("job failed", extra={"job_id": str(job.job_id), "error_code": code})
        if saved.callback is not None:
            await self.deliver_callback(saved.callback)

    # --- callbacks ----------------------------------------------------------

    async def deliver_callback(self, item: OutboxItem) -> None:
        """POST one callback. Any 2xx counts. Otherwise retry with backoff, dead after the limit.
        One log line per failed try: `status` is the HTTP status, or null when the backend was unreachable."""
        try:
            status = await self.post_callback(item)
            extra = {"outbox_id": item.outbox_id, "attempt": item.attempts, "status": status}
            if status is not None and 200 <= status < 300:
                await self._store.mark_outbox_sent(item.outbox_id)
            elif item.attempts >= self._cfg.callback_max_attempts:
                await self._store.mark_outbox_dead(item.outbox_id)
                log.error("callback gave up", extra=extra)
            else:
                await self._store.mark_outbox_retry(item.outbox_id, backoff_delay(CALLBACK_BACKOFF_S, item.attempts))
                log.warning("callback will retry", extra=extra)
        except Exception:
            log.exception("deliver_callback failed", extra={"outbox_id": item.outbox_id})

    async def post_callback(self, item: OutboxItem) -> int | None:
        """HTTP status of the callback POST; None when the backend could not be reached."""
        with contextlib.suppress(httpx.HTTPError, TimeoutError):
            # The timeout is the budget for the whole request; the response body is never read.
            async with (
                asyncio.timeout(self._cfg.callback_timeout_s),
                self._callbacks.stream("POST", item.callback_url, json=item.body) as response,
            ):
                return response.status_code
        return None

    async def send_due_callbacks(self) -> None:
        items = await self._store.claim_outbox(OUTBOX_BATCH)
        await asyncio.gather(*(self.deliver_callback(item) for item in items))

    # --- wake-ups and housekeeping -----------------------------------------

    async def listen_for_jobs(self) -> None:
        async for _ in self._store.listen_for_jobs():
            self._wake.set()

    async def _wake_jobs(self) -> None:
        self._wake.set()

    async def _requeue_expired(self) -> None:
        if count := await self._store.requeue_expired_jobs(self._cfg.max_attempts):
            log.warning("requeued expired jobs", extra={"count": count})

    async def run_periodically(self, seconds: float, step: Callable[[], Awaitable[None]]) -> None:
        """Run `step` now and then every `seconds` until shutdown. A failing step is logged and tried again."""
        while not self._stop.is_set():
            try:
                await step()
            except Exception:
                log.exception("%s failed", step.__qualname__)
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._stop.wait(), timeout=seconds)


# --- start-up ---------------------------------------------------------------


def _cuda_works() -> bool:
    """True only if torch sees a GPU and can allocate on it."""
    import torch

    if not torch.cuda.is_available():
        return False
    with contextlib.suppress(RuntimeError):
        torch.zeros(1, device="cuda")
        return True
    return False


def pick_device(requested: DeviceRequest, cuda_works: bool) -> Literal["cuda", "cpu"]:
    """auto = GPU if it works, else CPU; cuda = GPU or refuse to start; cpu = CPU."""
    if requested == "cpu":
        return "cpu"
    if cuda_works:
        return "cuda"
    if requested == "cuda":
        raise RuntimeError("DEVICE=cuda but no usable CUDA device")
    return "cpu"


def pick_dtype(device: str) -> Literal["bf16", "fp16", "fp32"]:
    if device == "cpu":
        return "fp32"
    import torch

    return "bf16" if torch.cuda.is_bf16_supported() else "fp16"


def load_models(cfg: Settings) -> tuple[Qwen3GuardClassifier, SiglipImageClassifier, RapidOcrEngine]:
    """Load the three models once and warm them up. Blocking: run it in a thread."""
    # Imported here: the API process and unit tests never load the model runtime.
    import torch
    import transformers

    from moderation.classifiers.image_siglip import SiglipImageClassifier
    from moderation.classifiers.ocr_rapidocr import RapidOcrEngine
    from moderation.classifiers.text_qwen3guard import Qwen3GuardClassifier

    transformers.logging.set_verbosity_error()  # type: ignore[no-untyped-call]
    cuda = _cuda_works()
    device = pick_device(cfg.device, cuda)
    if not cuda and cfg.device == "auto":
        log.warning("no usable GPU; running the models on CPU (slower)")
    if device == "cpu":
        torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))

    started = time.perf_counter()
    text = Qwen3GuardClassifier.load(
        Path(cfg.text_model_dir), device, pick_dtype(device), max_batch=cfg.model_max_batch
    )
    image = SiglipImageClassifier.load(
        Path(cfg.image_model_dir), Path(cfg.image_head_dir), device, pick_dtype(device)
    )
    ocr = RapidOcrEngine(Path(cfg.ocr_model_dir), threads=cfg.ocr_threads)
    loaded_s = time.perf_counter() - started
    # The GPU idles at a low clock; a few rounds bring it up so the first real job is not slow.
    for _ in range(WARMUP_ROUNDS):
        text.score_batch(["warm-up"])
        image.warmup()
    log.info(
        "models ready",
        extra={
            "device": device,
            "load_s": round(loaded_s, 1),
            "warmup_s": round(time.perf_counter() - started - loaded_s, 1),
            "gpu_mib": round(torch.cuda.memory_allocated() / 2**20) if cuda else 0,
        },
    )
    return text, image, ocr


def read_model_versions(cfg: Settings) -> dict[str, str]:
    """What produced each decision: the pinned revisions, the head build, the OCR files.
    Read once at start and saved with every decision, for D3 calibration and appeals."""
    from moderation.classifiers.image_siglip import HEAD_META_FILE
    from moderation.classifiers.ocr_rapidocr import ocr_model_paths
    from moderation.prompts import PROMPT_VERSION

    head = json.loads((Path(cfg.image_head_dir) / HEAD_META_FILE).read_text(encoding="utf-8"))
    det, rec = ocr_model_paths(Path(cfg.ocr_model_dir))
    return {
        "text": "{}@{}+{}".format(*MODEL_PINS["text"], PROMPT_VERSION),
        "image": "{}@{}".format(*MODEL_PINS["image"]),
        "image_head": f"{head['kind']}@{head['created_at']}",
        "ocr": f"{det.name}+{rec.name}",
    }


async def serve(cfg: Settings) -> None:
    async with contextlib.AsyncExitStack() as stack:
        store = Store(cfg)
        await store.open()
        stack.push_async_callback(store.close)
        text, image, ocr = await asyncio.to_thread(load_models, cfg)
        for model in (text, image, ocr):
            stack.callback(model.close)
        fetch_client = await stack.enter_async_context(
            httpx.AsyncClient(timeout=cfg.fetch_timeout_s, follow_redirects=False, trust_env=False)
        )
        callback_client = await stack.enter_async_context(httpx.AsyncClient(timeout=cfg.callback_timeout_s, trust_env=False))
        moderator = Moderator(
            text=text,
            image=image,
            ocr=ocr,
            fetcher=ImageFetcher(cfg, fetch_client),
            policy=D1_POLICY,
            model_versions=read_model_versions(cfg),
        )
        worker = Worker(cfg, store, moderator, callback_client)
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            # Windows has no add_signal_handler; Ctrl+C cancels asyncio.run there and the stack still closes.
            with contextlib.suppress(NotImplementedError):
                loop.add_signal_handler(sig, worker.stop)
        await worker.run()


def main() -> None:
    cfg = get_settings()
    configure_logging(cfg.log_level)
    # psycopg's async driver cannot use the Proactor loop that Windows defaults to.
    loop_factory = asyncio.SelectorEventLoop if sys.platform == "win32" else None
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(serve(cfg), loop_factory=loop_factory)


if __name__ == "__main__":
    main()
