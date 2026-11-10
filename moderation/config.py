from __future__ import annotations

from functools import lru_cache
from typing import Final, Literal, NamedTuple

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

DeviceRequest = Literal["auto", "cuda", "cpu"]


class ModelPin(NamedTuple):
    """A Hugging Face repo at one commit. The worker only loads from disk; model_download.py downloads."""

    repo: str
    revision: str


# Pinned model revisions. Change one, run `python model_download.py pull`, restart the worker.
MODEL_PINS: Final = {
    "text": ModelPin("Qwen/Qwen3Guard-Gen-0.6B", "fada3b2f655b89601929198343c94cd2f64d93cc"),
    "image": ModelPin("google/siglip2-base-patch16-384", "f775b65a79762255128c981547af89addcfe0f88"),
}


class Settings(BaseSettings):
    """Every setting, read from the environment or .env. The only settings class."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Database ---
    database_url: str = Field(description="Postgres DSN, psycopg format. Required: there is no default")
    db_pool_max_size: int = Field(default=10, ge=1, description="Maximum open DB connections per process")

    # --- Models (worker only) ---
    text_model_dir: str = Field(default="models/qwen3guard-gen-0.6b", description="Qwen3Guard weights")
    image_model_dir: str = Field(default="models/siglip2-base-patch16-384", description="SigLIP2 weights")
    image_head_dir: str = Field(default="models/siglip2-head", description="Image head built by model_download.py build-head")
    ocr_model_dir: str = Field(default="models/rapidocr", description="RapidOCR PP-OCRv5 det + rec files")
    device: DeviceRequest = Field(
        default="auto",
        description="auto: GPU if usable, else CPU (degraded). cuda: GPU or refuse to start. cpu: CPU only",
    )
    ocr_threads: int = Field(default=2, ge=1, description="ONNX Runtime threads for OCR (always CPU)")
    model_max_batch: int = Field(default=8, ge=1, description="Largest batch one text-model call takes")

    # --- Worker ---
    fast_workers: int = Field(default=4, ge=1, description="Concurrent jobs")
    poll_s: float = Field(
        default=5.0,
        gt=0,
        description="Fallback poll. New jobs wake the worker by notification; this picks up retries",
    )
    lease_s: int = Field(default=20, ge=5, description="Seconds a claimed job stays with its worker")
    max_attempts: int = Field(default=5, ge=1, description="Tries before a job fails")

    # --- Callbacks ---
    callback_timeout_s: float = Field(default=5.0, gt=0, description="HTTP timeout for one callback")
    callback_max_attempts: int = Field(default=10, ge=1, description="Callback tries before the row is dead")

    # --- Images ---
    image_allowed_hosts: str = Field(default="", description="Comma-separated hostnames. Empty refuses every URL")
    allow_http_localhost: bool = Field(default=False, description="Allow plain http:// to localhost, 127.0.0.1 and host.docker.internal (dev only, local testing)")
    image_allowed_root: str = Field(
        default="",
        description="Folder local image paths and raw uploads (POST /v1/moderations/uploads) may come from. "
        "Empty refuses local paths and disables uploads. In Docker this must be /app/uploads, the volume "
        "docker-compose shares between api and worker",
    )
    image_max_bytes: int = Field(default=10 * 1024 * 1024, ge=1024, description="Largest image accepted")
    fetch_timeout_s: float = Field(default=5.0, gt=0, description="HTTP timeout for one image download")

    # --- API ---
    api_max_body_bytes: int = Field(default=128 * 1024, ge=1024, description="Largest request body accepted, checked before it is read")
    api_max_images: int = Field(default=10, ge=0, description="Maximum images per request")
    api_max_text_chars: int = Field(default=10_000, ge=1, description="Maximum caption length")
    api_max_url_chars: int = Field(default=2048, ge=1, description="Maximum image URL length")
    api_max_callback_url_chars: int = Field(
        default=500, ge=1, le=2048, description="Maximum callback URL length (the database column allows 2048)"
    )
    api_wait_s: float = Field(
        default=10.0, ge=0, description="How long POST /v1/moderations waits for the decision before answering 202. 0: never wait"
    )
    api_wait_poll_s: float = Field(default=0.1, gt=0, description="How often a waiting request checks its job")
    api_keys: str = Field(
        default="",
        description="Comma-separated name:key pairs for the X-API-Key header, e.g. 'alice:abc123,bob:def456'. "
        "Empty disables auth entirely (dev only; never run exposed to the internet this way)",
    )

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO", description="Root log level")

    @property
    def allowed_hosts(self) -> frozenset[str]:
        return frozenset(h.strip().lower() for h in self.image_allowed_hosts.split(",") if h.strip())

    @property
    def api_key_names(self) -> dict[str, str]:
        """Key -> teammate name, parsed from `api_keys`. Empty means auth is disabled."""
        result = {}
        for pair in self.api_keys.split(","):
            if ":" in (pair := pair.strip()):
                name, key = pair.split(":", 1)
                result[key] = name
        return result


@lru_cache
def get_settings() -> Settings:
    return Settings()
