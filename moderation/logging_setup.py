from __future__ import annotations

import json
import logging
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from types import TracebackType
from typing import Any

BANNED_KEYS: frozenset[str] = frozenset({"text", "caption", "ocr_text"})
MAX_VALUE_LEN = 200
_RESERVED = frozenset(vars(logging.LogRecord("", 0, "", 0, "", None, None)).keys()) | {"message", "asctime"}


def redact(data: Mapping[str, Any]) -> dict[str, Any]:
    """Drop content-bearing keys and cut long strings, so user text never reaches logs
    but error details still do."""
    out: dict[str, Any] = {}
    for key, value in data.items():
        if key in BANNED_KEYS:
            continue
        if isinstance(value, str) and len(value) > MAX_VALUE_LEN:
            value = value[: MAX_VALUE_LEN - 3] + "..."
        out[key] = value
    return out


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        extras = {k: v for k, v in record.__dict__.items() if k not in _RESERVED}
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        payload.update(redact(extras))
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> None:
    """JSON logs on stdout. An exception nobody caught (e.g. a startup failure) is logged too."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)
    sys.excepthook = _log_uncaught


def _log_uncaught(kind: type[BaseException], exc: BaseException, tb: TracebackType | None) -> None:
    logging.getLogger("moderation").critical("uncaught exception", exc_info=(kind, exc, tb))
