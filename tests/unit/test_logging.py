from __future__ import annotations

import json
import logging

from moderation.logging_setup import MAX_VALUE_LEN, JsonFormatter, redact


def test_redact_drops_banned_keys() -> None:
    data = {"text": "hi", "caption": "c", "ocr_text": "o", "line": "fast"}
    assert redact(data) == {"line": "fast"}


def test_redact_cuts_long_string_values_instead_of_dropping_them() -> None:
    data = {"short": "ok", "long": "x" * (MAX_VALUE_LEN + 50), "num": 5}
    out = redact(data)
    assert out["short"] == "ok" and out["num"] == 5
    assert len(out["long"]) == MAX_VALUE_LEN and out["long"].endswith("...")


def test_redact_keeps_value_at_limit() -> None:
    data = {"edge": "x" * MAX_VALUE_LEN}
    assert redact(data) == data


def test_json_formatter_emits_parseable_line_without_banned_fields() -> None:
    record = logging.LogRecord("moderation.test", logging.INFO, __file__, 1, "job done", None, None)
    record.job_id = "j1"
    record.text = "secret caption"
    record.line = "fast"
    line = JsonFormatter().format(record)
    parsed = json.loads(line)
    assert parsed["msg"] == "job done"
    assert parsed["level"] == "INFO"
    assert parsed["logger"] == "moderation.test"
    assert parsed["job_id"] == "j1"
    assert parsed["line"] == "fast"
    assert "text" not in parsed
    assert "secret caption" not in line

