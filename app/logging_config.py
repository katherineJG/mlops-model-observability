"""Structured JSON logging. Only whitelisted fields are emitted, so payloads cannot leak."""

import json
import logging
import sys
from datetime import UTC, datetime

LOGGER_NAME = "app"

# Fields callers may attach with `extra=`. Anything else on the record is ignored.
FIELDS = (
    "request_id",
    "method",
    "route",
    "status",
    "latency_ms",
    "model_version",
    "batch_size",
    "outcome",
    "reason",
)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "msg": record.getMessage(),
        }
        entry.update({k: getattr(record, k) for k in FIELDS if hasattr(record, k)})
        if record.exc_info:  # stack traces stay in server logs, never in responses
            entry["exc"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str)


def configure_logging(level: str) -> logging.Logger:
    """Idempotent: replaces handlers on the `app` logger, so repeated calls do not duplicate."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger
