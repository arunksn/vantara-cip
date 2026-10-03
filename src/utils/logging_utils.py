"""Structured (JSON) logging. Use instead of print statements."""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any

_CONFIGURED = False


class JsonFormatter(logging.Formatter):
    """Format log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        extra = getattr(record, "extra_fields", None)
        if extra:
            payload.update(extra)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def setup_logging(level: str | None = None) -> None:
    """Configure root logging once."""
    global _CONFIGURED
    lvl = (level or os.environ.get("LOG_LEVEL") or "INFO").upper()
    root = logging.getLogger()
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        root.handlers = [handler]
        _CONFIGURED = True
    root.setLevel(lvl)


def get_logger(name: str) -> logging.Logger:
    """Return a module logger (configures logging on first use)."""
    setup_logging()
    return logging.getLogger(name)


def log_event(logger: logging.Logger, message: str, **fields: Any) -> None:
    """Log an INFO message with structured key/value fields."""
    logger.info(message, extra={"extra_fields": fields})
