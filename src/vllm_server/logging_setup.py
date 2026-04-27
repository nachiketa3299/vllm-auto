"""Stderr-only logging setup for vllm-server.

vllm's own stdout/stderr is routed to ``server.log``, so the
vllm-server CLI itself only needs human-readable progress on stderr::

    [2026-04-23 10:00:00] [INFO] start 시작
"""

from __future__ import annotations

import logging
import sys

_LOGGER_NAME = "vllm_server"
_FORMAT = "[%(asctime)s] [%(levelname)s] %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"

_RESET = "\033[0m"
_LEVEL_COLORS = {
    "DEBUG": "\033[90m",
    "INFO": "\033[36m",
    "WARNING": "\033[33m",
    "ERROR": "\033[31m",
    "CRITICAL": "\033[1;31m",
}


class _ColorLevelFormatter(logging.Formatter):
    """Wraps only the level name in ANSI colors, nothing else."""

    def format(self, record: logging.LogRecord) -> str:
        color = _LEVEL_COLORS.get(record.levelname)
        if not color:
            return super().format(record)
        original = record.levelname
        record.levelname = f"{color}{original}{_RESET}"
        try:
            return super().format(record)
        finally:
            record.levelname = original


def get_logger() -> logging.Logger:
    """Return the configured vllm-server logger.  Idempotent."""
    logger = logging.getLogger(_LOGGER_NAME)
    if logger.handlers:
        return logger

    handler = logging.StreamHandler(sys.stderr)
    if sys.stderr.isatty():
        handler.setFormatter(_ColorLevelFormatter(_FORMAT, datefmt=_DATEFMT))
    else:
        handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))

    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.propagate = False
    return logger
