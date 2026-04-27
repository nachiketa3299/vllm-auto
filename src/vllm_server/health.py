"""HTTP health-check helpers for the vLLM server."""

from __future__ import annotations

import subprocess
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse, urlunparse

from vllm_server.logging_setup import get_logger
from vllm_server.models import VllmConfig

_logger = get_logger()


class HealthCheckError(Exception):
    """Raised when the child process dies or the deadline is exceeded."""


def _health_url(base_url: str) -> str:
    """Resolve ``<base_url>`` to its sibling ``/health`` endpoint.

    vLLM exposes ``/health`` at the server root (peer to ``/v1``), so
    we strip any path component from the provided URL.
    """
    parsed = urlparse(base_url)
    return urlunparse((parsed.scheme, parsed.netloc, "/health", "", "", ""))


def probe(base_url: str, timeout: float = 5.0) -> bool:
    """Return True iff ``GET {base_url%/v1}/health`` returns 200.

    Treats 503 (loading) and all network errors as "not yet ready".
    """
    url = _health_url(base_url)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310
            return resp.status == 200
    except urllib.error.HTTPError:
        return False
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
        return False


def wait_until_healthy(cfg: VllmConfig, proc: subprocess.Popen) -> None:
    """Poll ``/health`` until 200, or raise on timeout/child death.

    The child process is re-checked on every iteration via
    ``proc.poll()``; if it exits (port conflict, OOM, bad args) we bail
    out immediately rather than waiting for the full deadline.
    """
    deadline = time.monotonic() + cfg.startup_timeout_sec
    interval = max(1, cfg.health_poll_interval_sec)
    _logger.info(
        "health check 시작: url=%s timeout=%ds interval=%ds",
        _health_url(cfg.base_url),
        cfg.startup_timeout_sec,
        interval,
    )
    while True:
        rc = proc.poll()
        if rc is not None:
            raise HealthCheckError(
                f"vllm process exited during startup (returncode={rc}); see log"
            )
        if probe(cfg.base_url):
            _logger.info("health check OK")
            return
        if time.monotonic() >= deadline:
            raise HealthCheckError(
                f"health check timed out after {cfg.startup_timeout_sec}s; see log"
            )
        time.sleep(interval)
