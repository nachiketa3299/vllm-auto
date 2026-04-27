"""Child process lifecycle: start, stop, health-gated readiness."""

from __future__ import annotations

import contextlib
import fcntl
import os
import shutil
import signal
import subprocess
import time
from datetime import datetime

from vllm_server import state
from vllm_server.config import to_argv
from vllm_server.health import HealthCheckError, probe, wait_until_healthy
from vllm_server.logging_setup import get_logger
from vllm_server.models import ServerMeta, StatusReport, VllmConfig

_logger = get_logger()


def _now_iso() -> str:
    return datetime.now().astimezone().replace(microsecond=0).isoformat()


def _report_from_meta(
    cfg: VllmConfig,
    meta: ServerMeta | None,
    *,
    healthy: bool | None,
    error: str | None = None,
    message: str | None = None,
) -> StatusReport:
    """Build a StatusReport from (optional) meta and the loaded config."""
    return StatusReport(
        running=meta is not None,
        healthy=healthy,
        pid=meta.pid if meta else None,
        pgid=meta.pgid if meta else None,
        base_url=meta.base_url if meta else cfg.base_url,
        model=meta.model if meta else cfg.model,
        started_at=meta.started_at if meta else None,
        log_path=str(state.log_path()),
        error=error,
        message=message,
    )


def _acquire_lock() -> int:
    """Acquire an exclusive fcntl lock on ``server.lock``.

    Raises BlockingIOError if another vllm-server invocation holds it.
    The fd is intentionally leaked for the lifetime of the CLI process;
    the kernel releases the lock on exit.
    """
    lock_fd = os.open(str(state.lock_path()), os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return lock_fd


def _terminate(proc: subprocess.Popen, grace_sec: int) -> None:
    """SIGTERM the process group, wait up to grace_sec, then SIGKILL."""
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + grace_sec
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(0.5)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        return
    # Reap any remaining status.
    with contextlib.suppress(subprocess.TimeoutExpired):
        proc.wait(timeout=5)


def start(cfg: VllmConfig) -> StatusReport:
    """Start vllm serve in the background, blocking until /health OK.

    Idempotent: if a healthy server is already running under our meta,
    returns its current status without disturbing it.
    """
    try:
        _acquire_lock()
    except BlockingIOError:
        return _report_from_meta(
            cfg,
            state.read_meta(),
            healthy=None,
            error="concurrent_start",
            message="another vllm-server start is in progress",
        )

    existing = state.read_meta()
    if existing is not None and not state.is_stale(existing):
        _logger.info("이미 실행 중: pid=%d", existing.pid)
        return _report_from_meta(cfg, existing, healthy=probe(cfg.base_url))

    if existing is not None:
        _logger.warning("stale meta 발견 → 정리: pid=%d", existing.pid)
        state.clear_state()

    vllm_bin = shutil.which("vllm")
    if vllm_bin is None:
        return _report_from_meta(
            cfg,
            None,
            healthy=None,
            error="vllm_not_found",
            message="'vllm' binary not found on PATH; install it first",
        )

    argv = to_argv(cfg)
    _logger.info("vllm serve 기동: %s", " ".join(argv))

    # Intentionally not a context manager: the fd is passed to Popen
    # (which dups it) and then closed by hand in `finally` below.
    log_file = open(state.log_path(), "ab", buffering=0)  # noqa: SIM115
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    proc: subprocess.Popen | None = None
    try:
        proc = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
            env=env,
        )
    finally:
        # Parent no longer needs the fd; the child has its own dup.
        log_file.close()

    # Build and persist meta immediately so stop/status can find the
    # child even if we crash before health passes.
    start_time_jiffies = state.read_start_time_jiffies(proc.pid)
    if start_time_jiffies is None:
        _logger.warning("start_time_jiffies 읽기 실패 (PID 재사용 방어 약화)")
        start_time_jiffies = 0
    meta = ServerMeta(
        pid=proc.pid,
        pgid=os.getpgid(proc.pid),
        start_time_jiffies=start_time_jiffies,
        base_url=cfg.base_url,
        model=cfg.model,
        started_at=_now_iso(),
    )
    state.write_meta(meta)

    try:
        wait_until_healthy(cfg, proc)
    except HealthCheckError as e:
        _logger.error("health check 실패: %s", e)
        _terminate(proc, cfg.shutdown_grace_sec)
        state.clear_state()
        return _report_from_meta(
            cfg,
            None,
            healthy=False,
            error="startup_timeout",
            message=str(e),
        )
    except KeyboardInterrupt:
        _logger.warning("Ctrl-C 감지 → 자식 종료")
        _terminate(proc, cfg.shutdown_grace_sec)
        state.clear_state()
        return _report_from_meta(
            cfg,
            None,
            healthy=False,
            error="interrupted",
            message="startup interrupted by user",
        )

    _logger.info("start 완료: pid=%d base_url=%s", meta.pid, meta.base_url)
    return _report_from_meta(cfg, meta, healthy=True)


def stop(cfg: VllmConfig) -> StatusReport:
    """Terminate the running server.  Idempotent.

    Sends SIGTERM to the process group, waits ``shutdown_grace_sec``,
    then SIGKILL if needed.  Missing or stale meta is treated as
    already-stopped and still returns success.
    """
    meta = state.read_meta()
    if meta is None:
        _logger.info("stop: 실행 중이 아님 (meta 없음)")
        return _report_from_meta(cfg, None, healthy=False)

    if state.is_stale(meta):
        _logger.info("stop: stale meta 정리")
        state.clear_state()
        return _report_from_meta(cfg, None, healthy=False)

    _logger.info(
        "stop: SIGTERM → pgid=%d (grace=%ds)", meta.pgid, cfg.shutdown_grace_sec
    )
    try:
        os.killpg(meta.pgid, signal.SIGTERM)
    except ProcessLookupError:
        state.clear_state()
        return _report_from_meta(cfg, None, healthy=False)

    deadline = time.monotonic() + cfg.shutdown_grace_sec
    while time.monotonic() < deadline:
        if not state.is_alive(meta.pid):
            break
        time.sleep(0.5)

    if state.is_alive(meta.pid):
        _logger.warning("grace 초과 → SIGKILL: pgid=%d", meta.pgid)
        with contextlib.suppress(ProcessLookupError):
            os.killpg(meta.pgid, signal.SIGKILL)

    state.clear_state()
    _logger.info("stop 완료")
    return _report_from_meta(cfg, None, healthy=False)


def status(cfg: VllmConfig, *, probe_http: bool = False) -> StatusReport:
    """Report current state.  ``probe_http`` enables a live /health call."""
    meta = state.read_meta()
    if meta is None or state.is_stale(meta):
        if meta is not None:
            state.clear_state()
        return _report_from_meta(cfg, None, healthy=None if not probe_http else False)

    healthy: bool | None = None
    if probe_http:
        healthy = probe(meta.base_url)
    return _report_from_meta(cfg, meta, healthy=healthy)
