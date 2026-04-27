"""Filesystem state management for vllm-server.

The state directory (``~/.local/state/vllm-server/`` by default) holds
three files across a server's lifetime:

- ``server.pid``  — PID of the child ``vllm serve`` process
- ``server.json`` — ``ServerMeta`` snapshot written at start
- ``server.log``  — child's stdout/stderr (append-only)
- ``server.lock`` — fcntl lock file guarding concurrent start attempts

Override the root via ``VLLM_SERVER_STATE_HOME``.  Stale detection
uses a three-way check (alive + boot-time + cmdline) so a reused PID
doesn't fool us into thinking the server is still up.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from vllm_server.models import ServerMeta


def state_home() -> Path:
    """Return the state directory, creating it with mode 0700 if missing.

    Override with ``VLLM_SERVER_STATE_HOME``.  Default:
    ``$XDG_STATE_HOME/vllm-server`` (typically
    ``~/.local/state/vllm-server``).  The 0700 mode limits log/meta
    readability since logs may surface prompt contents.
    """
    env = os.environ.get("VLLM_SERVER_STATE_HOME")
    if env:
        root = Path(env)
    else:
        xdg = os.environ.get("XDG_STATE_HOME", str(Path.home() / ".local" / "state"))
        root = Path(xdg) / "vllm-server"
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    return root


def pid_path() -> Path:
    return state_home() / "server.pid"


def meta_path() -> Path:
    return state_home() / "server.json"


def log_path() -> Path:
    return state_home() / "server.log"


def lock_path() -> Path:
    return state_home() / "server.lock"


def write_meta(meta: ServerMeta) -> None:
    """Persist meta to server.json and server.pid atomically."""
    mp = meta_path()
    tmp = mp.with_suffix(".json.tmp")
    tmp.write_text(meta.model_dump_json(indent=2), encoding="utf-8")
    tmp.replace(mp)
    pid_path().write_text(str(meta.pid), encoding="utf-8")


def read_meta() -> ServerMeta | None:
    """Read server.json; return None if missing or corrupt."""
    mp = meta_path()
    if not mp.exists():
        return None
    try:
        return ServerMeta.model_validate_json(mp.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, ValueError):
        return None


def clear_state() -> None:
    """Remove pid and meta files.  Idempotent; leaves log.txt alone."""
    for p in (pid_path(), meta_path()):
        p.unlink(missing_ok=True)


def is_alive(pid: int) -> bool:
    """True if ``pid`` currently refers to a live process."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Exists but owned by another user — treat as alive.
        return True
    return True


def read_start_time_jiffies(pid: int) -> int | None:
    """Read field 22 of /proc/<pid>/stat (process start time in jiffies).

    Returns None if unreadable.  Used to defeat PID reuse: the kernel's
    boot-relative start time is stable across the process's lifetime.
    """
    try:
        raw = Path(f"/proc/{pid}/stat").read_text()
    except (FileNotFoundError, PermissionError):
        return None
    # The comm field (argv[0]) can contain spaces and parens; split on
    # the last ')' to skip past it safely.
    rparen = raw.rfind(")")
    if rparen < 0:
        return None
    fields = raw[rparen + 2 :].split()
    # After comm, we're now at field 3 (state).  starttime is field 22,
    # which is index 22 - 3 = 19 in this tail slice.
    if len(fields) < 20:
        return None
    try:
        return int(fields[19])
    except ValueError:
        return None


def read_cmdline(pid: int) -> list[str] | None:
    """Return argv of ``pid`` or None if unreadable."""
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except (FileNotFoundError, PermissionError):
        return None
    if not raw:
        return None
    return [
        s.decode("utf-8", errors="replace") for s in raw.rstrip(b"\x00").split(b"\x00")
    ]


def _cmdline_looks_like_vllm(argv: list[str]) -> bool:
    """Heuristic: does this argv correspond to a vllm serve invocation?

    Matches the common shapes:
      - ``vllm serve ...``                             (argv[0] basename)
      - ``python /.../bin/vllm serve ...``             (wrapper script path)
      - ``python -m vllm.entrypoints.openai.api_server ...``
    """
    if not argv:
        return False
    for token in argv:
        if Path(token).name == "vllm":
            return True
    joined = " ".join(argv)
    return "vllm.entrypoints" in joined


def is_stale(meta: ServerMeta) -> bool:
    """Decide whether the meta refers to a no-longer-valid process.

    Three defensive checks; any failure flags the meta as stale.
    """
    if not is_alive(meta.pid):
        return True
    jiffies = read_start_time_jiffies(meta.pid)
    if jiffies is None or jiffies != meta.start_time_jiffies:
        return True
    argv = read_cmdline(meta.pid)
    return argv is None or not _cmdline_looks_like_vllm(argv)
