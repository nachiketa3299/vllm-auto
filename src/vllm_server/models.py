"""Pydantic models for vllm-server."""

from __future__ import annotations

from pydantic import BaseModel, Field


class VllmConfig(BaseModel):
    """Parsed config for a single vLLM serve invocation.

    Fields prefixed with a vllm flag name map 1:1 to ``vllm serve``
    arguments (snake_case → kebab-case).  Fields documented as
    "server-owned" are consumed by vllm-server itself and never
    forwarded to the child process.
    """

    model: str
    served_model_name: str | None = None
    host: str = "127.0.0.1"
    port: int = 8000
    dtype: str | None = None
    max_model_len: int | None = None
    gpu_memory_utilization: float | None = None
    tensor_parallel_size: int | None = None
    trust_remote_code: bool = False
    limit_mm_per_prompt: dict[str, int] | None = None
    enforce_eager: bool = False
    extra_args: list[str] = Field(default_factory=list)

    # Server-owned (not forwarded to vllm).
    startup_timeout_sec: int = 900
    shutdown_grace_sec: int = 30
    health_poll_interval_sec: int = 2

    @property
    def base_url(self) -> str:
        """OpenAI-compatible base URL consumers should hit."""
        return f"http://{self.host}:{self.port}/v1"


class ServerMeta(BaseModel):
    """Persisted metadata about a running vllm child process.

    Written to ``server.json`` at start, deleted on stop.  Used by
    ``status``/``stop`` and for stale-pid detection.
    """

    pid: int
    pgid: int
    start_time_jiffies: int
    base_url: str
    model: str
    started_at: str


class StatusReport(BaseModel):
    """Unified JSON output for ``start``/``stop``/``status``."""

    running: bool
    healthy: bool | None = None
    pid: int | None = None
    pgid: int | None = None
    base_url: str
    model: str
    started_at: str | None = None
    log_path: str
    error: str | None = None
    message: str | None = None
