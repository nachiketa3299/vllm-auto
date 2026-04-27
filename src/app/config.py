"""App configuration: YAML + environment variable override."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class AppConfig:
    host: str
    port: int
    vllm_base_url: str
    max_completion_tokens: int
    timeout_seconds: int
    max_image_bytes: int
    log_path: Path
    system_prompt_path: Path

    @classmethod
    def load(cls, config_path: Path | None = None) -> "AppConfig":
        path = config_path or _default_config_path()
        data: dict[str, Any] = {}
        if path.exists():
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

        repo_root = _repo_root()

        def _resolve_path(value: str) -> Path:
            p = Path(value)
            return p if p.is_absolute() else (repo_root / p).resolve()

        host = os.environ.get("APP_HOST", data.get("host", "0.0.0.0"))
        port = int(os.environ.get("APP_PORT", data.get("port", 8080)))
        vllm_base_url = os.environ.get(
            "VLLM_BASE_URL",
            data.get("vllm_base_url", "http://127.0.0.1:8000/v1"),
        )
        max_completion_tokens = int(
            os.environ.get(
                "APP_MAX_COMPLETION_TOKENS",
                data.get("max_completion_tokens", 10000),
            )
        )
        timeout_seconds = int(
            os.environ.get(
                "APP_TIMEOUT_SECONDS",
                data.get("timeout_seconds", 600),
            )
        )
        max_image_bytes = int(
            os.environ.get(
                "APP_MAX_IMAGE_BYTES",
                data.get("max_image_bytes", 15 * 1024 * 1024),
            )
        )
        log_path = _resolve_path(
            os.environ.get(
                "APP_LOG_PATH",
                data.get("log_path", "logs/requests.jsonl"),
            )
        )
        system_prompt_path = _resolve_path(
            os.environ.get(
                "APP_SYSTEM_PROMPT_PATH",
                data.get("system_prompt_path", "system_prompt.md"),
            )
        )

        return cls(
            host=host,
            port=port,
            vllm_base_url=vllm_base_url,
            max_completion_tokens=max_completion_tokens,
            timeout_seconds=timeout_seconds,
            max_image_bytes=max_image_bytes,
            log_path=log_path,
            system_prompt_path=system_prompt_path,
        )


def _repo_root() -> Path:
    # src/app/config.py → repo root is parent.parent.parent
    return Path(__file__).resolve().parent.parent.parent


def _default_config_path() -> Path:
    return _repo_root() / "configs" / "app.yaml"
