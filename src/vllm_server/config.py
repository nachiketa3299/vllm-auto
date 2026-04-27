"""YAML config loader and ``vllm serve`` argv builder."""

from __future__ import annotations

import json
import os
from pathlib import Path

import yaml

from vllm_server.models import VllmConfig

# Fields consumed by vllm-server itself; never forwarded to vllm serve.
_SERVER_OWNED_FIELDS = frozenset(
    {
        "startup_timeout_sec",
        "shutdown_grace_sec",
        "health_poll_interval_sec",
        "extra_args",
        "model",  # positional, handled separately
    }
)


def default_config_path() -> Path:
    """Return the shipped ``configs/default.yaml`` path.

    Supports both editable/source layout (``<repo>/configs/``) and an
    installed wheel where the configs directory is relocated next to
    the package (``<site-packages>/vllm_server/configs/``, via
    hatchling ``force-include``).
    """
    pkg_dir = Path(__file__).resolve().parent
    installed = pkg_dir / "configs" / "default.yaml"
    if installed.exists():
        return installed
    return pkg_dir.parent.parent / "configs" / "default.yaml"


def load_config(path: Path | None = None) -> VllmConfig:
    """Parse a YAML config file into a VllmConfig.

    Resolution order: explicit ``path`` arg > ``VLLM_SERVER_CONFIG`` env >
    shipped ``configs/default.yaml``.
    """
    if path is None:
        env_path = os.environ.get("VLLM_SERVER_CONFIG")
        if env_path:
            path = Path(env_path)
    cfg_path = path or default_config_path()
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    return VllmConfig.model_validate(data)


def _flag(name: str) -> str:
    return "--" + name.replace("_", "-")


def to_argv(cfg: VllmConfig) -> list[str]:
    """Build the argv list for ``vllm serve`` from a VllmConfig.

    Rules:
      - First tokens are always ``["vllm", "serve", <model>]``.
      - ``None`` values are skipped.
      - ``True`` becomes ``--flag`` (value omitted); ``False`` is skipped.
      - ``list`` values emit the flag once per element.
      - ``dict`` values are JSON-encoded and emitted as a single string.
      - snake_case field names become kebab-case flags.
      - Flags (other than the leading model) are sorted alphabetically for
        reproducible logs.
      - ``extra_args`` is appended verbatim at the end as an escape hatch.
    """
    fields = cfg.model_dump()
    flags: list[tuple[str, list[str]]] = []
    for name, value in fields.items():
        if name in _SERVER_OWNED_FIELDS:
            continue
        if value is None:
            continue
        if isinstance(value, bool):
            if value:
                flags.append((name, [_flag(name)]))
            continue
        if isinstance(value, list):
            tokens: list[str] = []
            for v in value:
                tokens.extend([_flag(name), str(v)])
            if tokens:
                flags.append((name, tokens))
            continue
        if isinstance(value, dict):
            flags.append((name, [_flag(name), json.dumps(value, ensure_ascii=False)]))
            continue
        flags.append((name, [_flag(name), str(value)]))

    flags.sort(key=lambda pair: pair[0])
    argv: list[str] = ["vllm", "serve", cfg.model]
    for _, tokens in flags:
        argv.extend(tokens)
    argv.extend(cfg.extra_args)
    return argv
