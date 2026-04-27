"""JSONL request logger.

One line per request. Captures full prompt and full output — internal-network
use only, no PII filtering.
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


@dataclass
class RequestRecord:
    ts: str
    client_ip: Optional[str]
    prompt: Optional[str]
    prompt_chars: int
    has_image: bool
    image_bytes: Optional[int]
    image_mime: Optional[str]
    max_completion_tokens: Optional[int]
    json_output: bool
    enable_thinking: bool
    output_text: Optional[str]
    reasoning_text: Optional[str]
    latency_ms: int
    status: str
    error: Optional[str]
    extra: dict[str, Any] = field(default_factory=dict)


class RequestLogger:
    def __init__(self, log_path: Path):
        self._path = log_path
        self._lock = threading.Lock()
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: RequestRecord) -> None:
        line = json.dumps(asdict(record), ensure_ascii=False)
        with self._lock:
            with self._path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

    @staticmethod
    def now_iso() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
