"""Command-line interface for vllm-server."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from vllm_server import process
from vllm_server.config import load_config
from vllm_server.models import StatusReport

_ROOT_DESC = "로컬 vLLM OpenAI-compatible 서버 lifecycle CLI."

_ROOT_EPILOG = """\
서브커맨드:
  start     백그라운드로 vllm serve 기동 후 /health OK까지 블록
  stop      실행 중인 서버 종료 (SIGTERM → grace → SIGKILL)
  status    현재 상태 조회 (--probe 시 /health 호출)

스트림 컨벤션:
  stdout  →  JSON (StatusReport, 파이프 안전)
  stderr  →  실시간 로그 (사람·AI가 보는 진행 상황)
  exit    →  성공 0 / 실패 1

상태 파일 위치 (XDG 표준):
  ~/.local/state/vllm-server/server.pid
  ~/.local/state/vllm-server/server.json
  ~/.local/state/vllm-server/server.log
  VLLM_SERVER_STATE_HOME 환경변수로 위치 변경 가능.

설정 파일:
  configs/default.yaml 하나를 사용한다. 향후 --config 플래그로 확장 예정.

다른 모듈에서 호출 (Python):
  result = subprocess.run(
      ["vllm-server", "start"],
      capture_output=True, text=True,
  )
  report = json.loads(result.stdout)
  if report["healthy"]:
      use(report["base_url"])
"""

_START_DESC = (
    "vllm serve 를 백그라운드에서 기동하고 /health 가 200 을 반환할 때까지 블록한다. "
    "이미 떠 있으면 멱등 성공 (현재 상태 JSON 반환). "
    "자식은 별도 세션(setsid)으로 분리되어 터미널 종료 후에도 유지된다."
)

_START_EPILOG = """\
출력 형식 (stdout, StatusReport JSON):
  {
    "running": true,
    "healthy": true,
    "pid": 12345,
    "pgid": 12345,
    "base_url": "http://127.0.0.1:8000/v1",
    "model": "/home/.../qwen3.5-27b",
    "started_at": "2026-04-23T10:00:00+09:00",
    "log_path": "/home/.../.local/state/vllm-server/server.log",
    "error": null,
    "message": null
  }

실패 시: running=false, healthy=false, error 코드 및 message 포함. exit=1.

예시:
  vllm-server start
  vllm-server start && vllm-server status --probe
"""

_STOP_DESC = (
    "실행 중인 vllm 자식 프로세스에 SIGTERM 을 보내고 "
    "shutdown_grace_sec 만큼 대기한다. 그래도 살아 있으면 SIGKILL. "
    "실행 중이 아니어도 에러가 아니다 (멱등)."
)

_STOP_EPILOG = """\
출력 형식: StatusReport JSON (running=false).
"""

_STATUS_DESC = (
    "현재 서버 상태를 JSON 으로 출력한다. 기본은 pidfile/메타만 확인 (HTTP 호출 없음). "
    "--probe 를 붙이면 /health 를 실제로 호출해 healthy 필드를 채운다."
)

_STATUS_EPILOG = """\
출력 형식: StatusReport JSON.
--probe 없이 호출하면 healthy=null (네트워크 X), 있으면 true|false.
"""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vllm-server",
        description=_ROOT_DESC,
        epilog=_ROOT_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML 설정 경로. 미지정 시 환경변수 VLLM_SERVER_CONFIG, "
             "그 다음 패키지 내장 configs/default.yaml 순으로 찾는다.",
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser(
        "start",
        help="백그라운드 기동 + health check",
        description=_START_DESC,
        epilog=_START_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers.add_parser(
        "stop",
        help="서버 종료 (멱등)",
        description=_STOP_DESC,
        epilog=_STOP_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    status_p = subparsers.add_parser(
        "status",
        help="상태 조회",
        description=_STATUS_DESC,
        epilog=_STATUS_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    status_p.add_argument(
        "--probe",
        action="store_true",
        help="/health 를 실제로 호출해 healthy 필드를 채움 (기본: 호출 안 함)",
    )

    return parser


def _emit_json(report: StatusReport) -> None:
    json.dump(report.model_dump(), sys.stdout, ensure_ascii=False, indent=2)
    print()


def main() -> None:
    """Entry point for the ``vllm-server`` CLI."""
    parser = _build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    cfg = load_config(args.config)

    if args.command == "start":
        report = process.start(cfg)
        _emit_json(report)
        sys.exit(0 if report.error is None else 1)

    if args.command == "stop":
        report = process.stop(cfg)
        _emit_json(report)
        sys.exit(0 if report.error is None else 1)

    if args.command == "status":
        report = process.status(cfg, probe_http=args.probe)
        _emit_json(report)
        sys.exit(0)


if __name__ == "__main__":
    main()
