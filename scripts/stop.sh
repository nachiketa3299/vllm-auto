#!/usr/bin/env bash
# 백그라운드 모드용 종료 스크립트.
# uvicorn(app)을 SIGTERM으로 끊고, vllm-server는 자체 stop 사용.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

export VLLM_SERVER_CONFIG="${VLLM_SERVER_CONFIG:-${REPO_ROOT}/configs/vllm.yaml}"

if pgrep -f "uvicorn app.main:app" >/dev/null; then
  echo ">>> uvicorn 종료"
  pkill -TERM -f "uvicorn app.main:app" || true
  # 최대 10초 대기 후 강제 종료
  for _ in $(seq 1 20); do
    pgrep -f "uvicorn app.main:app" >/dev/null || break
    sleep 0.5
  done
  if pgrep -f "uvicorn app.main:app" >/dev/null; then
    pkill -KILL -f "uvicorn app.main:app" || true
  fi
else
  echo ">>> uvicorn 미실행"
fi

echo ">>> vllm-server stop"
uv run vllm-server stop
