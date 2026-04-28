#!/usr/bin/env bash
# vllm-auto 서빙 시작.
# 기본: 백그라운드 모드. 즉시 셸로 복귀, 로그는 logs/start.out 에.
#   ./scripts/start.sh
# 포그라운드 모드 (Ctrl-C 로 직접 끊고 싶을 때):
#   ./scripts/start.sh --fg
# 종료:
#   ./scripts/stop.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

FG=0
case "${1:-}" in
  -f|--foreground|--fg) FG=1 ;;
  -h|--help)
    cat <<EOF
Usage: $0 [--fg]
  (no args)  백그라운드 모드 (기본). 즉시 복귀, log: logs/start.out
  --fg       포그라운드 모드 (Ctrl-C 직접 끊기)
EOF
    exit 0
    ;;
esac

export VLLM_SERVER_CONFIG="${VLLM_SERVER_CONFIG:-${REPO_ROOT}/configs/vllm.yaml}"
APP_CONFIG="${APP_CONFIG:-${REPO_ROOT}/configs/app.yaml}"

# app.yaml에서 host/port 추출 (환경변수가 우선)
read_yaml() {
  uv run python -c "
import sys, yaml, pathlib
p = pathlib.Path('${APP_CONFIG}')
data = yaml.safe_load(p.read_text(encoding='utf-8')) if p.exists() else {}
print(data.get('$1', '$2'))
"
}

HOST="${APP_HOST:-$(read_yaml host 0.0.0.0)}"
PORT="${APP_PORT:-$(read_yaml port 8080)}"

if [ "$FG" = "1" ]; then
  cleanup() {
    echo
    echo ">>> 종료 시그널 — vllm-server stop"
    uv run vllm-server stop || true
  }
  trap cleanup INT TERM

  echo ">>> vllm-server start (config=${VLLM_SERVER_CONFIG})"
  uv run vllm-server start

  echo ">>> uvicorn app on ${HOST}:${PORT}"
  exec uv run uvicorn app.main:app --host "${HOST}" --port "${PORT}"
fi

# 백그라운드 모드 — 모델 로딩 + uvicorn 다 detach. 부모 셸 즉시 복귀.
mkdir -p logs
LOG="${REPO_ROOT}/logs/start.out"

nohup bash -c "
  set -euo pipefail
  cd '${REPO_ROOT}'
  export VLLM_SERVER_CONFIG='${VLLM_SERVER_CONFIG}'
  echo '>>> vllm-server start'
  uv run vllm-server start
  echo '>>> uvicorn app on ${HOST}:${PORT}'
  exec uv run uvicorn app.main:app --host '${HOST}' --port '${PORT}'
" > "${LOG}" 2>&1 &

PID=$!
echo ">>> vllm-auto 백그라운드 기동"
echo "    parent PID: ${PID}"
echo "    log:        ${LOG}"
echo "    상태 확인:  tail -f ${LOG}"
echo "    health:     curl http://${HOST}:${PORT}/health  (모델 로딩 5~10분 후 응답 시작)"
echo "    종료:       ./scripts/stop.sh"
