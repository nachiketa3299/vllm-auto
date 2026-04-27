#!/usr/bin/env bash
# vllm-auto 서빙 시작.
# 1) vllm-server start (블록, healthy까지 대기)
# 2) uvicorn app 포그라운드 실행
# Ctrl-C 시 둘 다 정리.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

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

cleanup() {
  echo
  echo ">>> 종료 시그널 — vllm-server stop"
  uv run vllm-server stop || true
}
trap cleanup INT TERM

echo ">>> vllm-server start (config=${VLLM_SERVER_CONFIG})"
uv run vllm-server start

echo ">>> uvicorn app on ${HOST}:${PORT}"
uv run uvicorn app.main:app --host "${HOST}" --port "${PORT}"
