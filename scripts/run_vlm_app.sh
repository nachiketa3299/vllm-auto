#!/usr/bin/env bash
# vllm-auto: VLM 서버를 백그라운드로 띄우고 /v1/models 가 200 응답할 때까지 블록.
# Unity (또는 다른 클라이언트) 가 SSH 로 호출하는 진입점. 한 번 실행하면 모델이 ready 상태로
# 떠 있고, 추론은 클라이언트가 직접 http://<host>:8000/v1/chat/completions 로 친다.
# 종료는 별도로 stop_vlm_app.sh 를 호출.
#
# 출력 컨벤션:
#   - output/status.json 에 진행/완료/실패 상태를 atomic 하게 기록
#       {"current_status":"ready",  "base_url":"http://127.0.0.1:8000/v1"}
#       {"current_status":"failed", "message":"..."}
#   - vllm 자체 로그는 logs/vllm.log
#
# 가정:
#   - 같은 머신의 ./models/qwen3.5-27b 에 모델 가중치가 존재
#   - uv 가 시스템에 설치되어 있음 (없으면 https://docs.astral.sh/uv/ 참고)
#   - 포트 8000 이 비어 있음 (다른 vllm 인스턴스가 점유 중이면 자동 감지하여 부팅 스킵)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

mkdir -p logs output

VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_BASE_URL="http://${VLLM_HOST}:${VLLM_PORT}/v1"
MODEL_PATH="${MODEL_PATH:-./models/qwen3.5-27b}"
STARTUP_TIMEOUT_SEC="${STARTUP_TIMEOUT_SEC:-900}"

write_status() {
  # atomic publish: tmp 로 쓰고 mv. 폴링하는 클라이언트가 부분 파일을 읽지 않게.
  local payload="$1"
  printf '%s\n' "${payload}" > output/status.json.tmp
  mv -f output/status.json.tmp output/status.json
}

# 1. uv 가 PATH 에 없으면 자동 설치 (최초 1회만 인터넷 필요).
#    설치 후 셸 세션에서 즉시 쓸 수 있도록 PATH 를 ~/.local/bin 으로 prepend.
#    ssh 로 비대화형 호출되는 경우에도 동작 (bash -lc 가 .bashrc 못 읽어도 자체적으로 PATH 박음).
export PATH="${HOME}/.local/bin:${PATH}"
if ! command -v uv > /dev/null 2>&1; then
  echo "[run_vlm_app] Installing uv (first run only)..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
  if ! command -v uv > /dev/null 2>&1; then
    write_status '{"current_status":"failed","message":"uv install failed; PATH not updated"}'
    echo "[run_vlm_app] FAILED: uv installation succeeded but uv still not on PATH."
    exit 1
  fi
fi

# 2. 첫 실행이면 venv 부트스트랩 (vllm + 의존성 설치).
if [ ! -d .venv ]; then
  echo "[run_vlm_app] Setting up venv (first run; takes a while)..."
  uv venv
  uv pip install vllm
fi

# 3. 이미 떠 있으면 부팅 스킵 (안전망 — 같은 세션에서 연속 호출 시).
if curl -sf "${VLLM_BASE_URL}/models" > /dev/null 2>&1; then
  echo "[run_vlm_app] vllm already running at ${VLLM_BASE_URL}; skipping start."
else
  echo "[run_vlm_app] Starting vllm serve in background..."
  nohup .venv/bin/vllm serve "${MODEL_PATH}" \
    --host "${VLLM_HOST}" \
    --port "${VLLM_PORT}" \
    --max-model-len 50000 \
    --gpu-memory-utilization 0.90 \
    --reasoning-parser qwen3 \
    --trust-remote-code \
    > logs/vllm.log 2>&1 &
  echo $! > vllm.pid
fi

# 4. /v1/models 가 200 응답할 때까지 폴링 (최대 STARTUP_TIMEOUT_SEC).
echo "[run_vlm_app] Waiting for vllm to become healthy (timeout ${STARTUP_TIMEOUT_SEC}s)..."
deadline=$(( $(date +%s) + STARTUP_TIMEOUT_SEC ))
until curl -sf "${VLLM_BASE_URL}/models" > /dev/null 2>&1; do
  if [ "$(date +%s)" -gt "${deadline}" ]; then
    write_status '{"current_status":"failed","message":"vllm startup timeout"}'
    echo "[run_vlm_app] FAILED: startup timeout. See logs/vllm.log"
    exit 1
  fi
  sleep 5
done

# 5. ready 신호 작성 후 종료. 추론은 클라이언트가 ${VLLM_BASE_URL}/chat/completions 로.
write_status "$(printf '{"current_status":"ready","base_url":"%s"}' "${VLLM_BASE_URL}")"
echo "[run_vlm_app] Ready at ${VLLM_BASE_URL}"
exit 0
