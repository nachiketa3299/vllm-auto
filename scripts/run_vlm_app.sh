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

# 1.5. 시스템 빌드툴 사전 점검.
#      ARM aarch64 환경에선 fastsafetensors 등 일부 vllm 의존성이 prebuilt wheel 이
#      없어 source build 가 강제됨. 그 build 는 Python.h (python3-dev) 가 필요.
#      sudo 가 필요한 패키지라 자동 설치는 안 하고, 누락되면 에러 + 안내 후 종료.
PY_HEADER="/usr/include/python3.12/Python.h"
if [ ! -f "${PY_HEADER}" ] && [ -z "$(find /usr/include -maxdepth 2 -name 'Python.h' 2>/dev/null | head -1)" ]; then
  write_status '{"current_status":"failed","message":"python3-dev not installed (Python.h missing)"}'
  cat >&2 <<'HINT'
[run_vlm_app] FAILED: Python.h not found.
ARM aarch64 build of fastsafetensors (vllm 의존성) needs system Python headers.
Install once with:
    sudo apt install python3-dev python3.12-dev build-essential
Then re-run this script.
HINT
  exit 1
fi

# 2. 첫 실행이면 venv 부트스트랩 (vllm + 의존성 설치).
#    핵심: vllm 은 CUDA 13 nightly wheel index 에서 받아야 한다.
#    PyPI 기본 인덱스는 CUDA 12 + 일부 의존성(fastsafetensors 등) 의 ARM aarch64
#    prebuilt 가 없어 source build 를 강제 -> Python.h 등 시스템 빌드툴 필요해짐.
#    nightly index 에는 cu13 + aarch64 prebuilt 가 있어서 컴파일 단계 자체가 생략됨.
if [ ! -d .venv ]; then
  echo "[run_vlm_app] Setting up venv (first run; takes a while)..."
  uv venv
  uv pip install --pre vllm \
    --extra-index-url https://wheels.vllm.ai/nightly \
    --index-strategy unsafe-best-match
fi

# 2.3. ninja 사전 보장 — flashinfer 가 sampling/attention kernel 을 JIT 컴파일할 때
#      ninja 빌드 시스템이 필요하다. vllm 의존성에 자동으로 안 따라오는 케이스가
#      있어 명시 설치. PyPI 의 ninja 패키지는 venv/bin/ninja 바이너리를 함께 배포.
if [ ! -x ".venv/bin/ninja" ]; then
  echo "[run_vlm_app] Installing ninja into .venv (flashinfer JIT compile)..."
  uv pip install ninja
fi

# 2.4. .venv/bin 을 PATH 에 prepend — flashinfer 가 subprocess 로 ninja 를 호출할 때
#      PATH 에서 찾을 수 있어야 한다. uv run 외부에서 .venv/bin/vllm 을 직접 호출하므로
#      PATH 가 자동으로 잡히지 않음.
export PATH="${REPO_ROOT}/.venv/bin:${PATH}"

# 2.5. 모델 가중치 자동 다운로드 (없으면).
#      옛 install.sh 의 5단계를 흡수. config.json 존재 여부로 멱등 체크.
#      약 54GB. 인증이 필요한 경우 HF_TOKEN 환경변수 또는 사전 huggingface-cli login.
MODEL_REPO="${MODEL_REPO:-Qwen/Qwen3.5-27B}"
if [ ! -f "${MODEL_PATH}/config.json" ]; then
  echo "[run_vlm_app] Model weights missing. Downloading ${MODEL_REPO} -> ${MODEL_PATH} (~54GB, takes a while)..."
  mkdir -p "${MODEL_PATH}"
  if ! uv run hf download "${MODEL_REPO}" --local-dir "${MODEL_PATH}"; then
    write_status "$(printf '{"current_status":"failed","message":"model download failed for %s"}' "${MODEL_REPO}")"
    cat >&2 <<HINT
[run_vlm_app] FAILED: model download failed.
Hugging Face 인증이 필요한 모델이면:
    HF_TOKEN=hf_xxx ./scripts/run_vlm_app.sh
또는 사전에:
    uv run huggingface-cli login
또는 수동으로 ${MODEL_REPO} 의 모든 파일을 ${MODEL_PATH} 에 직접 받으세요.
HINT
    exit 1
  fi
fi

# vllm 내부에서 transformers/huggingface_hub 가 상대경로 './xxx' 를 repo_id 로
# 오인하는 경우가 있어 절대경로로 통일.
MODEL_PATH="$(cd "${MODEL_PATH}" && pwd)"

# 3. 이미 떠 있으면 부팅 스킵 (안전망 — 같은 세션에서 연속 호출 시).
if curl -sf "${VLLM_BASE_URL}/models" > /dev/null 2>&1; then
  echo "[run_vlm_app] vllm already running at ${VLLM_BASE_URL}; skipping start."
else
  echo "[run_vlm_app] Starting vllm serve in background... (model=${MODEL_PATH})"
  # --reasoning-parser qwen3: 모델이 emit 하는 </think> 를 기준으로 vllm 이 응답을
  # delta.reasoning_content 와 delta.content 두 필드로 자동 분리해 준다.
  # 클라이언트는 두 필드만 그대로 받아쓰면 되고, <think> 태그 직접 파싱 같은 게 필요 없어진다.
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
