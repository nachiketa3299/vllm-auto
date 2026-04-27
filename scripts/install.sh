#!/usr/bin/env bash
# vllm-auto 1회 부트스트랩.
# 빈 GX10에서 실행: 의존성 + Qwen 모델 다운로드.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

MODEL_REPO="${MODEL_REPO:-Qwen/Qwen3.5-27B}"
MODEL_DIR="models/qwen3.5-27b"

log() { printf '\n>>> %s\n' "$*"; }

# 1. uv 설치 확인
if ! command -v uv >/dev/null 2>&1; then
  log "uv 미설치 — 설치 진행"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # shellcheck disable=SC1091
  source "${HOME}/.local/bin/env" 2>/dev/null || export PATH="${HOME}/.local/bin:${PATH}"
fi
log "uv: $(uv --version)"

# 2. 일반 의존성 설치 (venv 생성 + sync)
log "uv sync — 일반 의존성 설치"
uv sync

# 3. vLLM 별도 설치 (CUDA 13 nightly index — DGX Spark/GB10용)
# PyPI 기본 wheel은 CUDA 12 빌드라 CUDA 13 머신에선 libcudart.so.12 missing 오류.
# vllm 공식 nightly에 cu13 빌드가 있어서 거기서 받는다.
if ! uv run python -c "import vllm" 2>/dev/null; then
  log "vLLM 설치 (CUDA 13 nightly wheel, ARM aarch64)"
  if ! uv pip install --pre vllm \
        --extra-index-url https://wheels.vllm.ai/nightly \
        --index-strategy unsafe-best-match; then
    cat >&2 <<'EOF'

[ERROR] vLLM 설치 실패.
머신의 CUDA 버전이 다르거나 wheel이 없을 수 있습니다. 다음을 확인:
  - nvidia-smi 의 CUDA Version (이 install.sh는 CUDA 13 가정)
  - https://docs.vllm.ai/en/latest/getting_started/installation.html
  - NVIDIA NGC 컨테이너 사용
EOF
    exit 1
  fi
else
  log "vLLM 이미 설치됨 — 건너뜀"
fi

# 4. 디렉터리 준비
mkdir -p models logs
[ -f system_prompt.md ] || : > system_prompt.md

# 5. 모델 다운로드
if [ -f "${MODEL_DIR}/config.json" ]; then
  log "모델 이미 존재 (${MODEL_DIR}/config.json) — 다운로드 건너뜀"
else
  log "모델 다운로드: ${MODEL_REPO} → ${MODEL_DIR}"
  if ! uv run hf download "${MODEL_REPO}" --local-dir "${MODEL_DIR}"; then
    cat >&2 <<EOF

[ERROR] 모델 다운로드 실패.
Hugging Face에서 ${MODEL_REPO} 의 모든 파일을 ${REPO_ROOT}/${MODEL_DIR}/ 에 수동으로 받아주세요.
필수 파일: config.json, tokenizer*, model.safetensors* 등.
설치 검증은 ${MODEL_DIR}/config.json 존재 여부로 합니다.
EOF
    exit 1
  fi
fi

# 6. 자가 검증
log "self-check"
uv run vllm-server --help >/dev/null
uv run python -c "from app.main import app; print('app routes:', [r.path for r in app.routes])"

log "install.sh 완료. ./scripts/start.sh 로 서빙 시작."
