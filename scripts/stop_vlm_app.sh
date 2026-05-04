#!/usr/bin/env bash
# vllm-auto: VLM 서버 종료. SIGTERM 으로 30초 grace, 그 후 SIGKILL.
# Unity (또는 다른 클라이언트) 가 추론 끝낸 뒤 SSH 로 호출.
# vllm.pid 가 없거나 프로세스가 이미 죽었으면 멱등 성공.
#
# 출력 컨벤션:
#   - output/status.json 에 종료 상태 기록
#       {"current_status":"complete","message":"vllm stopped"}
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

mkdir -p output

write_status() {
  local payload="$1"
  printf '%s\n' "${payload}" > output/status.json.tmp
  mv -f output/status.json.tmp output/status.json
}

if [ ! -f vllm.pid ]; then
  echo "[stop_vlm_app] No vllm.pid found; nothing to stop."
  write_status '{"current_status":"complete","message":"vllm not running"}'
  exit 0
fi

PID="$(cat vllm.pid)"

if ! kill -0 "${PID}" 2>/dev/null; then
  echo "[stop_vlm_app] PID ${PID} already gone."
  rm -f vllm.pid
  write_status '{"current_status":"complete","message":"vllm already stopped"}'
  exit 0
fi

echo "[stop_vlm_app] Sending SIGTERM to PID ${PID}..."
kill "${PID}" 2>/dev/null || true

# 30초 grace.
for _ in {1..30}; do
  if ! kill -0 "${PID}" 2>/dev/null; then
    break
  fi
  sleep 1
done

if kill -0 "${PID}" 2>/dev/null; then
  echo "[stop_vlm_app] SIGTERM grace expired; sending SIGKILL."
  kill -9 "${PID}" 2>/dev/null || true
fi

rm -f vllm.pid
write_status '{"current_status":"complete","message":"vllm stopped"}'
echo "[stop_vlm_app] Done."
