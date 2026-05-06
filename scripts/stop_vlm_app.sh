#!/usr/bin/env bash
# vllm-auto: VLM 서버 종료. SIGTERM 30초 grace, 그 후 SIGKILL.
# Unity (또는 다른 클라이언트) 가 추론 끝낸 뒤 SSH 로 호출.
#
# 종료 권위 기준은 "vllm 포트 (기본 8000) 가 비어 있는가". vllm 은 엔진/워커를
# 다중 프로세스로 띄우므로 vllm.pid 한 개만 죽이면 워커가 살아남아 포트를 계속 잡는다.
# 따라서:
#   1) vllm.pid 의 자기 + 후손 프로세스에 SIGTERM
#   2) 추가로 포트를 LISTEN 하는 모든 PID 의 트리에도 SIGTERM
#   3) 30 초 동안 포트가 비기를 기다림
#   4) 안 비면 동일 집합에 SIGKILL
#
# 출력:
#   output/status.json
#     {"current_status":"complete","message":"vllm stopped"} 또는
#     {"current_status":"failed","message":"vllm still listening on port"}
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

mkdir -p output

VLLM_PORT="${VLLM_PORT:-8000}"

write_status() {
  local payload="$1"
  printf '%s\n' "${payload}" > output/status.json.tmp
  mv -f output/status.json.tmp output/status.json
}

# ${VLLM_PORT} 를 LISTEN 중인 PID 들. 비어있으면 빈 출력.
# ss 출력 예: "users:((\"vllm\",pid=12345,fd=5))" — pid= 다음을 잘라냄.
list_port_pids() {
  ss -lptn "sport = :${VLLM_PORT}" 2>/dev/null \
    | awk -F'pid=' 'NR>1 {split($2,a,","); print a[1]}' \
    | sort -u
}

# 입력 PID 의 자기 + 모든 후손 PID 를 출력 (한 줄에 하나).
# pgrep -P 는 직계 자식만 주므로 재귀.
descendants() {
  local pid="$1"
  echo "$pid"
  local kids
  kids=$(pgrep -P "$pid" 2>/dev/null || true)
  for k in ${kids}; do
    descendants "$k"
  done
}

# 죽일 PID 집합 수집 — vllm.pid 트리 + 포트 LISTEN 트리.
collect_targets() {
  local out=""
  if [ -f vllm.pid ]; then
    local recorded
    recorded="$(cat vllm.pid)"
    if [ -n "${recorded}" ] && kill -0 "${recorded}" 2>/dev/null; then
      out+=$'\n'"$(descendants "${recorded}")"
    fi
  fi
  local p
  for p in $(list_port_pids); do
    out+=$'\n'"$(descendants "$p")"
  done
  echo "${out}" | grep -E '^[0-9]+$' | sort -u || true
}

# 1) 1차 SIGTERM 대상 모음.
TARGETS="$(collect_targets)"

if [ -z "${TARGETS}" ]; then
  echo "[stop_vlm_app] Nothing to stop (no pid file process, no port ${VLLM_PORT} listener)."
  rm -f vllm.pid
  write_status '{"current_status":"complete","message":"vllm not running"}'
  exit 0
fi

echo "[stop_vlm_app] SIGTERM -> $(echo "${TARGETS}" | tr '\n' ' ')"
echo "${TARGETS}" | xargs -r kill 2>/dev/null || true

# 2) 30 초 동안 포트가 비기를 대기.
for _ in $(seq 1 30); do
  if [ -z "$(list_port_pids)" ]; then
    rm -f vllm.pid
    write_status '{"current_status":"complete","message":"vllm stopped"}'
    echo "[stop_vlm_app] vllm stopped cleanly."
    exit 0
  fi
  sleep 1
done

# 3) 안 비면 SIGKILL — 그 사이 PID 가 바뀌었을 수 있으니 다시 수집.
REMAINING="$(collect_targets)"
if [ -n "${REMAINING}" ]; then
  echo "[stop_vlm_app] SIGTERM grace expired; SIGKILL -> $(echo "${REMAINING}" | tr '\n' ' ')"
  echo "${REMAINING}" | xargs -r kill -9 2>/dev/null || true
  sleep 2
fi

rm -f vllm.pid

if [ -z "$(list_port_pids)" ]; then
  write_status '{"current_status":"complete","message":"vllm stopped (forced)"}'
  echo "[stop_vlm_app] vllm forcibly stopped."
  exit 0
else
  write_status '{"current_status":"failed","message":"vllm still listening on port"}'
  echo "[stop_vlm_app] FAILED: vllm still listening on port ${VLLM_PORT}." >&2
  exit 1
fi
