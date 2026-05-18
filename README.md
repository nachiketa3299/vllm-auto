# vllm-auto

## 개요

DGX Spark (GB10) 머신에 Qwen3.5-27B(VL) 모델을 vLLM으로 띄우는 셸 스크립트 묶음

- 실행은 `run_vlm_app.sh` (자동 설치, 멱등)
- 종료는 `stop_vlm_app.sh`

추론 API 는 vLLM 자체의 OpenAI 호환 엔드포인트를 그대로 노출하므로, 클라이언트는 해당 엔드포인트로 직접 추론 요청할 것

인증 없으며 내부망 전용, 단일 머신당 1개 인스턴스

## 사전 조건

DGX Spark(GB10) 기본 이미지엔 보통 다 있다. 새 머신에선 다음을 직접 점검:

- 관리자가 미리 점검할 것
    - Ubuntu aarch64
    - CUDA Version: 13.0
    - Python 3.11+, git, curl
    - build-essential, python3-dev, python3.12-dev
    - 포트는 기본 8000이나, 변경하려면 VLLM_PORT를 환경 변수로 변경할 것

- 셸이 알아서 처리하는 것 (첫 실행, 5~10분)
    - uv 설치, .venv 생성, vllm 설치 (CUDA13용 nightly-index)
    - ninja 설치
    - 모델 가중치 다운로드 (Qwen/Qwen3.5-27B, ~54GB → ./models/qwen3.5-27b)
    - 이후 호출은 위 단계를 모두 점검만 하고 `ready` 신호만 박음 (멱등)

## 사용법

클라이언트는 부팅 후 직접 vLLM 엔드포인트를 이용할 것

모델 기본 정보 조회

```bash
curl http://<host>:8000/v1/models
```

텍스트만 보내려면 다음과 같이 한다

```bash
curl http://<host>:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.5-27b",
    "messages": [{"role":"user","content":"한국의 수도는?"}],
    "max_completion_tokens": 100
  }'
```

이미지와 텍스트를 동시에 보내려면 다음과 같이 한다.
이미지를 base64 data URL 로 인코딩해서 메시지에 포함한다.

```bash
IMG_B64=$(base64 -w 0 photo.jpg)
curl http://<host>:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d "{
    \"model\": \"qwen3.5-27b\",
    \"messages\": [{
      \"role\":\"user\",
      \"content\":[
        {\"type\":\"text\",\"text\":\"이 사진을 묘사해라\"},
        {\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/jpeg;base64,${IMG_B64}\"}}
      ]
    }]
  }"
```

---

## 환경변수

`run_vlm_app.sh`는 다음의 환경 변수를 읽는다

- VLLM_HOST(0.0.0.0) - vLLM의 바인딩 호스트
- VLLM_PORT(8000) - vLLM의 바인딩 포트
- MODEL_PATH(./models/qwen3.5-27b) - 모델 가중치의 디렉토리
- STARTUP_TIMEOUT_SEC(900) - 모델 로딩의 대기 타임아웃(초)

## 출력/로그 위치

- output/status.json - 현재 lifecycle 상태이며, 클라이언트가 폴링하여 ready를 감지할 것
- logs/vllm.log - vLLM 자체의 stdout/stderr
- vllm.pid - vLLM 프로세스의 PID로, stop_vlm_app.sh가 사용

`output/status.json` 의 형식:

```json
// run_vlm_app.sh 가 ready 신호
{"current_status":"ready","base_url":"http://0.0.0.0:8000/v1"}
// 부팅 실패
{"current_status":"failed","message":"vllm startup timeout"}
// stop_vlm_app.sh 완료
{"current_status":"complete","message":"vllm stopped"}
```

## 설정

- max_model_len: 50000 으로 사용할 것

## 디렉토리 구조

```
vllm-auto/
├── scripts/
│   ├── run_vlm_app.sh    # 진입점: venv 셋업 + vllm 부팅 + ready 신호
│   └── stop_vlm_app.sh   # 종료: SIGTERM(grace 30s) → SIGKILL
├── models/               # 모델 가중치가 여기에 다운로드됨
├── logs/                 # vllm 자체 로그 누적
└── output/               # 런타임 상태 파일 (status.json)
```
