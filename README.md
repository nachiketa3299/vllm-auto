# vllm-auto

사내 GX10(NVIDIA DGX Spark / GB10) 머신에 Qwen3.5-27B(VL) 모델을 vLLM 으로 띄우는 **lifecycle 셸 묶음**.

- 추론 API 는 vLLM 자체의 OpenAI 호환 엔드포인트(`/v1/chat/completions` 등)를 그대로 노출.
- 클라이언트(AI Create Studio Unity 등) 가 SSH 로 본 셸을 호출하여 모델을 ready 상태로 만든 뒤, HTTP 로 vLLM 에 직접 추론 요청.
- 인증 없음, 사내망 전용. 단일 머신당 1개 인스턴스.

이전 버전(FastAPI 래퍼 + lifecycle CLI) 은 폐기됐다. 이유: 클라이언트가 매 요청마다 모델을 콜드 부팅(load → infer → unload) 하는 운영 패턴이 결정되면서 HTTP 래퍼 레이어가 잉여가 됨. 추론 통신은 클라이언트가 vLLM 과 직접 OpenAI API 로 대화한다.

---

## 사전 조건

DGX Spark(GB10) 기본 이미지엔 보통 다 있다. 새 머신에선 다음을 직접 점검:

### 시스템 (미리 설치돼 있어야 함)
- **Ubuntu aarch64**
- **NVIDIA 드라이버 (CUDA 13 지원, 580+ 권장)** — `nvidia-smi` 로 `CUDA Version: 13.x` 확인
- **Python 3.11+** — `python3 --version`
- **`git`, `curl`, `build-essential`**
- **`python3-dev` / `python3.12-dev`** — ARM aarch64 환경에선 `fastsafetensors` 등 일부 vllm 의존성이 prebuilt wheel 이 없어 source build 가 강제됨. Python 헤더(`Python.h`) 가 필요. `run_vlm_app.sh` 가 시작 시 점검하여 누락되면 안내 후 종료. 한 번에 설치:
  ```
  sudo apt install python3-dev python3.12-dev build-essential
  ```

### 셸이 알아서 처리하는 것 (최초 `run_vlm_app.sh` 1회, 인터넷 필요)
- `uv` 설치 (없으면 `curl -LsSf https://astral.sh/uv/install.sh | sh`, `~/.local/bin` 에 설치)
- `.venv/` 생성 (`uv venv`)
- vLLM 설치: `uv pip install --pre vllm --extra-index-url https://wheels.vllm.ai/nightly --index-strategy unsafe-best-match`
  - nightly index 를 쓰는 이유: CUDA 13 + ARM aarch64 prebuilt wheel 이 PyPI 안정 버전엔 없고 nightly 에만 있음. 안 쓰면 fastsafetensors 등 의존성이 source build 강제 → Python.h 필요 등 시스템 빌드툴 의존성이 줄줄이 따라옴.

### 사용자가 미리 준비해야 할 수 있는 것
- **Hugging Face 인증** (모델이 gated 면): `uv run huggingface-cli login` 또는 `HF_TOKEN=hf_xxx ./scripts/run_vlm_app.sh`. 인증 없이 다운로드 되는 모델이면 불필요.
- **모델 가중치**: `run_vlm_app.sh` 가 첫 실행 시 자동 다운로드 (`Qwen/Qwen3.5-27B` -> `./models/qwen3.5-27b/`, 약 54GB). 이미 있으면 (`config.json` 존재) 다운로드 건너뜀. `MODEL_REPO` 환경변수로 다른 모델 지정 가능.

### 머신 자원
- **디스크 ~70GB 여유** — 모델 ~54GB + .venv ~6GB
- **GPU 메모리** — `gpu_memory_utilization: 0.90` 이라 vLLM 이 GPU 한 대를 통째로 잡는다. 다른 모델과 공존 불가.
- **포트 8000 비어있을 것** — 다른 서비스가 점유 중이면 `VLLM_PORT` 환경변수로 변경

---

## 빠른 시작

```bash
git clone <repo-url> vllm-auto
cd vllm-auto

# 모델 가중치를 ./models/qwen3.5-27b/ 에 배치 (위 참고)

# 부팅 (최초 호출은 venv 셋업 + 모델 로딩까지 5~10분 정도 걸린다)
./scripts/run_vlm_app.sh

# 부팅 완료 신호 확인
cat output/status.json
# {"current_status":"ready","base_url":"http://127.0.0.1:8000/v1"}

# 이 시점 이후 클라이언트가 http://<host>:8000/v1/chat/completions 로 직접 추론.
# (예시는 아래 참고)

# 종료
./scripts/stop_vlm_app.sh
cat output/status.json
# {"current_status":"complete","message":"vllm stopped"}
```

`run_vlm_app.sh` 는 멱등이다. 이미 떠 있으면 부팅을 스킵하고 ready 만 다시 신호한다.

---

## 추론 호출 (vLLM OpenAI 호환 API)

부팅이 끝난 후 클라이언트가 직접 vLLM 에 친다. `vllm-auto` 자체는 추론 경로에 끼지 않는다.

### 모델 정보 조회
```bash
curl http://<host>:8000/v1/models
```

### 텍스트만
```bash
curl http://<host>:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.5-27b",
    "messages": [{"role":"user","content":"한국의 수도는?"}],
    "max_completion_tokens": 100
  }'
```

### 이미지 + 텍스트 (멀티모달)
이미지를 base64 data URL 로 인코딩해서 메시지에 포함.
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

### 스트리밍 (SSE)
요청 본문에 `"stream": true` 를 추가하면 응답이 SSE 청크로 흘러온다. 클라이언트는 `data: ` 라인을 파싱하여 `choices[0].delta.content` 를 누적.

### 사고 과정 (thinking) 토큰
요청 본문에 `"chat_template_kwargs": {"enable_thinking": true}` 를 추가하면 모델이 `<think>...</think>` 블록 안에 사고 과정을 먼저 생성하고 그 뒤에 본문을 출력한다. **현재 서버는 `--reasoning-parser` 옵션을 사용하지 않으므로** 사고 토큰은 `<think>...</think>` 태그 글자 그대로 `delta.content` 에 포함되어 흘러온다. 클라이언트가 별도 분리하지 않고 그대로 보여줄 수 있도록 한 의도적 선택.

### JSON 출력 강제
요청 본문에 `"response_format": {"type": "json_object"}` 를 추가.

상세는 vLLM 문서 또는 OpenAI Chat Completions 스펙: <https://platform.openai.com/docs/api-reference/chat/create>.

---

## 환경변수

`run_vlm_app.sh` 가 읽는 환경변수:

| 변수 | 기본 | 설명 |
|---|---|---|
| `VLLM_HOST` | `127.0.0.1` | vLLM 바인딩 호스트. 외부 접속 허용하려면 `0.0.0.0`. |
| `VLLM_PORT` | `8000` | vLLM 바인딩 포트. |
| `MODEL_PATH` | `./models/qwen3.5-27b` | 모델 가중치 디렉토리. |
| `STARTUP_TIMEOUT_SEC` | `900` | 모델 로딩 대기 타임아웃 (초). 초과 시 status 가 `failed`. |

```bash
VLLM_HOST=0.0.0.0 ./scripts/run_vlm_app.sh
```

---

## 출력/로그 위치

| 파일 | 내용 |
|---|---|
| `output/status.json` | 현재 lifecycle 상태. 클라이언트가 폴링하여 ready 감지. |
| `logs/vllm.log` | vLLM 자체 stdout/stderr (로딩 진행, 에러 등) |
| `vllm.pid` | vllm 프로세스 PID. `stop_vlm_app.sh` 가 사용. |

`output/status.json` 의 형식:

```json
// 부팅 진행 중에는 파일 없음 (또는 이전 종료 상태가 남아있음)
// run_vlm_app.sh 가 끝나면:
{"current_status":"ready","base_url":"http://127.0.0.1:8000/v1"}

// 또는 부팅 실패:
{"current_status":"failed","message":"vllm startup timeout"}

// stop_vlm_app.sh 가 끝나면:
{"current_status":"complete","message":"vllm stopped"}
```

원자적 쓰기(`*.tmp` 작성 후 `mv`)를 보장하므로 클라이언트가 부분 파일을 읽지 않는다.

---

## 설정 참고 (`configs/vllm.yaml`)

이전 버전이 사용하던 vLLM 인자 묶음. 현재 셸은 인자를 직접 박지만, 파라미터 기준값(예: `max_model_len: 50000`, `gpu_memory_utilization: 0.90`) 은 이 파일을 보고 맞췄다. 변경하려면 `scripts/run_vlm_app.sh` 의 `vllm serve` 호출을 직접 편집.

---

## 디렉토리 구조

```
vllm-auto/
├── scripts/
│   ├── run_vlm_app.sh    # 진입점: venv 셋업 + vllm 부팅 + ready 신호
│   └── stop_vlm_app.sh   # 종료: SIGTERM(grace 30s) → SIGKILL
├── configs/
│   └── vllm.yaml         # 참고용 파라미터 (셸이 직접 사용하지는 않음)
├── models/               # .gitignore. 모델 가중치를 여기에 둔다 (사용자가 준비)
├── logs/                 # .gitignore. vllm 자체 로그 누적
└── output/               # 런타임 상태 파일 (status.json)
```

---

## 보안 고지

인증 없음. 기본 바인딩이 `127.0.0.1` 이라 외부에서 직접 접속 불가. `VLLM_HOST=0.0.0.0` 로 외부 노출 시 같은 사내 네트워크의 누구든 호출 가능. **사내망에서만** 띄울 것.
