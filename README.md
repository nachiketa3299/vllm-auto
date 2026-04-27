# vllm-auto

사내 GX10(NVIDIA DGX Spark / GB10) 머신에 Qwen3.5-27B(VL) 모델을 vLLM으로 띄우고, **사내 자동화 워크플로**가 호출할 수 있는 HTTP 엔드포인트를 노출하는 단일 레포 부트스트랩.

- 외부엔 애플리케이션 레이어 엔드포인트만 노출. vLLM은 가려져 있음
- 인증 없음, 사내망 전용
- 단일 머신당 1개 인스턴스, 클론 → install → start

---

## 사전 조건

DGX Spark(GB10) 기본 이미지엔 보통 다 있다. 새 머신에선 다음을 직접 점검:

### 시스템 (미리 설치돼 있어야 함)
- **Ubuntu aarch64**
- **NVIDIA 드라이버 (CUDA 13 지원, 580+ 권장)** — `nvidia-smi` 로 `CUDA Version: 13.x` 확인. install.sh는 드라이버를 깔지 못함 (커널 모듈 + sudo 필요)
- **Python 3.11+** — `python3 --version`
- **`git`, `curl`, `build-essential`**

### install.sh가 알아서 설치하는 것 (시스템 X, 프로젝트 .venv 안)
- `uv` (없으면 `~/.local/bin`에 설치)
- 일반 의존성 (FastAPI, httpx, pydantic, pyyaml 등)
- vLLM CUDA 13 nightly wheel + `nvidia-cuda-runtime-cu13` / `nvidia-cudnn-cu13` 등 런타임 libs (.venv 격리)
- `huggingface_hub` (vllm 의존성으로 따라옴)
- Qwen3.5-27B 모델 (`./models/qwen3.5-27b/`)

### 머신 자원
- **디스크 ~70GB 여유** — 모델 ~54GB + .venv ~6GB + 캐시 여유. `df -h .` 으로 확인
- **외부 인터넷 접근** — `huggingface.co`, `astral.sh`, `wheels.vllm.ai`, `pypi.org`. 사내망에서 외부 통신이 막혀 있다면 사내 미러 셋업 필요 (이 레포는 그건 다루지 않음)
- **포트 8080, 8000 비어있을 것** — 다른 서비스가 점유 중이면 `APP_PORT` / `configs/vllm.yaml`의 `port` 변경

> **드라이버와 런타임 라이브러리는 다른 층이다.** `nvidia-smi` 가 보이는데 vLLM이 `libcudart.so.X not found` 로 죽으면 드라이버는 OK인데 런타임 libs 매칭이 안 된다는 뜻 — install.sh의 vLLM 설치 단계가 실패했을 가능성이 큼.

---

## 빠른 시작

```bash
git clone <repo-url> vllm-auto
cd vllm-auto
./scripts/install.sh   # 1회. uv + 의존성 + vLLM + 모델 다운로드
./scripts/start.sh     # 매번. vllm-server 기동 + 앱 서버 기동
```

**첫 install은 1~3시간 걸릴 수 있음** — 모델 ~54GB + (필요 시) vLLM 소스 빌드.

---

## 엔드포인트

서버 기본 바인딩: `0.0.0.0:8080`. 사내 IP로 접근 가능.

### `GET /health`

```bash
curl -s http://<host>:8080/health
# {"status":"ok"}
```

### `POST /generate`

multipart/form-data. **`image`와 `user_request` 중 하나 이상 필수.**

| 필드 | 타입 | 기본 | 설명 |
|---|---|---|---|
| `image` | file | — | 이미지 파일 (jpeg/png 등). 최대 15 MiB |
| `user_request` | string | — | 텍스트 프롬프트 |
| `max_completion_tokens` | int | 10000 | 생성 최대 토큰 |
| `timeout_seconds` | int | 600 | vLLM 응답 대기 (초) |
| `max_image_bytes` | int | 15728640 | 이미지 크기 제한 |
| `json_output` | bool | false | true면 모델이 JSON 객체로만 응답하도록 강제 |
| `enable_thinking` | bool | false | true면 reasoning 토큰 생성 (느려지지만 품질↑) |

응답 (200):
```json
{
  "output_text": "...",
  "reasoning_text": "",
  "logs": ["...", "..."]
}
```

에러 응답:
```json
{ "detail": "Either an image or prompt text is required.", "logs": [...] }
```

### curl 예시

**텍스트 전용:**
```bash
curl -s -X POST http://<host>:8080/generate \
  -F 'user_request=한국의 수도는 어디?' | jq .output_text
```

**이미지 + 텍스트:**
```bash
curl -s -X POST http://<host>:8080/generate \
  -F 'image=@photo.jpg' \
  -F 'user_request=이 사진을 한 줄로 묘사해라' | jq .output_text
```

**JSON 강제 + 사고 사용:**
```bash
curl -s -X POST http://<host>:8080/generate \
  -F 'user_request=key=value 두 쌍을 JSON으로 만들어줘' \
  -F 'json_output=true' \
  -F 'enable_thinking=true' | jq .
```

### Python 예시

```python
import httpx

with open("photo.jpg", "rb") as f:
    r = httpx.post(
        "http://<host>:8080/generate",
        data={"user_request": "이 사진을 묘사해라"},
        files={"image": ("photo.jpg", f, "image/jpeg")},
        timeout=600,
    )
print(r.json()["output_text"])
```

자동 OpenAPI 스펙: `http://<host>:8080/docs`, `http://<host>:8080/openapi.json`.

---

## AI 에이전트에게 노출하기

이 서버는 OpenAPI 3.1 스펙을 자동으로 노출한다. 다른 AI 개발자가 자기 에이전트(Claude, GPT, Cursor 등)에게 이 서버를 사용시키려면 다음 정보 한 묶음을 시스템 프롬프트나 도구 정의에 넣으면 된다.

### 핵심 URL

| 경로 | 용도 |
|---|---|
| `http://<host>:8080/openapi.json` | OpenAPI 3.1 스펙 (JSON). AI가 직접 읽고 호출 코드 생성 |
| `http://<host>:8080/docs` | Swagger UI. 사람이 브라우저로 탐색 + "Try it out"로 즉시 호출 |
| `http://<host>:8080/redoc` | ReDoc UI. 정적인 가독성 좋은 문서 |

### 시스템 프롬프트 예시 (AI에게 던지는 한 줄)

```
다음 사내 멀티모달 LLM 엔드포인트를 사용해서 작업해라.
- OpenAPI 스펙: http://<host>:8080/openapi.json
- 핵심 엔드포인트: POST /generate (multipart/form-data)
- 필수: image 또는 user_request 중 하나
- 응답: {output_text, reasoning_text, logs}
- 모델: Qwen3.5-27B (비전+텍스트), 최대 컨텍스트 50000 토큰
- 인증 없음, 사내망 전용
```

이 정도만 줘도 똑똑한 LLM은 OpenAPI를 fetch해서 알아서 호출 코드를 만든다. 응답 스키마 (`GenerateResponse`, `GenerateErrorResponse`)도 컴포넌트로 정의돼 있어 AI가 응답 파싱까지 맞춰 짠다.

### OpenAI function/tool 형식이 필요할 때

OpenAI/Anthropic API의 `tools=[...]` 인자에 직접 박을 수 있는 단일 도구 정의:

```json
{
  "name": "qwen_generate",
  "description": "사내 GX10에서 돌아가는 Qwen3.5-27B(VL)에게 텍스트/이미지 입력으로 응답을 받는다. 이미지·텍스트 중 하나는 필수.",
  "input_schema": {
    "type": "object",
    "properties": {
      "user_request": {"type": "string", "description": "텍스트 프롬프트"},
      "image_path": {"type": "string", "description": "업로드할 이미지 파일 경로 (선택)"},
      "max_completion_tokens": {"type": "integer", "default": 10000},
      "json_output": {"type": "boolean", "default": false}
    }
  }
}
```

각 AI 개발자가 이 도구 정의를 자기 에이전트에 등록하고, 도구 실행 핸들러에서 `POST /generate`로 multipart 호출하면 된다.

### 빠른 호출 패턴 요약

```python
import httpx

def call(prompt: str, image_path: str | None = None) -> str:
    files = {}
    if image_path:
        files["image"] = open(image_path, "rb")
    r = httpx.post(
        "http://<host>:8080/generate",
        data={"user_request": prompt},
        files=files or None,
        timeout=600,
    )
    r.raise_for_status()
    return r.json()["output_text"]
```

---

## 설정 변경

### 포트/호스트/타임아웃 등 (앱 레이어)

`configs/app.yaml` 편집 또는 환경변수:

| 환경변수 | 기본 |
|---|---|
| `APP_HOST` | `0.0.0.0` |
| `APP_PORT` | `8080` |
| `VLLM_BASE_URL` | `http://127.0.0.1:8000/v1` |
| `APP_MAX_COMPLETION_TOKENS` | `10000` |
| `APP_TIMEOUT_SECONDS` | `600` |
| `APP_MAX_IMAGE_BYTES` | `15728640` |
| `APP_LOG_PATH` | `logs/requests.jsonl` |
| `APP_SYSTEM_PROMPT_PATH` | `system_prompt.md` |

```bash
APP_PORT=9090 ./scripts/start.sh
```

### vLLM 설정

`configs/vllm.yaml` 편집. `model`, `max_model_len`, `gpu_memory_utilization` 등이 거기 있음. 변경 후 재시작 필요.

### 시스템 프롬프트

루트의 `system_prompt.md` 편집. **요청마다 파일을 읽으므로 재시작 불필요.**
빈 파일이면 system 메시지를 보내지 않음 (현재 기본).

---

## 운영

### 상태 확인

```bash
uv run vllm-server status --probe
```

### 로그

```bash
tail -f logs/requests.jsonl | jq .
```

JSONL 한 줄 = 한 요청. 필드:
`ts, client_ip, prompt, prompt_chars, has_image, image_bytes, image_mime, max_completion_tokens, json_output, enable_thinking, output_text, reasoning_text, latency_ms, status, error`.

vLLM 자체 로그는 `~/.local/state/vllm-server/server.log`.

### 백그라운드 실행

```bash
nohup ./scripts/start.sh > /tmp/vllm-auto.out 2>&1 &
./scripts/stop.sh   # 종료
```

---

## 모델 수동 다운로드

install.sh의 자동 다운로드가 실패한 경우, 직접 받아서 이 위치에 둔다:

```
./models/qwen3.5-27b/
├── config.json
├── tokenizer.json
├── tokenizer_config.json
├── model.safetensors-00001-of-00011.safetensors
├── ... (전체 safetensors 파일)
└── ...
```

`config.json`이 존재하면 install.sh가 통과한다. 모델 출처: <https://huggingface.co/Qwen/Qwen3.5-27B>.

---

## vLLM 빌드 실패 시

GB10/Blackwell + aarch64 조합은 prebuilt wheel이 없을 수 있다. 시도 순서:

1. `uv pip install --pre vllm` (nightly)
2. 소스 빌드: <https://docs.vllm.ai/en/latest/getting_started/installation.html>
3. NVIDIA 공식 컨테이너(NGC)에서 vLLM 가져오기

한 머신에서 뚫리면 동일 환경의 다른 머신엔 같은 절차로 재현된다.

---

## 디렉터리 구조

```
vllm-auto/
├── configs/
│   ├── app.yaml             # 앱 서버 설정
│   └── vllm.yaml            # vLLM 설정 (vllm serve 인자 1:1 매핑)
├── scripts/
│   ├── install.sh           # 1회 부트스트랩
│   ├── start.sh             # 서빙 시작
│   └── stop.sh              # 백그라운드 종료
├── src/
│   ├── app/                 # FastAPI 앱
│   └── vllm_server/         # vLLM lifecycle CLI
├── models/                  # .gitignore. 모델 가중치
├── logs/                    # .gitignore. requests.jsonl 누적
├── system_prompt.md         # 편집 가능. 매 요청마다 read
└── pyproject.toml
```

---

## 보안 고지

인증 없음. `0.0.0.0:8080` 바인딩이라 같은 사내 네트워크의 누구든 호출 가능. **사내망에서만** 띄울 것.
