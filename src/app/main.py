"""FastAPI app: /health and /generate."""

from __future__ import annotations

import asyncio
import contextlib
import time
from typing import Optional

from fastapi import FastAPI, File, Form, Request, Response, UploadFile
from fastapi.responses import JSONResponse

from .config import AppConfig
from .models import (
    AppError,
    GenerateErrorResponse,
    GenerateResponse,
    HealthResponse,
    RequestLog,
)
from .request_logger import RequestLogger, RequestRecord
from .service import GenerationService
from .vllm_client import VLLMClient

CONFIG = AppConfig.load()
VLLM_CLIENT = VLLMClient(CONFIG)
SERVICE = GenerationService(config=CONFIG, client=VLLM_CLIENT)
REQUEST_LOGGER = RequestLogger(CONFIG.log_path)

app = FastAPI(
    title="vllm-auto",
    description=(
        "사내망용 Qwen3.5-27B(VL) 추론 엔드포인트. 같은 머신의 vLLM 서버를 한 번 래핑한다.\n\n"
        "- **POST `/generate`**: 텍스트, 이미지, 또는 둘 다 보내고 모델 응답을 받는다. "
        "이미지/텍스트 중 하나는 필수.\n"
        "- **GET `/health`**: 단순 헬스체크.\n\n"
        "인증 없음. 사내망 전용. 모든 응답은 application/json."
    ),
    version="0.1.0",
)


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="헬스체크",
    description="앱 서버가 응답 가능한지만 확인한다. 항상 200 OK + `{\"status\":\"ok\"}`.",
)
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post(
    "/generate",
    response_model=GenerateResponse,
    responses={
        400: {
            "model": GenerateErrorResponse,
            "description": "잘못된 요청 (이미지/텍스트 둘 다 비어있음, 양수가 아닌 옵션 값 등)",
        },
        499: {"description": "클라이언트가 연결 끊음 (요청 취소)"},
        500: {"model": GenerateErrorResponse, "description": "서버 내부 오류"},
        502: {
            "model": GenerateErrorResponse,
            "description": "vLLM 서버에 도달 불가 또는 vLLM이 비정상 응답",
        },
    },
    summary="텍스트/이미지 입력으로 모델 응답 생성",
    description=(
        "multipart/form-data 로 호출한다. **`image` 또는 `user_request` 중 하나는 필수**.\n\n"
        "## 옵션 동작\n"
        "- `max_completion_tokens` (기본 10000): 응답 최대 토큰. 짧을수록 빠름.\n"
        "- `timeout_seconds` (기본 600): vLLM 응답 대기 한도(초).\n"
        "- `max_image_bytes` (기본 15728640 = 15 MiB): 이미지 크기 제한.\n"
        "- `json_output` (기본 false): true 면 모델 출력이 JSON 객체가 되도록 vLLM에 강제.\n"
        "- `enable_thinking` (기본 false): true 면 모델이 사고 과정 토큰을 추가 생성. 응답이 느려짐.\n\n"
        "## 예시 — curl\n"
        "```bash\n"
        "# 텍스트만\n"
        "curl -X POST http://<host>:8080/generate \\\n"
        "  -F 'user_request=2 + 2 는?' -F 'max_completion_tokens=20'\n"
        "\n# 이미지 + 텍스트\n"
        "curl -X POST http://<host>:8080/generate \\\n"
        "  -F 'image=@photo.jpg' -F 'user_request=이게 뭐야?'\n"
        "```\n\n"
        "## 예시 — Python (httpx)\n"
        "```python\n"
        "import httpx\n"
        "with open('photo.jpg', 'rb') as f:\n"
        "    r = httpx.post(\n"
        "        'http://<host>:8080/generate',\n"
        "        data={'user_request': '이 사진 묘사해'},\n"
        "        files={'image': ('photo.jpg', f, 'image/jpeg')},\n"
        "        timeout=600,\n"
        "    )\n"
        "print(r.json()['output_text'])\n"
        "```"
    ),
)
async def generate(
    request: Request,
    image: Optional[UploadFile] = File(
        default=None,
        description="이미지 파일 (jpeg/png 등). 최대 `max_image_bytes` 바이트.",
    ),
    user_request: Optional[str] = Form(
        default=None,
        description="사용자 텍스트 프롬프트.",
        examples=["한국의 수도는?"],
    ),
    max_completion_tokens: Optional[int] = Form(
        default=None,
        description="응답 최대 토큰. 미지정 시 서버 기본(10000).",
        examples=[200],
    ),
    timeout_seconds: Optional[int] = Form(
        default=None,
        description="vLLM 응답 대기 한도(초). 미지정 시 서버 기본(600).",
        examples=[600],
    ),
    max_image_bytes: Optional[int] = Form(
        default=None,
        description="이미지 크기 제한(바이트). 미지정 시 서버 기본(15 MiB).",
    ),
    json_output: Optional[bool] = Form(
        default=None,
        description="true 면 모델이 JSON 객체로만 응답하도록 강제.",
        examples=[False],
    ),
    enable_thinking: Optional[bool] = Form(
        default=None,
        description="true 면 사고 과정(reasoning) 토큰을 함께 생성. 응답 느려짐.",
        examples=[False],
    ),
) -> Response:
    log = RequestLog(entries=[])
    started = time.monotonic()
    client_ip = request.client.host if request.client else None

    generation_task = asyncio.create_task(
        SERVICE.generate(
            image,
            log,
            user_request=user_request,
            max_completion_tokens=max_completion_tokens,
            timeout_seconds=timeout_seconds,
            max_image_bytes=max_image_bytes,
            json_output=json_output,
            enable_thinking=enable_thinking,
        )
    )
    disconnect_task = asyncio.create_task(_wait_for_disconnect(request))

    try:
        done, _ = await asyncio.wait(
            {generation_task, disconnect_task},
            return_when=asyncio.FIRST_COMPLETED,
        )

        if disconnect_task in done:
            log.add("Client disconnected; cancelling vLLM request")
            generation_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await generation_task
            _log_request(
                client_ip=client_ip,
                user_request=user_request,
                image=image,
                prepared_image=None,
                max_completion_tokens=max_completion_tokens,
                json_output=json_output,
                enable_thinking=enable_thinking,
                started=started,
                output_text=None,
                reasoning_text=None,
                status="cancelled",
                error="client disconnected",
            )
            return Response(status_code=499)

        disconnect_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await disconnect_task

        generated, prepared_image = await generation_task
        _log_request(
            client_ip=client_ip,
            user_request=user_request,
            image=image,
            prepared_image=prepared_image,
            max_completion_tokens=max_completion_tokens,
            json_output=json_output,
            enable_thinking=enable_thinking,
            started=started,
            output_text=generated.text,
            reasoning_text=generated.reasoning,
            status="ok",
            error=None,
        )
        return JSONResponse(
            {
                "output_text": generated.text,
                "reasoning_text": generated.reasoning,
                "logs": log.entries,
            }
        )
    except AppError as exc:
        log.add(f"[ERROR] {exc.detail}")
        _log_request(
            client_ip=client_ip,
            user_request=user_request,
            image=image,
            prepared_image=None,
            max_completion_tokens=max_completion_tokens,
            json_output=json_output,
            enable_thinking=enable_thinking,
            started=started,
            output_text=None,
            reasoning_text=None,
            status="error",
            error=f"{exc.status_code}: {exc.detail}",
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail, "logs": log.entries},
        )
    except asyncio.CancelledError:
        _log_request(
            client_ip=client_ip,
            user_request=user_request,
            image=image,
            prepared_image=None,
            max_completion_tokens=max_completion_tokens,
            json_output=json_output,
            enable_thinking=enable_thinking,
            started=started,
            output_text=None,
            reasoning_text=None,
            status="cancelled",
            error="cancelled",
        )
        return Response(status_code=499)
    except Exception as exc:
        detail = f"Unhandled server error: {type(exc).__name__}: {exc}"
        log.add(f"[ERROR] {detail}")
        _log_request(
            client_ip=client_ip,
            user_request=user_request,
            image=image,
            prepared_image=None,
            max_completion_tokens=max_completion_tokens,
            json_output=json_output,
            enable_thinking=enable_thinking,
            started=started,
            output_text=None,
            reasoning_text=None,
            status="error",
            error=detail,
        )
        return JSONResponse(
            status_code=500,
            content={"detail": detail, "logs": log.entries},
        )
    finally:
        for task in (generation_task, disconnect_task):
            if not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task


async def _wait_for_disconnect(request: Request) -> None:
    while not await request.is_disconnected():
        await asyncio.sleep(0.25)


def _log_request(
    *,
    client_ip: Optional[str],
    user_request: Optional[str],
    image: Optional[UploadFile],
    prepared_image,
    max_completion_tokens: Optional[int],
    json_output: Optional[bool],
    enable_thinking: Optional[bool],
    started: float,
    output_text: Optional[str],
    reasoning_text: Optional[str],
    status: str,
    error: Optional[str],
) -> None:
    note = (user_request or "").strip()
    image_bytes = None
    image_mime = None
    if prepared_image is not None:
        image_bytes = len(prepared_image.bytes_data)
        image_mime = prepared_image.mime_type
    record = RequestRecord(
        ts=RequestLogger.now_iso(),
        client_ip=client_ip,
        prompt=note or None,
        prompt_chars=len(note),
        has_image=image is not None,
        image_bytes=image_bytes,
        image_mime=image_mime,
        max_completion_tokens=max_completion_tokens,
        json_output=bool(json_output),
        enable_thinking=bool(enable_thinking),
        output_text=output_text,
        reasoning_text=reasoning_text,
        latency_ms=int((time.monotonic() - started) * 1000),
        status=status,
        error=error,
    )
    REQUEST_LOGGER.write(record)
