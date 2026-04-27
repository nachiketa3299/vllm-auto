import base64
from dataclasses import dataclass
from typing import Optional

from pydantic import BaseModel, Field


@dataclass
class RequestLog:
    entries: list[str]

    def add(self, message: str) -> None:
        self.entries.append(message)


class GenerateResponse(BaseModel):
    """Successful response from POST /generate."""

    output_text: str = Field(
        description="모델이 생성한 본문 텍스트. markdown 코드 펜스(```)는 자동 제거됨. "
        "json_output=true 였으면 JSON 문자열이 그대로 들어감 (파싱은 클라이언트 책임).",
        examples=["서울"],
    )
    reasoning_text: str = Field(
        default="",
        description="enable_thinking=true 일 때 모델 사고 과정. 일반적으론 빈 문자열.",
        examples=[""],
    )
    logs: list[str] = Field(
        default_factory=list,
        description="서버에서 일어난 단계별 로그 (디버깅용). 클라이언트는 보통 무시 가능.",
        examples=[
            [
                "Using prompt text (12 chars)",
                "Sending request to vLLM at http://127.0.0.1:8000/v1 (model=qwen3.5-27b)",
                "Received response from vLLM",
            ]
        ],
    )


class GenerateErrorResponse(BaseModel):
    """Error response from POST /generate (HTTP 400 / 502 / 500)."""

    detail: str = Field(
        description="에러 메시지 (사람이 읽는 형식).",
        examples=["Either an image or prompt text is required."],
    )
    logs: list[str] = Field(
        default_factory=list,
        description="에러 직전까지의 단계별 로그.",
    )


class HealthResponse(BaseModel):
    """Response from GET /health."""

    status: str = Field(examples=["ok"])


@dataclass(frozen=True)
class PreparedImage:
    bytes_data: bytes
    mime_type: str

    @property
    def data_url(self) -> str:
        encoded = base64.b64encode(self.bytes_data).decode("ascii")
        return f"data:{self.mime_type};base64,{encoded}"


@dataclass(frozen=True)
class GeneratedPayload:
    text: str
    reasoning: str = ""


@dataclass(frozen=True)
class ProbedModelInfo:
    model: str
    max_model_len: Optional[int]
    model_path: Optional[str]


class AppError(Exception):
    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail
