from pathlib import Path
from typing import Optional

from fastapi import UploadFile

from .config import AppConfig
from .image import ImagePreparer
from .models import AppError, GeneratedPayload, PreparedImage, RequestLog
from .parsing import ResponseParser
from .vllm_client import VLLMClient


class GenerationService:
    def __init__(
        self,
        *,
        config: AppConfig,
        client: VLLMClient,
    ):
        self.config = config
        self.client = client

    async def generate(
        self,
        upload: Optional[UploadFile],
        log: RequestLog,
        *,
        user_request: Optional[str] = None,
        max_completion_tokens: Optional[int] = None,
        timeout_seconds: Optional[int] = None,
        max_image_bytes: Optional[int] = None,
        json_output: Optional[bool] = None,
        enable_thinking: Optional[bool] = None,
    ) -> tuple[GeneratedPayload, Optional[PreparedImage]]:
        note = (user_request or "").strip()

        resolved_max_completion_tokens = self._positive_int(
            "max_completion_tokens",
            max_completion_tokens,
            self.config.max_completion_tokens,
        )
        resolved_timeout_seconds = self._positive_int(
            "timeout_seconds", timeout_seconds, self.config.timeout_seconds
        )
        resolved_max_image_bytes = self._positive_int(
            "max_image_bytes", max_image_bytes, self.config.max_image_bytes
        )
        resolved_json_output = bool(json_output)
        resolved_enable_thinking = bool(enable_thinking)

        prepared_image: Optional[PreparedImage] = None
        if upload is not None:
            prepared_image = ImagePreparer.from_upload(
                upload,
                max_image_bytes=resolved_max_image_bytes,
                log=log,
            )

        if prepared_image is None and not note:
            raise AppError(400, "Either an image or prompt text is required.")

        if prepared_image is not None:
            log.add("Using image input")
        if note:
            log.add(f"Using prompt text ({len(note)} chars)")
        log.add(
            "Request config: "
            f"max_completion_tokens={resolved_max_completion_tokens}, "
            f"timeout_seconds={resolved_timeout_seconds}, "
            f"max_image_bytes={resolved_max_image_bytes}, "
            f"json_output={resolved_json_output}, "
            f"enable_thinking={resolved_enable_thinking}"
        )

        system_prompt = self._read_system_prompt()

        response = await self.client.create_chat_completion(
            system_prompt=system_prompt,
            user_text=note or None,
            image_data_url=(
                prepared_image.data_url if prepared_image is not None else None
            ),
            response_format={"type": "json_object"} if resolved_json_output else None,
            max_completion_tokens=resolved_max_completion_tokens,
            timeout_seconds=resolved_timeout_seconds,
            enable_thinking=resolved_enable_thinking,
            log=log,
        )
        log.add(f"raw preview: {ResponseParser.preview_content(response)}")
        output_text = ResponseParser.extract_normalized_content(response)
        if not output_text:
            raise AppError(502, "vLLM returned an empty response.")
        reasoning_text = ResponseParser.extract_reasoning_content(response)

        return GeneratedPayload(text=output_text, reasoning=reasoning_text), prepared_image

    def _read_system_prompt(self) -> Optional[str]:
        path: Path = self.config.system_prompt_path
        if not path.exists():
            return None
        text = path.read_text(encoding="utf-8").strip()
        return text or None

    @staticmethod
    def _positive_int(field_name: str, override: Optional[int], default: int) -> int:
        value = default if override is None else override
        if value <= 0:
            raise AppError(400, f"{field_name} must be a positive integer.")
        return value
