from typing import Any

from .models import AppError


class ResponseParser:
    @staticmethod
    def preview_content(response: dict[str, Any], limit: int = 400) -> str:
        normalized = ResponseParser.extract_normalized_content(response)
        preview = normalized[:limit]
        if len(normalized) > limit:
            preview += "...(truncated)"
        return preview

    @staticmethod
    def extract_normalized_content(response: dict[str, Any]) -> str:
        content = ResponseParser._extract_message_content(response)
        return ResponseParser._strip_markdown_fence(content)

    @staticmethod
    def extract_reasoning_content(response: dict[str, Any]) -> str:
        try:
            message = response["choices"][0]["message"]
        except (KeyError, IndexError, TypeError):
            return ""
        if not isinstance(message, dict):
            return ""
        reasoning = message.get("reasoning_content")
        if not isinstance(reasoning, str):
            return ""
        return reasoning.strip()

    @staticmethod
    def _extract_message_content(response: dict[str, Any]) -> str:
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise AppError(502, "vLLM response did not contain a chat message.") from exc

        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text = item.get("text")
                    if isinstance(text, str):
                        text_parts.append(text)
            content = "\n".join(text_parts)

        if not isinstance(content, str) or not content.strip():
            raise AppError(502, "vLLM returned an empty response.")

        return content

    @staticmethod
    def _strip_markdown_fence(content: str) -> str:
        normalized = content.strip()
        if not normalized.startswith("```"):
            return normalized

        lines = normalized.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()
