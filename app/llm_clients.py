from __future__ import annotations

import base64
import io
import json
import re
import time
from pathlib import Path
from typing import Any, Callable

from openai import OpenAI
from PIL import Image

from app.config import ModelEndpointConfig


class JsonResponseMixin:
    @staticmethod
    def parse_json(text: str) -> dict[str, Any]:
        cleaned = JsonResponseMixin._clean_model_text(text)
        candidate_texts = [cleaned]

        fenced_blocks = re.findall(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.IGNORECASE | re.DOTALL)
        candidate_texts.extend(block.strip() for block in fenced_blocks if block.strip())

        first_brace = min((idx for idx in (cleaned.find("{"), cleaned.find("[")) if idx != -1), default=-1)
        if first_brace != -1:
            candidate_texts.append(cleaned[first_brace:].strip())

        decoder = json.JSONDecoder()
        seen: set[str] = set()
        for candidate in candidate_texts:
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            try:
                value = json.loads(candidate)
            except json.JSONDecodeError:
                value = JsonResponseMixin._scan_first_json_object(candidate, decoder)
            if isinstance(value, dict):
                return value
        raise ValueError(f"模型未返回可解析 JSON，原始文本: {cleaned[:1000]}")

    @staticmethod
    def normalize_content(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                JsonResponseMixin._content_item_text(item)
                for item in content
                if JsonResponseMixin._content_item_type(item) in {"text", "output_text"}
            )
        return str(content)

    @staticmethod
    def _clean_model_text(text: str) -> str:
        cleaned = text.strip()
        cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.IGNORECASE | re.DOTALL).strip()
        if cleaned.startswith("```"):
            lines = [line for line in cleaned.splitlines() if not line.strip().startswith("```")]
            cleaned = "\n".join(lines).strip()
        return cleaned

    @staticmethod
    def _scan_first_json_object(text: str, decoder: json.JSONDecoder) -> dict[str, Any] | None:
        for index in range(len(text) - 1, -1, -1):
            if text[index] not in "{[":
                continue
            try:
                value, _ = decoder.raw_decode(text[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
        return None

    @staticmethod
    def _content_item_type(item: Any) -> str | None:
        if isinstance(item, dict):
            return item.get("type")
        return getattr(item, "type", None)

    @staticmethod
    def _content_item_text(item: Any) -> str:
        if isinstance(item, dict):
            return str(item.get("text", ""))
        return str(getattr(item, "text", "") or "")


class OpenAICompatibleJsonClient(JsonResponseMixin):
    def __init__(
        self,
        config: ModelEndpointConfig,
        max_retries: int = 0,
        retry_backoff_seconds: float = 0.0,
        event_logger: Callable[..., None] | None = None,
    ) -> None:
        self.config = config
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.event_logger = event_logger
        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout_seconds,
        )

    @staticmethod
    def _image_to_data_url(image_path: Path) -> str:
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            max_size = 2048
            if max(image.size) > max_size:
                image.thumbnail((max_size, max_size))
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=92)
            encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        mime = "jpeg"
        return f"data:image/{mime};base64,{encoded}"

    def analyze_image(self, system_prompt: str, user_prompt: str, image_path: Path) -> dict[str, Any]:
        data_url = self._image_to_data_url(image_path)
        kwargs: dict[str, Any] = {}
        if self.config.reasoning_split:
            kwargs["extra_body"] = {"reasoning_split": True}
        return self._request_json_with_retry(
            request_name="analyze_image",
            request_payload=dict(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_prompt},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    },
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                **kwargs,
            ),
        )

    def analyze_text(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if self.config.reasoning_split:
            kwargs["extra_body"] = {"reasoning_split": True}
        return self._request_json_with_retry(
            request_name="analyze_text",
            request_payload=dict(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                **kwargs,
            ),
        )

    def _request_json_with_retry(self, request_name: str, request_payload: dict[str, Any]) -> dict[str, Any]:
        last_error: Exception | None = None
        for parse_attempt in range(1, self.max_retries + 2):
            response = self._request_with_retry(request_name=request_name, request_payload=request_payload)
            try:
                return self.parse_json(self._extract_response_text(response))
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if self.event_logger:
                    self.event_logger(
                        "model_response_parse_error",
                        request_name=request_name,
                        model=self.config.model,
                        attempt=parse_attempt,
                        error=str(exc),
                    )
                if parse_attempt > self.max_retries:
                    break
                if self.retry_backoff_seconds > 0:
                    time.sleep(self.retry_backoff_seconds * parse_attempt)
        assert last_error is not None
        raise last_error

    def _extract_response_text(self, response: Any) -> str:
        choices = getattr(response, "choices", None)
        if not choices:
            raise ValueError("模型响应缺少 choices")
        first_choice = choices[0]
        message = getattr(first_choice, "message", None)
        if message is None:
            raise ValueError("模型响应缺少 message")
        content = getattr(message, "content", None)
        if content is None:
            raise ValueError("模型响应缺少 content")
        return self.normalize_content(content)

    def _request_with_retry(self, request_name: str, request_payload: dict[str, Any]) -> Any:
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 2):
            try:
                if self.event_logger:
                    self.event_logger(
                        "model_request_start",
                        request_name=request_name,
                        model=self.config.model,
                        timeout_seconds=self.config.timeout_seconds,
                        attempt=attempt,
                    )
                response = self.client.chat.completions.create(**request_payload)
                if self.event_logger:
                    self.event_logger(
                        "model_request_success",
                        request_name=request_name,
                        model=self.config.model,
                        attempt=attempt,
                    )
                return response
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if self.event_logger:
                    self.event_logger(
                        "model_request_error",
                        request_name=request_name,
                        model=self.config.model,
                        attempt=attempt,
                        error=str(exc),
                    )
                if attempt > self.max_retries:
                    break
                if self.retry_backoff_seconds > 0:
                    time.sleep(self.retry_backoff_seconds * attempt)
        assert last_error is not None
        raise last_error
