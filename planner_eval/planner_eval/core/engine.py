from __future__ import annotations

import json
import logging
import os
from typing import Any

import backoff
import httpx
import tiktoken
from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    BadRequestError,
    OpenAI,
    RateLimitError,
)

from planner_eval.core.llm_trace import record_llm_input, sanitize_messages


RETRYABLE_EXCEPTIONS = (
    APIConnectionError,
    APIError,
    RateLimitError,
    APITimeoutError,
    httpx.ReadTimeout,
)


def _is_context_length_error(error: Exception) -> bool:
    message = str(error)
    needles = [
        "longer than the maximum model length",
        "maximum context length",
        "context length",
        "max_model_len",
        "Please reduce the length of the messages",
    ]
    return any(needle in message for needle in needles)


class OpenAIChatEngine:
    """OpenAI-compatible chat engine with conservative prompt truncation."""

    DEFAULT_MAX_CONTEXT_TOKENS = 16384

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str,
        rate_limit: int = -1,
        temperature: float | None = None,
        organization: str | None = None,
        **_: Any,
    ) -> None:
        if not model:
            raise ValueError("model must be provided")
        self.model = model
        self.base_url = base_url or None
        self.api_key = api_key or None
        self.organization = organization or None
        self.request_interval = 0 if rate_limit == -1 else 60.0 / rate_limit
        self.temperature = temperature
        self._client: OpenAI | None = None
        self._logger = logging.getLogger(__name__)

    def _get_client(self) -> OpenAI:
        if self._client is None:
            api_key = self.api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "Missing API key. Set OPENAI_API_KEY or provide api_key explicitly."
                )
            timeout_s = float(os.getenv("LMM_API_TIMEOUT_SECONDS", "600"))
            timeout = httpx.Timeout(
                timeout_s,
                connect=60.0,
                read=timeout_s,
                write=30.0,
                pool=10.0,
            )
            kwargs: dict[str, Any] = {
                "api_key": api_key,
                "timeout": timeout,
            }
            if self.organization:
                kwargs["organization"] = self.organization
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = OpenAI(**kwargs)
        return self._client

    def _get_max_context_tokens(self) -> int:
        try:
            return int(
                os.getenv(
                    "LMM_MAX_CONTEXT_TOKENS",
                    str(self.DEFAULT_MAX_CONTEXT_TOKENS),
                )
            )
        except Exception:
            return self.DEFAULT_MAX_CONTEXT_TOKENS

    def _get_reserved_output_tokens(self, max_new_tokens: int | None) -> int:
        if max_new_tokens is None:
            return int(os.getenv("LMM_RESERVED_OUTPUT_TOKENS", "1024"))
        return int(max_new_tokens)

    def _count_message_tokens(self, messages: Any) -> int:
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
        except Exception:
            encoding = tiktoken.encoding_for_model("gpt-4o")

        def _count_any(value: Any) -> int:
            if value is None:
                return 0
            if isinstance(value, str):
                return len(encoding.encode(value))
            if isinstance(value, dict):
                return sum(_count_any(v) for v in value.values())
            if isinstance(value, list):
                return sum(_count_any(v) for v in value)
            return len(encoding.encode(str(value)))

        return _count_any(messages)

    def _truncate_messages_to_fit(
        self,
        messages: list[dict[str, Any]],
        *,
        max_new_tokens: int | None = None,
    ) -> list[dict[str, Any]]:
        max_ctx = self._get_max_context_tokens()
        reserved = self._get_reserved_output_tokens(max_new_tokens)
        safety = int(os.getenv("LMM_TRUNCATE_SAFETY_MARGIN", "512"))
        budget = max_ctx - reserved - safety
        if budget <= 0:
            budget = max(1, max_ctx - safety)

        if self._count_message_tokens(messages) <= budget:
            return messages

        if not messages:
            return messages

        kept: list[dict[str, Any]] = []
        start_idx = 0
        if messages and messages[0].get("role") == "system":
            kept.append(messages[0])
            start_idx = 1

        tail = messages[start_idx:]
        if not tail:
            return kept

        min_tail = int(os.getenv("LMM_MIN_TAIL_MESSAGES", "4"))
        tail_kept = tail[-min_tail:] if len(tail) >= min_tail else tail
        candidate = kept + tail_kept

        while len(candidate) > len(kept) + 1 and self._count_message_tokens(candidate) > budget:
            tail_kept.pop(0)
            candidate = kept + tail_kept

        if self._count_message_tokens(candidate) > budget:
            last = candidate[-1]
            content = last.get("content")
            encoding = tiktoken.get_encoding("cl100k_base")
            if isinstance(content, str):
                tokens = encoding.encode(content)[-max(1, budget // 2) :]
                last["content"] = encoding.decode(tokens)
            elif isinstance(content, list):
                new_parts = []
                for part in content:
                    if (
                        isinstance(part, dict)
                        and part.get("type") == "text"
                        and isinstance(part.get("text"), str)
                    ):
                        tokens = encoding.encode(part["text"])[-max(1, budget // 2) :]
                        new_part = dict(part)
                        new_part["text"] = encoding.decode(tokens)
                        new_parts.append(new_part)
                    else:
                        new_parts.append(part)
                last["content"] = new_parts

        try:
            record_llm_input(
                provider_tag="OpenAIChatEngine",
                messages=sanitize_messages(candidate),
                meta={"stage": "sent_after_truncation"},
            )
        except Exception:
            pass

        return candidate

    def _serialize_completion(self, completion: Any) -> str | None:
        if completion is None:
            return None
        try:
            if hasattr(completion, "model_dump_json"):
                return completion.model_dump_json()
            if hasattr(completion, "model_dump"):
                return json.dumps(completion.model_dump(), ensure_ascii=False)
            if hasattr(completion, "to_dict"):
                return json.dumps(completion.to_dict(), ensure_ascii=False)
            if isinstance(completion, dict):
                return json.dumps(completion, ensure_ascii=False)
        except Exception:
            return None
        return repr(completion)

    def _log_llm_failure(self, error: Exception) -> None:
        payload: Any = None
        response = getattr(error, "response", None)
        if response is not None:
            try:
                if hasattr(response, "json"):
                    payload = response.json()
                elif hasattr(response, "text"):
                    payload = response.text
            except Exception:
                payload = getattr(response, "text", None)

        if payload is None:
            payload = str(error)

        try:
            rendered = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
        except Exception:
            rendered = str(payload)
        self._logger.error("LLM_RESPONSE_ERROR(%s): %s", self.model, rendered)

    @backoff.on_exception(backoff.expo, RETRYABLE_EXCEPTIONS)
    def _create_with_auto_truncate(
        self,
        *,
        create_fn,
        messages: list[dict[str, Any]],
        max_new_tokens: int | None = None,
        **kwargs: Any,
    ):
        try:
            return create_fn(messages=messages, **kwargs)
        except BadRequestError as error:
            if not _is_context_length_error(error):
                raise
            self._logger.warning("Context length error detected, retrying with truncation.")
            truncated = self._truncate_messages_to_fit(
                messages,
                max_new_tokens=max_new_tokens,
            )
            if truncated == messages:
                raise
            return create_fn(messages=truncated, **kwargs)
        except Exception as error:
            self._log_llm_failure(error)
            if not _is_context_length_error(error):
                raise
            self._logger.warning("Context length error detected, retrying with truncation.")
            truncated = self._truncate_messages_to_fit(
                messages,
                max_new_tokens=max_new_tokens,
            )
            if truncated == messages:
                raise
            return create_fn(messages=truncated, **kwargs)

    @backoff.on_exception(backoff.expo, RETRYABLE_EXCEPTIONS)
    def generate(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.0,
        max_new_tokens: int | None = None,
        **kwargs: Any,
    ) -> str:
        messages = self._truncate_messages_to_fit(messages, max_new_tokens=max_new_tokens)
        if os.getenv("LOG_LLM_INPUTS", "0") == "1":
            try:
                self._logger.info(
                    "LLM_INPUT(OpenAICompatible): %s",
                    json.dumps(sanitize_messages(messages), ensure_ascii=False),
                )
            except Exception:
                pass

        completion = self._create_with_auto_truncate(
            create_fn=lambda **call_kwargs: self._get_client().chat.completions.create(
                model=self.model,
                temperature=(
                    temperature if self.temperature is None else self.temperature
                ),
                **kwargs,
                **call_kwargs,
            ),
            messages=messages,
            max_new_tokens=max_new_tokens,
        )

        try:
            serialized = self._serialize_completion(completion)
            if serialized:
                self._logger.info("LLM_RESPONSE(%s): %s", self.model, serialized)
        except Exception:
            pass

        return completion.choices[0].message.content

    def generate_with_thinking(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.0,
        max_new_tokens: int | None = None,
        **kwargs: Any,
    ) -> str:
        """Compatibility shim for models that expose reasoning but use the same API."""
        return self.generate(
            messages,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )
