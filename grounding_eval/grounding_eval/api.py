from __future__ import annotations

import os
import time
from typing import Any

import httpx
from openai import OpenAI


def normalize_base_url(raw: str | None) -> str:
    value = (raw or "").strip()
    if value.endswith("/chat/completions"):
        value = value[: -len("/chat/completions")]
    return value.rstrip("/")


def base_url_from_env() -> str:
    return normalize_base_url(
        os.getenv("GROUNDING_BASE_URL") or os.getenv("OPENAI_BASE_URL") or os.getenv("API_URL")
    )


def create_client(*, base_url: str | None, api_key_env: str = "OPENAI_API_KEY") -> OpenAI:
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise RuntimeError(f"{api_key_env} is required")
    timeout_s = float(os.getenv("GROUNDING_TIMEOUT_SECONDS", "240"))
    return OpenAI(
        base_url=normalize_base_url(base_url) or None,
        api_key=api_key,
        timeout=httpx.Timeout(timeout_s, connect=60.0, read=timeout_s, write=60.0, pool=10.0),
    )


def chat_completion_with_retries(
    *,
    client: OpenAI,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    max_retries: int,
) -> str:
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
                max_tokens=max_tokens,
            )
            return completion.choices[0].message.content or ""
        except Exception as exc:  # pragma: no cover - exercised in live API runs
            last_error = exc
            time.sleep(min(2**attempt, 8))
    raise RuntimeError(str(last_error) if last_error else "chat completion failed")
