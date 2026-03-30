"""Utilities for capturing LLM inputs per task/step and exporting them into result.json.

Design goals:
- Capture each LLM call's final messages (after any truncation)
- Exclude image bytes/base64 by default to avoid huge logs
- Easy to attach to step results

Controlled by env vars:
- CAPTURE_LLM_INPUTS=1                Enable capture
- CAPTURE_LLM_INPUTS_EXCLUDE_IMAGES=1 Exclude image parts (default)

If CAPTURE_LLM_INPUTS is not set, capture is enabled by default (safe because images are omitted).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional


def _exclude_images_enabled() -> bool:
    return os.getenv("CAPTURE_LLM_INPUTS_EXCLUDE_IMAGES", "1") == "1"


def capture_enabled() -> bool:
    # Default ON to avoid env propagation issues between subprocesses.
    return os.getenv("CAPTURE_LLM_INPUTS", "1") == "1"


def sanitize_messages(messages: Any) -> Any:
    """Return a JSON-serializable copy of messages with optional image removal."""
    if not _exclude_images_enabled():
        return messages

    if isinstance(messages, list):
        return [sanitize_messages(m) for m in messages]

    if isinstance(messages, dict):
        out: Dict[str, Any] = {}
        for k, v in messages.items():
            if k != "content":
                out[k] = sanitize_messages(v)
                continue

            content = v
            if isinstance(content, list):
                new_parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") in (
                        "image_url",
                        "image",
                        "input_image",
                    ):
                        new_parts.append({"type": part.get("type"), "omitted": True})
                    else:
                        new_parts.append(sanitize_messages(part))
                out["content"] = new_parts
            else:
                out["content"] = sanitize_messages(content)
        return out

    return messages


_BUFFER: List[Dict[str, Any]] = []
_CURRENT_STEP: Optional[int] = None


def set_current_step(step: Optional[int]) -> None:
    global _CURRENT_STEP
    _CURRENT_STEP = step


def reset_buffer() -> None:
    _BUFFER.clear()


def record_llm_input(*, provider_tag: str, messages: Any, meta: Optional[Dict[str, Any]] = None) -> None:
    if not capture_enabled():
        return
    item: Dict[str, Any] = {
        "step": _CURRENT_STEP,
        "provider": provider_tag,
        "messages": sanitize_messages(messages),
    }
    if meta:
        item["meta"] = meta
    _BUFFER.append(item)


def drain_for_step(step: int) -> List[Dict[str, Any]]:
    taken = [x for x in _BUFFER if x.get("step") == step]
    if taken:
        _BUFFER[:] = [x for x in _BUFFER if x.get("step") != step]
    return taken


def snapshot_all() -> List[Dict[str, Any]]:
    return list(_BUFFER)
