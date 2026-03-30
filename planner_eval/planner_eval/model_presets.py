from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any


BUILTIN_PRESETS: dict[str, dict[str, Any]] = {
    "gpt-5": {
        "provider": "openai",
        "model": "gpt-5",
        "api_key_env": "OPENAI_API_KEY",
        "base_url": "",
    },
    "gpt-4.1-mini": {
        "provider": "openai",
        "model": "gpt-4.1-mini",
        "api_key_env": "OPENAI_API_KEY",
        "base_url": "",
    },
}


def _load_external_presets(preset_file: str | None) -> dict[str, dict[str, Any]]:
    if not preset_file:
        return {}
    path = Path(preset_file)
    if not path.exists():
        raise FileNotFoundError(f"Preset file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Preset file must contain a JSON object keyed by preset name.")
    return data


def _resolve_field(spec: dict[str, Any], key: str, override: Any = None) -> Any:
    if override not in (None, ""):
        return override
    env_key = spec.get(f"{key}_env")
    if env_key:
        env_value = os.getenv(env_key)
        if env_value not in (None, ""):
            return env_value
    return spec.get(key)


def resolve_model_config(
    preset_name: str,
    *,
    preset_file: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    provider: str | None = None,
    temperature: float | None = None,
) -> dict[str, Any]:
    presets = deepcopy(BUILTIN_PRESETS)
    presets.update(_load_external_presets(preset_file))
    if preset_name not in presets:
        raise KeyError(f"Unknown model preset: {preset_name}")

    spec = presets[preset_name]
    resolved = {
        "provider": provider or spec.get("provider", "openai"),
        "model": model or spec.get("model"),
        "api_key": _resolve_field(spec, "api_key", api_key),
        "base_url": _resolve_field(spec, "base_url", base_url) or "",
        "temperature": temperature
        if temperature is not None
        else spec.get("temperature"),
    }

    if not resolved["model"]:
        raise ValueError(f"Preset {preset_name!r} does not define a model name.")
    return resolved
