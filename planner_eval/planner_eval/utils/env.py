from __future__ import annotations

import os
from pathlib import Path


def _strip_wrapping_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def load_local_env() -> None:
    """Load a local .env file without overriding explicit environment variables."""

    candidate_paths: list[Path] = []
    cwd_env = Path.cwd().resolve() / ".env"
    project_env = Path(__file__).resolve().parents[2] / ".env"

    candidate_paths.append(cwd_env)
    if project_env != cwd_env:
        candidate_paths.append(project_env)

    seen_paths: set[Path] = set()
    for env_path in candidate_paths:
        env_path = env_path.resolve()
        if env_path in seen_paths or not env_path.exists():
            continue
        seen_paths.add(env_path)

        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            if not key or key in os.environ:
                continue
            os.environ[key] = _strip_wrapping_quotes(value.strip())

