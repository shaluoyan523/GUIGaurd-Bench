from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .api import base_url_from_env, normalize_base_url
from .data import build_samples
from .runner import run_evaluation


def _read_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _parse_key_value(values: list[str] | None) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"Expected KEY=VALUE, got {value!r}")
        key, raw = value.split("=", 1)
        result[key] = raw
    return result


def _list_value(args_value: list[str] | None, config: dict[str, Any], key: str, default: list[str]) -> list[str]:
    if args_value:
        return args_value
    value = config.get(key)
    if isinstance(value, list):
        return [str(v) for v in value]
    return default


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ScreenSpot-style GUI grounding evaluation.")
    parser.add_argument("--config", help="Optional JSON config file.")
    parser.add_argument("--manifest", help="Manifest JSONL with id/image/plan/action fields.")
    parser.add_argument("--boxes", nargs="+", help="One or more JSONL files with id/bbox_xyxy/action fields.")
    parser.add_argument(
        "--image-root",
        action="append",
        help="Image root in MASK=PATH form. Repeat for original/black/replace_llm/etc.",
    )
    parser.add_argument("--out-dir", help="Output directory.")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible base URL.")
    parser.add_argument("--api-key-env", default=None, help="Environment variable holding the API key.")
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--masks", nargs="+")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--start-id", type=int)
    parser.add_argument("--end-id", type=int)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--jpeg-quality", type=int)
    parser.add_argument("--max-retries", type=int)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--require-contiguous-ids", action="store_true")
    parser.add_argument("--norm-mode", action="append", help="Per-model override in MODEL=MODE form.")
    parser.add_argument("--parse-mode", action="append", help="Per-model override in MODEL=MODE form.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = _read_config(args.config)

    manifest = args.manifest or config.get("manifest")
    boxes = args.boxes or config.get("boxes")
    image_roots = _parse_key_value(args.image_root) or {
        str(k): str(v) for k, v in (config.get("image_roots") or {}).items()
    }
    if not manifest:
        raise RuntimeError("Set --manifest or config.manifest")
    if not boxes:
        raise RuntimeError("Set --boxes or config.boxes")
    if not image_roots:
        raise RuntimeError("Set at least one --image-root MASK=PATH or config.image_roots")

    models = _list_value(args.models, config, "models", [])
    masks = _list_value(args.masks, config, "masks", list(image_roots))
    if not models:
        raise RuntimeError("Set --models or config.models")

    out_dir = Path(args.out_dir or config.get("out_dir") or "outputs/grounding_eval")
    base_url = normalize_base_url(args.base_url or config.get("base_url") or base_url_from_env())
    if not base_url:
        raise RuntimeError("Set --base-url, config.base_url, GROUNDING_BASE_URL, or OPENAI_BASE_URL")

    samples = build_samples(
        manifest_path=Path(manifest),
        boxes_paths=[Path(p) for p in boxes],
        image_roots={mask: Path(path) for mask, path in image_roots.items()},
        start_id=args.start_id if args.start_id is not None else config.get("start_id"),
        end_id=args.end_id if args.end_id is not None else config.get("end_id"),
        require_contiguous_ids=bool(args.require_contiguous_ids or config.get("require_contiguous_ids", False)),
    )
    if args.limit or config.get("limit"):
        samples = samples[: int(args.limit or config["limit"])]

    summary = run_evaluation(
        samples=samples,
        models=models,
        masks=masks,
        out_dir=out_dir,
        base_url=base_url,
        api_key_env=args.api_key_env or config.get("api_key_env") or "OPENAI_API_KEY",
        workers=int(args.workers or config.get("workers", 4)),
        jpeg_quality=int(args.jpeg_quality or config.get("jpeg_quality", 90)),
        max_retries=int(args.max_retries or config.get("max_retries", 2)),
        max_tokens=int(args.max_tokens or config.get("max_tokens", 1024)),
        norm_modes={**(config.get("norm_modes") or {}), **_parse_key_value(args.norm_mode)},
        parse_modes={**(config.get("parse_modes") or {}), **_parse_key_value(args.parse_mode)},
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
