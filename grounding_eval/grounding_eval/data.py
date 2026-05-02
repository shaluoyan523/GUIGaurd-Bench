from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .io_utils import load_jsonl
from .types import BBox, Sample


def clean_target(action: str) -> str:
    target = action.strip()
    target = re.sub(r"^#\s*Click\s*:\s*", "", target, flags=re.I)
    return target.strip()


def _first_non_empty(row: dict[str, Any], keys: list[str], default: str = "") -> str:
    for key in keys:
        value = row.get(key)
        if value:
            return str(value)
    return default


def _manifest_image_name(row: dict[str, Any]) -> str:
    return _first_non_empty(row, ["copied_image_name", "image_name", "image", "screenshot"])


def _manifest_action(row: dict[str, Any]) -> str:
    actions = row.get("actions")
    if isinstance(actions, list) and actions:
        return str(actions[0])
    return _first_non_empty(row, ["action", "target_action", "instruction"])


def _bbox_from_row(row: dict[str, Any]) -> BBox | None:
    value = row.get("bbox_xyxy") or row.get("bbox") or row.get("box")
    if isinstance(value, dict) and all(k in value for k in ["x1", "y1", "x2", "y2"]):
        value = [value["x1"], value["y1"], value["x2"], value["y2"]]
    if not isinstance(value, list) or len(value) < 4:
        return None
    return tuple(float(v) for v in value[:4])  # type: ignore[return-value]


def _id_range(start_id: int | None, end_id: int | None) -> set[int] | None:
    if start_id is None and end_id is None:
        return None
    start = start_id if start_id is not None else -10**18
    end = end_id if end_id is not None else 10**18
    return set(range(start, end + 1)) if end - start < 2_000_000 else None


def build_samples(
    *,
    manifest_path: Path,
    boxes_paths: list[Path],
    image_roots: dict[str, Path],
    start_id: int | None = None,
    end_id: int | None = None,
    require_contiguous_ids: bool = False,
) -> list[Sample]:
    """Build samples from a manifest JSONL and one or more bbox JSONL files.

    Expected manifest fields are intentionally permissive:
    `id`, `copied_image_name` or `image_name`, `plan`, and either `actions`
    or `action`. Box rows should provide `id`, `bbox_xyxy`, and optionally
    `action` plus `target_visible`.
    """

    id_filter = _id_range(start_id, end_id)
    manifest: dict[int, dict[str, Any]] = {}
    for row in load_jsonl(manifest_path):
        row_id = int(row["id"])
        if id_filter is not None and row_id not in id_filter:
            continue
        if start_id is not None and row_id < start_id:
            continue
        if end_id is not None and row_id > end_id:
            continue
        manifest[row_id] = row

    boxes: dict[int, dict[str, Any]] = {}
    for path in boxes_paths:
        for row in load_jsonl(path):
            row_id = int(row["id"])
            if row_id in manifest:
                boxes[row_id] = row

    samples: list[Sample] = []
    for row_id in sorted(manifest):
        if row_id not in boxes:
            continue
        m = manifest[row_id]
        b = boxes[row_id]
        image_name = _manifest_image_name(m)
        if not image_name:
            raise ValueError(f"Manifest row {row_id} has no image name")
        action = str(b.get("action") or _manifest_action(m))
        bbox = _bbox_from_row(b)
        visible = bool(b.get("target_visible", True)) and bbox is not None
        bbox = bbox or (0.0, 0.0, 0.0, 0.0)
        samples.append(
            Sample(
                id=row_id,
                image_name=image_name,
                plan=str(m.get("plan", "")),
                action=action,
                target=clean_target(action),
                bbox_xyxy=bbox,
                target_visible=visible,
                image_paths={mask: str(root / image_name) for mask, root in image_roots.items()},
            )
        )

    if require_contiguous_ids and start_id is not None and end_id is not None:
        expected = set(range(start_id, end_id + 1))
        actual = {sample.id for sample in samples}
        missing = sorted(expected - actual)
        if missing:
            raise RuntimeError(f"Missing samples for ids: {missing[:20]} ... total={len(missing)}")
    return samples
