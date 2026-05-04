from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from .io_utils import load_jsonl
from .types import BBox, Sample


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}


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
    value = row.get("bbox_xyxy") or row.get("bbox") or row.get("box") or row.get("points")
    if isinstance(value, dict) and all(k in value for k in ["x1", "y1", "x2", "y2"]):
        value = [value["x1"], value["y1"], value["x2"], value["y2"]]
    if not isinstance(value, list) or len(value) < 4:
        return None
    x1, y1, x2, y2 = (float(v) for v in value[:4])
    return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)


def _id_range(start_id: int | None, end_id: int | None) -> set[int] | None:
    if start_id is None and end_id is None:
        return None
    start = start_id if start_id is not None else -10**18
    end = end_id if end_id is not None else 10**18
    return set(range(start, end + 1)) if end - start < 2_000_000 else None


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _labels_from_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("items", "data", "annotations", "labels"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    raise ValueError("Privacy label file must be a JSON list or a dict with items/data/annotations/labels")


def _default_privacy_label_path(example_root: Path) -> Path:
    candidates = [
        example_root / "image_privacy_labels_example.json",
        example_root / "image_privacy_labels.json",
        example_root / "privacy_labels.json",
        example_root / "annotations.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    names = ", ".join(path.name for path in candidates)
    raise FileNotFoundError(f"No privacy label file found under {example_root}. Tried: {names}")


def _parse_label_info(info: str) -> tuple[str | None, str | None, str]:
    parts = [part for part in info.replace("\\", "/").split("/") if part]
    filename = parts[-1] if parts else ""
    for platform in ("Android", "PC"):
        if platform in parts:
            index = parts.index(platform)
            task = parts[index + 1] if index + 1 < len(parts) else None
            return platform, task, filename
    return None, None, filename


def _iter_example_images(example_root: Path) -> list[Path]:
    images: list[Path] = []
    for platform in ("Android", "PC"):
        platform_root = example_root / platform
        if not platform_root.exists():
            continue
        for root, _dirnames, filenames in os.walk(platform_root):
            for filename in filenames:
                path = Path(root) / filename
                if path.suffix.lower() in IMAGE_EXTENSIONS:
                    images.append(path)
    return sorted(images)


def _example_image_index(example_root: Path) -> tuple[dict[tuple[str, str, str], Path], dict[str, list[Path]]]:
    by_platform_task_name: dict[tuple[str, str, str], Path] = {}
    by_name: dict[str, list[Path]] = {}
    for image_path in _iter_example_images(example_root):
        rel_parts = image_path.relative_to(example_root).parts
        if len(rel_parts) < 3:
            continue
        platform = rel_parts[0]
        task = rel_parts[1]
        by_platform_task_name[(platform, task, image_path.name)] = image_path
        by_name.setdefault(image_path.name, []).append(image_path)
    return by_platform_task_name, by_name


def _resolve_labeled_image(
    *,
    label_item: dict[str, Any],
    example_root: Path,
    by_platform_task_name: dict[tuple[str, str, str], Path],
    by_name: dict[str, list[Path]],
) -> Path | None:
    info = _first_non_empty(label_item, ["info", "image", "image_path", "path", "url"])
    platform, task, filename = _parse_label_info(info)

    if platform and task and filename:
        image_path = by_platform_task_name.get((platform, task, filename))
        if image_path is not None:
            return image_path

    if info:
        normalized_info = info.replace("\\", "/")
        for image_path in _iter_example_images(example_root):
            rel = image_path.relative_to(example_root).as_posix()
            if normalized_info.endswith(rel) or normalized_info.endswith(image_path.name):
                return image_path

    matches = by_name.get(filename, []) if filename else []
    if len(matches) == 1:
        return matches[0]
    return None


def _task_dir_for_image(example_root: Path, image_path: Path) -> Path:
    rel_parts = image_path.relative_to(example_root).parts
    if len(rel_parts) >= 3:
        return example_root / rel_parts[0] / rel_parts[1]
    return image_path.parent


def _derive_instruction(task_dir: Path, fallback: str) -> str:
    instruction_file = task_dir / "instruction.txt"
    if instruction_file.exists():
        instruction = instruction_file.read_text(encoding="utf-8").strip()
        if instruction:
            return instruction

    task_result_file = task_dir / "task_result.json"
    if task_result_file.exists():
        try:
            payload = _load_json(task_result_file)
            for key in ("goal", "instruction", "task", "query"):
                value = payload.get(key) if isinstance(payload, dict) else None
                if isinstance(value, str) and value.strip():
                    return value.strip()
        except Exception:
            pass

    traj_file = task_dir / "traj.jsonl"
    if traj_file.exists():
        try:
            for row in load_jsonl(traj_file):
                for key in ("goal", "instruction", "task", "query"):
                    value = row.get(key)
                    if isinstance(value, str) and value.strip():
                        return value.strip()
        except Exception:
            pass

    return fallback


def _label_attr(label: dict[str, Any]) -> dict[str, Any]:
    attr = label.get("attr")
    if not isinstance(attr, dict):
        return {}
    nested = attr.get("attr")
    merged = dict(nested) if isinstance(nested, dict) else {}
    merged.update(attr)
    merged.pop("attr", None)
    return merged


def _label_target(label: dict[str, Any]) -> str:
    attr = _label_attr(label)
    text = str(attr.get("ocrResult") or attr.get("text") or attr.get("value") or "").strip()
    label_name = str(label.get("label") or "").strip()
    category = str(attr.get("分类") or attr.get("category") or "").strip()
    parts = [part for part in (label_name, category, text) if part]
    return " | ".join(parts) if parts else "annotated privacy region"


def _item_label_rows(item: dict[str, Any]) -> list[dict[str, Any]]:
    labels = item.get("labels")
    if isinstance(labels, list):
        return [label for label in labels if isinstance(label, dict)]
    return [item]


def build_samples_from_example(
    *,
    example_root: Path,
    privacy_labels_path: Path | None = None,
    start_id: int | None = None,
    end_id: int | None = None,
) -> list[Sample]:
    """Build grounding samples directly from the public `dataset_example` layout.

    This convenience loader treats each region-level privacy box as one
    ScreenSpot-style target. Full click-grounding experiments should still use
    the manifest/box loader when human next-action click boxes are available.
    """

    example_root = example_root.resolve()
    if not example_root.exists():
        raise FileNotFoundError(f"Example root does not exist: {example_root}")

    labels_path = privacy_labels_path.resolve() if privacy_labels_path else _default_privacy_label_path(example_root)
    items = _labels_from_payload(_load_json(labels_path))
    by_platform_task_name, by_name = _example_image_index(example_root)

    samples: list[Sample] = []
    sample_id = 0
    for item in items:
        image_path = _resolve_labeled_image(
            label_item=item,
            example_root=example_root,
            by_platform_task_name=by_platform_task_name,
            by_name=by_name,
        )
        if image_path is None:
            continue

        task_dir = _task_dir_for_image(example_root, image_path)
        task_name = task_dir.name
        plan = _derive_instruction(task_dir, task_name)
        image_name = image_path.relative_to(example_root).as_posix()

        for label in _item_label_rows(item):
            bbox = _bbox_from_row(label)
            if bbox is None:
                continue
            sample_id += 1
            if start_id is not None and sample_id < start_id:
                continue
            if end_id is not None and sample_id > end_id:
                break
            target = _label_target(label)
            action = f"# Click: {target}"
            samples.append(
                Sample(
                    id=sample_id,
                    image_name=image_name,
                    plan=plan,
                    action=action,
                    target=target,
                    bbox_xyxy=bbox,
                    target_visible=True,
                    image_paths={"original": str(image_path)},
                )
            )
        if end_id is not None and sample_id > end_id:
            break

    if not samples:
        raise RuntimeError(
            "No example grounding samples were built. Check that the example root "
            "contains Android/PC screenshots and matching privacy label boxes."
        )
    return samples


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
