from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .api import create_client, chat_completion_with_retries
from .io_utils import append_jsonl, image_to_data_url, load_jsonl, write_json
from .metrics import iou, normalize_point, point_in_bbox
from .parsing import parse_prediction
from .prompting import infer_norm_mode, infer_parse_mode, prompt_for
from .types import Sample


def sample_from_json(row: dict[str, Any]) -> Sample:
    return Sample(
        id=int(row["id"]),
        image_name=str(row["image_name"]),
        plan=str(row.get("plan", "")),
        action=str(row.get("action", "")),
        target=str(row.get("target", "")),
        bbox_xyxy=tuple(float(v) for v in row["bbox_xyxy"]),  # type: ignore[arg-type]
        target_visible=bool(row.get("target_visible", True)),
        image_paths={str(k): str(v) for k, v in row["image_paths"].items()},
    )


def run_one(task: dict[str, Any]) -> dict[str, Any]:
    sample = sample_from_json(task["sample"])
    model = str(task["model"])
    mask = str(task["mask"])
    base_url = str(task.get("base_url") or "")
    api_key_env = str(task.get("api_key_env") or "OPENAI_API_KEY")
    quality = int(task["jpeg_quality"])
    max_retries = int(task["max_retries"])
    max_tokens = int(task["max_tokens"])
    norm_mode = str(task.get("norm_mode") or infer_norm_mode(model))
    parse_mode = str(task.get("parse_mode") or infer_parse_mode(model))

    client = create_client(base_url=base_url, api_key_env=api_key_env)
    image_url, width, height = image_to_data_url(sample.image_paths[mask], quality)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_for(model, sample, width=width, height=height, norm_mode=norm_mode)},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        }
    ]

    start = time.time()
    response_text = ""
    parsed: dict[str, Any] | None = None
    error: str | None = None
    for _ in range(max_retries):
        try:
            response_text = chat_completion_with_retries(
                client=client,
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                max_retries=1,
            )
            parsed = parse_prediction(response_text, parse_mode)
            if parsed["ok"]:
                break
            error = parsed.get("error")
        except Exception as exc:  # pragma: no cover - exercised in live API runs
            error = str(exc)

    if parsed is None or not parsed["ok"]:
        parsed = parse_prediction(response_text, parse_mode) if response_text else {
            "ok": False,
            "error": error,
            "raw_bbox": None,
            "raw_point": None,
            "prediction_kind": None,
        }

    pred_point, pred_bbox = normalize_point(
        parsed.get("raw_point"),
        parsed.get("raw_bbox"),
        width=width,
        height=height,
        mode=norm_mode,
    )
    correct = bool(sample.target_visible and parsed["ok"] and point_in_bbox(pred_point, sample.bbox_xyxy))

    return {
        "id": sample.id,
        "image_name": sample.image_name,
        "mask": mask,
        "model": model,
        "norm_mode": norm_mode,
        "parse_mode": parse_mode,
        "image_size": [width, height],
        "target": sample.target,
        "action": sample.action,
        "gt_bbox_xyxy": list(sample.bbox_xyxy),
        "target_visible": sample.target_visible,
        "raw_response": response_text,
        "parse_ok": bool(parsed["ok"]),
        "parse_error": parsed.get("error"),
        "prediction_kind": parsed.get("prediction_kind"),
        "raw_bbox": parsed.get("raw_bbox"),
        "raw_point": parsed.get("raw_point"),
        "pred_bbox_xyxy": pred_bbox,
        "pred_point_xy": pred_point,
        "point_in_gt_bbox": correct,
        "iou": iou(pred_bbox, sample.bbox_xyxy),
        "latency_s": round(time.time() - start, 3),
    }


def read_done_ids(path: Path) -> set[int]:
    done: set[int] = set()
    if not path.exists():
        return done
    for row in load_jsonl(path):
        if row.get("parse_ok"):
            done.add(int(row["id"]))
    return done


def summarize_results(out_dir: Path, models: list[str], masks: list[str]) -> dict[str, Any]:
    summary: dict[str, Any] = {"models": {}}
    for model in models:
        model_summary: dict[str, Any] = {}
        for mask in masks:
            path = out_dir / model / f"{mask}.jsonl"
            rows = load_jsonl(path) if path.exists() else []
            evaluated = [r for r in rows if r.get("target_visible")]
            correct = sum(1 for r in evaluated if r.get("point_in_gt_bbox"))
            parse_ok = sum(1 for r in evaluated if r.get("parse_ok"))
            prediction_kinds: dict[str, int] = {}
            for row in evaluated:
                kind = row.get("prediction_kind") or "none"
                prediction_kinds[kind] = prediction_kinds.get(kind, 0) + 1
            model_summary[mask] = {
                "count": len(rows),
                "evaluated": len(evaluated),
                "correct": correct,
                "point_accuracy": correct / len(evaluated) if evaluated else 0.0,
                "parse_ok": parse_ok,
                "parse_rate": parse_ok / len(evaluated) if evaluated else 0.0,
                "prediction_kinds": dict(sorted(prediction_kinds.items())),
            }
        summary["models"][model] = model_summary
    return summary


def run_evaluation(
    *,
    samples: list[Sample],
    models: list[str],
    masks: list[str],
    out_dir: Path,
    base_url: str,
    api_key_env: str,
    workers: int,
    jpeg_quality: int,
    max_retries: int,
    max_tokens: int,
    norm_modes: dict[str, str] | None = None,
    parse_modes: dict[str, str] | None = None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    norm_modes = norm_modes or {}
    parse_modes = parse_modes or {}

    serializable_samples = [sample.to_json() for sample in samples]
    write_json(out_dir / "samples.json", serializable_samples)

    done_ids_by_output: dict[tuple[str, str], set[int]] = {}
    for model in models:
        for mask in masks:
            done_ids_by_output[(model, mask)] = read_done_ids(out_dir / model / f"{mask}.jsonl")

    tasks: list[dict[str, Any]] = []
    for sample in samples:
        for model in models:
            for mask in masks:
                if sample.id in done_ids_by_output[(model, mask)]:
                    continue
                tasks.append(
                    {
                        "model": model,
                        "mask": mask,
                        "base_url": base_url,
                        "api_key_env": api_key_env,
                        "jpeg_quality": jpeg_quality,
                        "max_retries": max_retries,
                        "max_tokens": max_tokens,
                        "norm_mode": norm_modes.get(model),
                        "parse_mode": parse_modes.get(model),
                        "sample": sample.to_json(),
                    }
                )

    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "sample_count": len(samples),
                "pending_tasks": len(tasks),
                "models": models,
                "masks": masks,
                "workers": workers,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    started = time.time()
    completed = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_one, task) for task in tasks]
        for future in as_completed(futures):
            row = future.result()
            append_jsonl(out_dir / row["model"] / f"{row['mask']}.jsonl", row)
            completed += 1
            if completed % 10 == 0 or completed == len(tasks):
                print(
                    json.dumps(
                        {
                            "completed": completed,
                            "pending": len(tasks) - completed,
                            "elapsed_s": round(time.time() - started, 1),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    summary = summarize_results(out_dir, models, masks)
    summary["out_dir"] = str(out_dir)
    summary["sample_count"] = len(samples)
    summary["elapsed_s"] = round(time.time() - started, 3)
    write_json(out_dir / "summary.json", summary)
    return summary
