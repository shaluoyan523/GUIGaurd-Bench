from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from planner_eval.judges.batch_evaluate_parameterized import run_batch_evaluation
from planner_eval.model_presets import resolve_model_config
from planner_eval.trajectory import evaluate_directory
from planner_eval.utils.env import load_local_env


MASK_TYPES = (
    "output_black_mask",
    "output_mosaic_mask",
    "output_randblocks_mask",
    "output_replace_llm_mask",
)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}

logger = logging.getLogger(__name__)


def _redact_model_config(model_config: dict[str, Any]) -> dict[str, Any]:
    redacted = dict(model_config)
    if redacted.get("api_key"):
        redacted["api_key"] = "<redacted>"
    return redacted


def _safe_model_label(model_label: str) -> str:
    return model_label.replace("\\", "_").replace("/", "_").replace(":", "_")


@dataclass
class TaskTarget:
    name: str
    output_stem: str
    instruction: str
    screenshot_dir: Path


def configure_logging(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)


def derive_instruction(task_dir: Path, fallback: str) -> str:
    instruction_file = task_dir / "instruction.txt"
    if instruction_file.exists():
        text = instruction_file.read_text(encoding="utf-8").strip()
        if text:
            return text

    task_result_file = task_dir / "task_result.json"
    if task_result_file.exists():
        try:
            payload = json.loads(task_result_file.read_text(encoding="utf-8"))
            goal = payload.get("goal")
            if isinstance(goal, str) and goal.strip():
                return goal.strip()
        except Exception:
            pass

    if "_" in fallback:
        parts = fallback.split("_", 2)
        if len(parts) >= 3 and parts[0].isdigit():
            return parts[2].replace("_", " ")
    return fallback


def _normalize_task_name(task_name: str) -> str:
    return task_name.replace("\\", "/").strip("/")


def _has_image_files(filenames: list[str]) -> bool:
    return any(Path(filename).suffix.lower() in IMAGE_EXTENSIONS for filename in filenames)


@lru_cache(maxsize=None)
def _discover_task_entries(base_dir_str: str, platform: str) -> tuple[tuple[str, str], ...]:
    base_dir = Path(base_dir_str)
    if not base_dir.exists():
        return ()

    task_entries: list[tuple[str, str]] = []
    for root_str, dirnames, filenames in os.walk(base_dir):
        root = Path(root_str)

        is_task_dir = False
        if platform == "android":
            is_task_dir = "images" in dirnames
        else:
            is_task_dir = _has_image_files(filenames)

        if is_task_dir:
            task_id = root.relative_to(base_dir).as_posix()
            task_entries.append((task_id, str(root)))
            dirnames[:] = []

    task_entries.sort(key=lambda item: item[0])
    return tuple(task_entries)


def _build_task_index(base_dir: Path, platform: str) -> dict[str, Path]:
    base_dir = base_dir.resolve()
    return {
        task_id: Path(task_dir)
        for task_id, task_dir in _discover_task_entries(str(base_dir), platform)
    }


def _resolve_task_identifier(base_dir: Path, platform: str, task_name: str) -> str | None:
    normalized = _normalize_task_name(task_name)
    task_index = _build_task_index(base_dir, platform)
    if normalized in task_index:
        return normalized

    basename_matches = [
        task_id for task_id in task_index if Path(task_id).name == normalized
    ]
    if len(basename_matches) == 1:
        return basename_matches[0]
    if len(basename_matches) > 1:
        match_preview = ", ".join(sorted(basename_matches)[:5])
        logger.warning(
            "Ambiguous task name %r under %s. Use a relative path instead. Matches: %s",
            task_name,
            base_dir,
            match_preview,
        )
        return None

    logger.warning("Task %r not found under %s", task_name, base_dir)
    return None


def _task_output_stem(task_id: str) -> str:
    return _normalize_task_name(task_id).replace("/", "__")


def resolve_platform_base(base_dir: Path, platform: str) -> Path:
    if not base_dir.exists():
        return base_dir

    platform_candidates = (
        ("Android", "android", "Android_public", "android_public", "Android_private", "android_private")
        if platform == "android"
        else ("PC", "pc", "PC_public", "pc_public", "PC_private", "pc_private")
    )
    for candidate_name in platform_candidates:
        candidate = base_dir / candidate_name
        if candidate.exists() and _discover_task_entries(str(candidate.resolve()), platform):
            return candidate.resolve()

    if _discover_task_entries(str(base_dir.resolve()), platform):
        return base_dir.resolve()
    return base_dir.resolve()


def discover_task_names(base_dir: Path, platform: str, explicit_tasks: list[str] | None) -> list[str]:
    if not base_dir.exists():
        return []
    if explicit_tasks:
        resolved_tasks = []
        for task_name in explicit_tasks:
            task_id = _resolve_task_identifier(base_dir, platform, task_name)
            if task_id is not None:
                resolved_tasks.append(task_id)
        return resolved_tasks
    return sorted(_build_task_index(base_dir, platform))


def resolve_task_target(base_dir: Path, platform: str, task_name: str) -> TaskTarget | None:
    task_id = _resolve_task_identifier(base_dir, platform, task_name)
    if task_id is None:
        return None
    task_dir = _build_task_index(base_dir, platform)[task_id]

    screenshot_dir = task_dir / "images" if platform == "android" else task_dir
    if not screenshot_dir.exists():
        logger.warning("Screenshot directory does not exist: %s", screenshot_dir)
        return None

    instruction = derive_instruction(task_dir, Path(task_id).name)
    return TaskTarget(
        name=task_id,
        output_stem=_task_output_stem(task_id),
        instruction=instruction,
        screenshot_dir=screenshot_dir,
    )


def append_screenshot_log(output_subdir: Path, screenshot_dir: Path, mask_type: str | None = None) -> None:
    log_file = output_subdir / "screenshot_paths.log"
    prefix = f"Mask类型: {mask_type} | " if mask_type else ""
    with log_file.open("a", encoding="utf-8") as handle:
        handle.write(f"{prefix}截图路径: {screenshot_dir}\n")


def run_task(
    *,
    model_config: dict[str, Any],
    run_dir: Path,
    output_subdir_name: str,
    platform: str,
    task: TaskTarget,
    max_steps: int = 30,
    max_trajectory_length: int = 8,
    memory_mode: str = "auto",
    enable_reflection: bool = True,
    mask_type: str | None = None,
) -> Path:
    output_subdir = run_dir / output_subdir_name
    output_subdir.mkdir(parents=True, exist_ok=True)
    output_file = output_subdir / f"{task.output_stem}_result.json"

    logger.info("")
    logger.info("=" * 64)
    logger.info("平台        : %s", platform)
    logger.info("任务        : %s", task.name)
    logger.info("指令        : %s", task.instruction)
    logger.info("截图目录    : %s", task.screenshot_dir)
    if mask_type:
        logger.info("Mask类型    : %s", mask_type)
    logger.info("=" * 64)

    results = evaluate_directory(
        screenshot_dir=str(task.screenshot_dir),
        instruction=task.instruction,
        provider=model_config["provider"],
        model=model_config["model"],
        model_url=model_config.get("base_url", ""),
        model_api_key=model_config.get("api_key", ""),
        model_temperature=model_config.get("temperature"),
        max_steps=max_steps,
        platform="linux",
        max_trajectory_length=max_trajectory_length,
        memory_mode=memory_mode,
        enable_reflection=enable_reflection,
        output_file=output_file,
    )

    append_screenshot_log(output_subdir, task.screenshot_dir, mask_type=mask_type)
    logger.info(
        "任务完成，状态: %s",
        "completed" if results["completed"] else "failed" if results["failed"] else "incomplete",
    )
    logger.info("结果文件: %s", output_file)
    return output_file


def prepare_replay_base(
    *,
    run_dir: Path,
    platform: str,
    model_label: str,
) -> Path | None:
    temp_replay_base = run_dir / f".eval_temp_{platform}"
    if temp_replay_base.exists():
        shutil.rmtree(temp_replay_base)
    temp_replay_base.mkdir(parents=True, exist_ok=True)

    found_mask_dir = False
    safe_model_label = _safe_model_label(model_label)
    for mask_type in MASK_TYPES:
        masked_dir = run_dir / f"masked_{mask_type}_{platform}"
        if not masked_dir.exists():
            continue
        result_files = list(masked_dir.glob("*_result.json"))
        if not result_files:
            continue
        found_mask_dir = True
        mask_method_dir = temp_replay_base / f"{safe_model_label}_{mask_type}_{platform}"
        mask_method_dir.mkdir(parents=True, exist_ok=True)
        for result_file in result_files:
            new_name = result_file.name.replace("_result.json", "_replay_result.json")
            shutil.copy2(result_file, mask_method_dir / new_name)

    if not found_mask_dir:
        shutil.rmtree(temp_replay_base)
        return None
    return temp_replay_base


def render_summary_text(
    *,
    run_name: str,
    model_config: dict[str, Any],
    android_tasks: Iterable[str],
    pc_tasks: Iterable[str],
    evaluation_paths: dict[str, str],
) -> str:
    lines = [
        "planner_eval run summary",
        "========================",
        f"run_name   : {run_name}",
        f"provider   : {model_config['provider']}",
        f"model      : {model_config['model']}",
        "",
        "Android tasks:",
    ]
    lines.extend(f"  - {task}" for task in android_tasks)
    lines.append("")
    lines.append("PC tasks:")
    lines.extend(f"  - {task}" for task in pc_tasks)
    lines.append("")
    lines.append("Mask types:")
    lines.extend(f"  - {mask_type}" for mask_type in MASK_TYPES)
    lines.append("")
    lines.append("Evaluation outputs:")
    if evaluation_paths:
        lines.extend(f"  - {platform}: {path}" for platform, path in evaluation_paths.items())
    else:
        lines.append("  - skipped")
    lines.append("")
    return "\n".join(lines)


def run_pipeline(args) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"{args.model_preset or args.model}_{timestamp}"
    output_base_dir = Path(args.output_base_dir).resolve()
    run_dir = output_base_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(run_dir / "combined.log")

    if args.model_preset:
        model_config = resolve_model_config(
            args.model_preset,
            preset_file=args.model_presets_file,
            api_key=args.api_key,
            base_url=args.base_url,
            model=args.model,
            provider=args.provider,
            temperature=args.temperature,
        )
    else:
        if not args.model:
            raise ValueError("Either --model-preset or --model must be provided.")
        model_config = {
            "provider": args.provider,
            "model": args.model,
            "api_key": args.api_key,
            "base_url": args.base_url or "",
            "temperature": args.temperature,
        }

    safe_model_config = _redact_model_config(model_config)

    logger.info("Run directory: %s", run_dir)
    logger.info("Model config: %s", safe_model_config)

    original_android_base = (
        resolve_platform_base(Path(args.original_android_base).resolve(), "android")
        if args.original_android_base
        else None
    )
    original_pc_base = (
        resolve_platform_base(Path(args.original_pc_base).resolve(), "pc")
        if args.original_pc_base
        else None
    )
    mask_dataset_root = Path(args.mask_dataset_root).resolve()

    if original_android_base:
        logger.info("Resolved Android base: %s", original_android_base)
    if original_pc_base:
        logger.info("Resolved PC base: %s", original_pc_base)

    android_task_names = (
        discover_task_names(original_android_base, "android", args.android_task)
        if original_android_base
        else []
    )
    pc_task_names = (
        discover_task_names(original_pc_base, "pc", args.pc_task)
        if original_pc_base
        else []
    )

    logger.info("Android task count: %s", len(android_task_names))
    logger.info("PC task count: %s", len(pc_task_names))

    for task_name in android_task_names:
        task = resolve_task_target(original_android_base, "android", task_name)
        if task:
            run_task(
                model_config=model_config,
                run_dir=run_dir,
                output_subdir_name="original_android",
                platform="android",
                task=task,
                max_steps=args.max_steps,
                max_trajectory_length=args.max_trajectory_length,
                memory_mode=args.memory_mode,
                enable_reflection=not args.no_reflection,
            )

    for task_name in pc_task_names:
        task = resolve_task_target(original_pc_base, "pc", task_name)
        if task:
            run_task(
                model_config=model_config,
                run_dir=run_dir,
                output_subdir_name="original_pc",
                platform="pc",
                task=task,
                max_steps=args.max_steps,
                max_trajectory_length=args.max_trajectory_length,
                memory_mode=args.memory_mode,
                enable_reflection=not args.no_reflection,
            )

    for mask_type in MASK_TYPES:
        logger.info("")
        logger.info("Running mask type: %s", mask_type)
        mask_type_root = mask_dataset_root / mask_type
        if original_android_base:
            mask_android_base = resolve_platform_base(mask_type_root, "android")
            for task_name in android_task_names:
                task = resolve_task_target(mask_android_base, "android", task_name)
                if task:
                    run_task(
                        model_config=model_config,
                        run_dir=run_dir,
                        output_subdir_name=f"masked_{mask_type}_android",
                        platform="android",
                        task=task,
                        max_steps=args.max_steps,
                        max_trajectory_length=args.max_trajectory_length,
                        memory_mode=args.memory_mode,
                        enable_reflection=not args.no_reflection,
                        mask_type=mask_type,
                    )
        if original_pc_base:
            mask_pc_base = resolve_platform_base(mask_type_root, "pc")
            for task_name in pc_task_names:
                task = resolve_task_target(mask_pc_base, "pc", task_name)
                if task:
                    run_task(
                        model_config=model_config,
                        run_dir=run_dir,
                        output_subdir_name=f"masked_{mask_type}_pc",
                        platform="pc",
                        task=task,
                        max_steps=args.max_steps,
                        max_trajectory_length=args.max_trajectory_length,
                        memory_mode=args.memory_mode,
                        enable_reflection=not args.no_reflection,
                        mask_type=mask_type,
                    )

    evaluation_paths: dict[str, str] = {}
    if not args.skip_evaluation:
        if not args.judge_api_key:
            logger.warning("Judge API key not provided, skipping evaluation stage.")
        else:
            if (run_dir / "original_android").exists():
                replay_base = prepare_replay_base(
                    run_dir=run_dir,
                    platform="android",
                    model_label=args.model_preset or args.model,
                )
                if replay_base:
                    output_dir = run_dir / "evaluation" / "android"
                    run_batch_evaluation(
                        gt_dir=run_dir / "original_android",
                        replay_base_dir=replay_base,
                        output_dir=output_dir,
                        model_name=f"{run_name}_android",
                        judge_model=args.judge_model,
                        judge_api_key=args.judge_api_key,
                        judge_base_url=args.judge_base_url,
                    )
                    evaluation_paths["android"] = str(output_dir)
                    shutil.rmtree(replay_base, ignore_errors=True)
            if (run_dir / "original_pc").exists():
                replay_base = prepare_replay_base(
                    run_dir=run_dir,
                    platform="pc",
                    model_label=args.model_preset or args.model,
                )
                if replay_base:
                    output_dir = run_dir / "evaluation" / "pc"
                    run_batch_evaluation(
                        gt_dir=run_dir / "original_pc",
                        replay_base_dir=replay_base,
                        output_dir=output_dir,
                        model_name=f"{run_name}_pc",
                        judge_model=args.judge_model,
                        judge_api_key=args.judge_api_key,
                        judge_base_url=args.judge_base_url,
                    )
                    evaluation_paths["pc"] = str(output_dir)
                    shutil.rmtree(replay_base, ignore_errors=True)

    summary_json = {
        "run_name": run_name,
        "model_config": safe_model_config,
        "android_tasks": android_task_names,
        "pc_tasks": pc_task_names,
        "mask_types": list(MASK_TYPES),
        "evaluation_paths": evaluation_paths,
        "run_dir": str(run_dir),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary_json, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (run_dir / "test_summary.txt").write_text(
        render_summary_text(
            run_name=run_name,
            model_config=model_config,
            android_tasks=android_task_names,
            pc_tasks=pc_task_names,
            evaluation_paths=evaluation_paths,
        ),
        encoding="utf-8",
    )

    logger.info("Pipeline finished: %s", run_dir)
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the full planner_eval pipeline")
    parser.add_argument("--output-base-dir", default="runs")
    parser.add_argument("--run-name", default=None)

    parser.add_argument("--model-preset", default=None)
    parser.add_argument("--model-presets-file", default=None)
    parser.add_argument("--provider", default="openai")
    parser.add_argument("--model", default=None)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", ""))
    parser.add_argument("--temperature", type=float, default=None)

    parser.add_argument("--judge-model", default="gpt-5")
    parser.add_argument(
        "--judge-api-key",
        default=os.getenv("JUDGE_API_KEY", os.getenv("OPENAI_API_KEY", "")),
    )
    parser.add_argument(
        "--judge-base-url",
        default=os.getenv("JUDGE_BASE_URL", os.getenv("OPENAI_BASE_URL", "")),
    )
    parser.add_argument("--skip-evaluation", action="store_true")

    parser.add_argument("--mask-dataset-root", required=True)
    parser.add_argument("--original-android-base", default="")
    parser.add_argument("--original-pc-base", default="")
    parser.add_argument("--android-task", action="append", default=[])
    parser.add_argument("--pc-task", action="append", default=[])

    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--max-trajectory-length", type=int, default=8)
    parser.add_argument(
        "--memory-mode",
        default="auto",
        choices=["auto", "online_full", "local_single_image"],
    )
    parser.add_argument("--no-reflection", action="store_true")
    return parser


def main() -> None:
    load_local_env()
    parser = build_parser()
    args = parser.parse_args()
    run_dir = run_pipeline(args)
    print(run_dir)


if __name__ == "__main__":
    main()
