import argparse
import glob
import json
from pathlib import Path

import privacy


def is_supported_task_dir(task_dir: Path) -> bool:
    if not (task_dir / "traj.jsonl").is_file():
        return False
    is_android = (task_dir / "task_result.json").is_file() and (task_dir / "images").is_dir()
    is_pc = (task_dir / "instruction.txt").is_file()
    return is_android or is_pc


def discover_task_dirs(input_path: Path) -> list[Path]:
    if is_supported_task_dir(input_path):
        return [input_path]

    task_dirs = {
        traj_path.parent
        for traj_path in input_path.rglob("traj.jsonl")
        if is_supported_task_dir(traj_path.parent)
    }
    return sorted(task_dirs, key=lambda path: path.as_posix())


def platform_name(task_dir: Path) -> str:
    for path in (task_dir, *task_dir.parents):
        if path.name in {"Android", "PC"}:
            return path.name
    return task_dir.parent.name


def resolve_pc_image(task_dir: Path, screenshot_name: str) -> str | None:
    if not screenshot_name:
        return None

    direct_path = task_dir / screenshot_name
    if direct_path.is_file():
        return str(direct_path)

    if screenshot_name.startswith("step") and screenshot_name.endswith(".png"):
        suffix = screenshot_name[4:-4]
        if suffix.isdigit():
            fallback_path = task_dir / f"step_{suffix}.png"
            if fallback_path.is_file():
                return str(fallback_path)

    return None


def collect_task(task_dir: Path):
    traj_path = task_dir / "traj.jsonl"
    if not traj_path.is_file():
        raise ValueError(f"Missing traj.jsonl: {task_dir}")

    if (task_dir / "task_result.json").is_file() and (task_dir / "images").is_dir():
        with open(task_dir / "task_result.json", "r", encoding="utf-8") as f:
            goal = str(json.load(f).get("goal", "")).strip()

        responses = []
        with open(traj_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                manager = data.get("manager")
                if isinstance(manager, dict) and manager.get("response"):
                    responses.append(str(manager["response"]).strip())

        image_files = sorted(glob.glob(str(task_dir / "images" / "*.png")))
        count = min(len(image_files), len(responses))
        return "android", goal, image_files[:count], responses[:count]

    if (task_dir / "instruction.txt").is_file():
        goal = (task_dir / "instruction.txt").read_text(encoding="utf-8").strip()
        image_files = []
        responses = []

        with open(traj_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                plan = str(data.get("plan") or "").strip()
                if not plan:
                    continue

                image_path = resolve_pc_image(
                    task_dir,
                    str(data.get("screenshot_file") or data.get("图片") or "").strip(),
                )
                if not image_path:
                    continue

                image_files.append(image_path)
                responses.append(plan)

        return "pc", goal, image_files, responses

    raise ValueError(f"Unsupported task layout: {task_dir}")


def run_task(task_dir: Path, model_name: str, output_root: Path, overwrite: bool):
    task_type, goal, image_files, responses = collect_task(task_dir)
    if not image_files or not responses:
        raise ValueError(f"No usable image/response pairs: {task_dir}")

    model_dir = model_name.replace("/", "_").replace(":", "_")
    output_dir = output_root / platform_name(task_dir) / task_dir.name / model_dir
    json_file = output_dir / "ai_results.json"
    if json_file.is_file() and not overwrite:
        print(f"Skipping existing result: {json_file}")
        return "skipped"

    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_template = privacy.get_prompt_template()
    all_results = []

    print(f"\nProcessing {task_dir} [{task_type}]")
    print(f"Images: {len(image_files)}")

    for idx, (image_file, response_text) in enumerate(zip(image_files, responses), start=1):
        print(f"Processing image {idx}/{len(image_files)}: {Path(image_file).name}")
        prompt = prompt_template.format(goal=goal, response=response_text)
        ai_output, _ = privacy.call_vlm_api(image_file, prompt, model_name)
        items = privacy.parse_ai_output(ai_output)
        all_results.append(items)
        privacy.draw_boxes_on_image(
            image_file,
            items,
            str(output_dir / Path(image_file).name.replace(".png", "_annotated.png")),
        )

    json_output = privacy.convert_to_json_format(str(task_dir), image_files, all_results)
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(json_output, f, indent=2, ensure_ascii=False)

    print(f"Saved: {json_file}")
    return "processed"


def main():
    parser = argparse.ArgumentParser(description="Batch runner for ScreenPriv datasets")
    parser.add_argument("input_path", type=str, help="Task folder or dataset root")
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="gpt-4o-mini",
        help="Model name",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="outputs/predictions",
        help="Root directory for outputs",
    )
    parser.add_argument(
        "--task-limit",
        type=int,
        default=None,
        help="Only process the first N tasks",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs",
    )
    args = parser.parse_args()

    privacy.get_client()

    input_path = Path(args.input_path)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    task_dirs = discover_task_dirs(input_path)

    if args.task_limit is not None:
        task_dirs = task_dirs[: args.task_limit]

    processed = 0
    skipped = 0
    failed = 0
    for task_dir in task_dirs:
        try:
            status = run_task(task_dir, args.model, Path(args.output_root), args.overwrite)
            if status == "skipped":
                skipped += 1
            else:
                processed += 1
        except Exception as exc:
            failed += 1
            print(f"Failed: {task_dir}")
            print(f"Error: {exc}")

    print(
        f"\nSummary: processed={processed}, skipped={skipped}, failed={failed}, "
        f"output_root={args.output_root}"
    )


if __name__ == "__main__":
    main()
