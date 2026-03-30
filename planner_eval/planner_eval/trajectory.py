from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image

from planner_eval.agents.agent_s import AgentS3
from planner_eval.core.llm_trace import (
    drain_for_step,
    reset_buffer,
    set_current_step,
    snapshot_all,
)
from planner_eval.simple_grounding import SimpleACI
from planner_eval.utils.env import load_local_env


logger = logging.getLogger(__name__)


def load_screenshots_from_directory(screenshot_dir: str) -> List[str]:
    screenshot_path = Path(screenshot_dir)
    if not screenshot_path.exists():
        raise ValueError(f"Screenshot directory does not exist: {screenshot_dir}")

    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}
    screenshot_files = []
    for ext in image_extensions:
        screenshot_files.extend(screenshot_path.glob(f"*{ext}"))

    screenshot_files = sorted(screenshot_files, key=lambda x: x.name)
    if not screenshot_files:
        raise ValueError(f"No screenshots found in directory: {screenshot_dir}")

    logger.info("Found %s screenshots in %s", len(screenshot_files), screenshot_dir)
    return [str(path) for path in screenshot_files]


def load_screenshot_bytes(image_path: str) -> bytes:
    try:
        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")
        import io

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()
    except Exception as exc:
        logger.error("Failed to load screenshot %s: %s", image_path, exc)
        raise


def build_agent(
    *,
    provider: str,
    model: str,
    model_url: str = "",
    model_api_key: str = "",
    model_temperature: float | None = None,
    platform: str = "linux",
    max_trajectory_length: int = 8,
    enable_reflection: bool = True,
) -> AgentS3:
    engine_params = {
        "engine_type": provider,
        "model": model,
        "base_url": model_url,
        "api_key": model_api_key,
        "temperature": model_temperature,
    }
    grounding_agent = SimpleACI()
    return AgentS3(
        engine_params,
        grounding_agent,
        platform=platform,
        max_trajectory_length=max_trajectory_length,
        enable_reflection=enable_reflection,
    )


def run_trajectory_evaluation(
    agent: AgentS3,
    instruction: str,
    screenshot_paths: List[str],
    max_steps: int | None = None,
) -> Dict[str, Any]:
    if max_steps is None:
        max_steps = len(screenshot_paths)

    reset_buffer()
    results: Dict[str, Any] = {
        "instruction": instruction,
        "total_screenshots": len(screenshot_paths),
        "steps_executed": 0,
        "actions": [],
        "completed": False,
        "failed": False,
    }

    logger.info("=" * 80)
    logger.info("Starting trajectory evaluation")
    logger.info("Task: %s", instruction)
    logger.info("Total screenshots: %s", len(screenshot_paths))
    logger.info("Max steps: %s", max_steps)
    logger.info("=" * 80)

    for step_idx in range(min(max_steps, len(screenshot_paths))):
        set_current_step(step_idx + 1)
        screenshot_path = screenshot_paths[step_idx]

        logger.info("")
        logger.info("=" * 80)
        logger.info("Step %s/%s", step_idx + 1, min(max_steps, len(screenshot_paths)))
        logger.info("Screenshot: %s", screenshot_path)
        logger.info("=" * 80)

        try:
            screenshot_bytes = load_screenshot_bytes(screenshot_path)
        except Exception as exc:
            logger.error("Failed to load screenshot, skipping step: %s", exc)
            continue

        observation = {
            "screenshot": screenshot_bytes,
            "screenshot_path": screenshot_path,
        }

        try:
            info, actions = agent.predict(instruction=instruction, observation=observation)
            logger.info("")
            logger.info("Agent Output:")
            logger.info("  Plan: %s", info.get("plan", "N/A"))
            logger.info("  Actions: %s", actions)

            step_result = {
                "step": step_idx + 1,
                "screenshot_path": screenshot_path,
                "info": info,
                "actions": actions,
                "llm_inputs": drain_for_step(step_idx + 1),
            }
            results["actions"].append(step_result)
            results["steps_executed"] = step_idx + 1

            if actions:
                action_str = str(actions[0]).lower()
                if "done" in action_str:
                    logger.info("")
                    logger.info("Agent signaled task completion (DONE)")
                    results["completed"] = True
                    break
                if "fail" in action_str:
                    logger.info("")
                    logger.info("Agent signaled task failure (FAIL)")
                    results["failed"] = True
                    break
        except Exception as exc:
            logger.error("Error during agent prediction: %s", exc, exc_info=True)
            results["actions"].append(
                {
                    "step": step_idx + 1,
                    "screenshot_path": screenshot_path,
                    "error": str(exc),
                    "llm_inputs": drain_for_step(step_idx + 1),
                }
            )

    logger.info("")
    logger.info("=" * 80)
    logger.info("Trajectory Evaluation Complete")
    logger.info(
        "Steps executed: %s/%s",
        results["steps_executed"],
        len(screenshot_paths),
    )
    logger.info("Task completed: %s", results["completed"])
    logger.info("Task failed: %s", results["failed"])
    logger.info("=" * 80)
    return results


def make_json_serializable_results(results: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "instruction": results["instruction"],
        "total_screenshots": results["total_screenshots"],
        "steps_executed": results["steps_executed"],
        "completed": results["completed"],
        "failed": results["failed"],
        "llm_inputs_all": snapshot_all(),
        "actions": [
            {
                "step": action.get("step"),
                "screenshot_path": action.get("screenshot_path"),
                "plan": action.get("info", {}).get("plan"),
                "actions": [str(item) for item in action.get("actions", [])],
                "llm_inputs": action.get("llm_inputs", []),
                "error": action.get("error"),
            }
            for action in results["actions"]
        ],
    }


def save_results(results: Dict[str, Any], output_file: str | Path) -> None:
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(make_json_serializable_results(results), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Results saved to: %s", output_path)


def evaluate_directory(
    *,
    screenshot_dir: str,
    instruction: str,
    provider: str,
    model: str,
    model_url: str = "",
    model_api_key: str = "",
    model_temperature: float | None = None,
    max_steps: int | None = None,
    platform: str = "linux",
    max_trajectory_length: int = 8,
    enable_reflection: bool = True,
    output_file: str | Path | None = None,
) -> Dict[str, Any]:
    screenshot_paths = load_screenshots_from_directory(screenshot_dir)
    agent = build_agent(
        provider=provider,
        model=model,
        model_url=model_url,
        model_api_key=model_api_key,
        model_temperature=model_temperature,
        platform=platform,
        max_trajectory_length=max_trajectory_length,
        enable_reflection=enable_reflection,
    )
    results = run_trajectory_evaluation(
        agent=agent,
        instruction=instruction,
        screenshot_paths=screenshot_paths,
        max_steps=max_steps,
    )
    if output_file:
        save_results(results, output_file)
    return results


def main() -> None:
    load_local_env()
    parser = argparse.ArgumentParser(
        description="Run trajectory evaluation on prerecorded screenshots."
    )
    parser.add_argument("--provider", default="openai")
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-url", default="")
    parser.add_argument("--model-api-key", default="")
    parser.add_argument("--model-temperature", type=float, default=None)
    parser.add_argument("--screenshot-dir", required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--platform",
        default="linux",
        choices=["linux", "darwin", "windows"],
    )
    parser.add_argument("--max-trajectory-length", type=int, default=8)
    parser.add_argument(
        "--enable-reflection",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-reflection",
        dest="enable_reflection",
        action="store_false",
    )
    parser.add_argument("--output-file", default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    results = evaluate_directory(
        screenshot_dir=args.screenshot_dir,
        instruction=args.instruction,
        provider=args.provider,
        model=args.model,
        model_url=args.model_url,
        model_api_key=args.model_api_key,
        model_temperature=args.model_temperature,
        max_steps=args.max_steps,
        platform=args.platform,
        max_trajectory_length=args.max_trajectory_length,
        enable_reflection=args.enable_reflection,
        output_file=args.output_file,
    )

    print("")
    print("=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Task: {args.instruction}")
    print(
        "Screenshots processed: "
        f"{results['steps_executed']}/{results['total_screenshots']}"
    )
    print(
        "Status: "
        + (
            "Completed"
            if results["completed"]
            else "Failed"
            if results["failed"]
            else "Incomplete"
        )
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
