#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import instructor
from openai import OpenAI

from planner_eval.judges.base import Judgment
from planner_eval.judges.graders.semantic_consistency import SemanticConsistencyGrader
from planner_eval.utils.env import load_local_env


class CustomSemanticConsistencyGrader(SemanticConsistencyGrader):
    """Semantic consistency grader backed by an OpenAI-compatible endpoint."""

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        api_log_path: str | None = None,
    ) -> None:
        super().__init__(model)
        self.api_key = api_key
        self.base_url = base_url
        self.api_log_path = api_log_path
        self._client = None

    def _get_client(self):
        if self._client is None:
            openai_client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                default_headers={
                    "HTTP-Referer": "https://github.com",
                    "X-Title": "planner_eval semantic consistency evaluation",
                },
            )
            self._client = instructor.from_openai(openai_client)
        return self._client

    def judge(self, input: str = None, output: str = None, expected: str = None):
        from textwrap import dedent

        system_prompt = (
            "You are an expert at evaluating semantic consistency of AI agent action plans."
        )
        task_context = f"\nTask Context: {input}\n" if input else ""
        user_prompt = dedent(
            f"""
            Assess the semantic consistency between the GROUND TRUTH PLAN and the REPLAY PLAN on a five-point scale:

            [0] Completely Inconsistent: Plans describe entirely different actions or goals.
            [1] Minimally Consistent: Plans share some superficial similarity but have different intents or targets.
            [2] Partially Consistent: Plans have the same general intent but differ in approach or implementation.
            [3] Mostly Consistent: Plans are semantically similar with only minor differences.
            [4] Fully Consistent: Plans are semantically identical, expressing the exact same action, target, and intent.
            {task_context}
            Ground Truth Plan:
            {expected}

            Replay Plan:
            {output}

            Semantic Consistency Score (0-4):
            """
        )
        messages = self._build_messages(user_prompt, system_prompt)
        client = self._get_client()

        attempt = 0
        backoff_seconds = 1.0
        while True:
            try:
                judgment = client.chat.completions.create(
                    messages=messages,
                    temperature=0.0,
                    response_model=Judgment,
                    model=self.model,
                )
                self._append_api_log(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "model": self.model,
                        "status": "success",
                        "attempt": attempt + 1,
                        "input": {
                            "system_prompt": system_prompt,
                            "user_prompt": user_prompt,
                        },
                        "response": judgment.model_dump(),
                    }
                )
                return judgment
            except Exception as exc:
                attempt += 1
                payload: Any = None
                response = getattr(exc, "response", None)
                if response is not None:
                    try:
                        if hasattr(response, "json"):
                            payload = response.json()
                        elif hasattr(response, "text"):
                            payload = response.text
                    except Exception:
                        payload = getattr(response, "text", None)
                if payload is None:
                    payload = str(exc)
                logging.error(
                    "Judge LLM error (model=%s attempt=%d): %s",
                    self.model,
                    attempt,
                    payload,
                )
                self._append_api_log(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "model": self.model,
                        "status": "error",
                        "attempt": attempt,
                        "input": {
                            "system_prompt": system_prompt,
                            "user_prompt": user_prompt,
                        },
                        "error": payload,
                    }
                )
                time.sleep(backoff_seconds)
                backoff_seconds = min(backoff_seconds * 1.5, 60.0)

    def _append_api_log(self, entry: dict) -> None:
        if not self.api_log_path:
            return
        try:
            with open(self.api_log_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False))
                handle.write("\n")
        except Exception:
            logging.getLogger(__name__).exception("Failed to write judge API log")


def load_json_file(file_path: Path) -> dict:
    return json.loads(file_path.read_text(encoding="utf-8"))


def extract_plans(data: dict) -> list[str]:
    plans = []
    for action in data.get("actions", []):
        plan = action.get("plan")
        if plan:
            plans.append(plan)
    return plans


def get_score_label(score: int) -> str:
    labels = {
        0: "完全不一致",
        1: "最低程度一致",
        2: "部分一致",
        3: "大部分一致",
        4: "完全一致",
    }
    return labels.get(score, "未知")


def _coerce_score(score: Any) -> float:
    if isinstance(score, (int, float)):
        return float(score)
    if isinstance(score, str):
        score = score.strip()
        try:
            return float(score)
        except ValueError:
            pass
        import re

        match = re.search(r"-?\d+(?:\.\d+)?", score)
        if match:
            return float(match.group(0))
    raise ValueError(f"Unable to parse numeric score from: {score!r}")


def evaluate_sample(
    *,
    gt_file: Path,
    replay_file: Path,
    grader: CustomSemanticConsistencyGrader,
    task_name: str,
) -> dict:
    gt_data = load_json_file(gt_file)
    replay_data = load_json_file(replay_file)

    gt_plans = extract_plans(gt_data)
    replay_plans = extract_plans(replay_data)

    step_results = []
    for index in range(min(len(gt_plans), len(replay_plans))):
        gt_plan = gt_plans[index]
        replay_plan = replay_plans[index]
        judgment = grader.judge(
            input=f"Task: {task_name}",
            output=replay_plan,
            expected=gt_plan,
        )
        numeric_score = _coerce_score(judgment.score)
        step_results.append(
            {
                "step": index + 1,
                "gt_plan": gt_plan,
                "replay_plan": replay_plan,
                "score": numeric_score,
                "score_label": get_score_label(int(round(numeric_score))),
                "reasoning": judgment.reasoning,
            }
        )

    total_score = sum(item["score"] for item in step_results)
    max_score = len(step_results) * 4
    avg_score = total_score / len(step_results) if step_results else 0.0
    consistency_rate = total_score / max_score if max_score else 0.0

    return {
        "task_name": task_name,
        "gt_file": str(gt_file),
        "replay_file": str(replay_file),
        "gt_steps": len(gt_plans),
        "replay_steps": len(replay_plans),
        "evaluated_steps": len(step_results),
        "total_score": total_score,
        "max_score": max_score,
        "avg_score": avg_score,
        "consistency_rate": consistency_rate,
        "step_results": step_results,
    }


def evaluate_mask_method(
    *,
    mask_name: str,
    gt_dir: Path,
    replay_dir: Path,
    grader: CustomSemanticConsistencyGrader,
    output_dir: Path,
) -> dict | None:
    print("")
    print("=" * 80)
    print(f"评估 Mask 方法: {mask_name}")
    print("=" * 80)

    gt_files = list(gt_dir.glob("*_result.json"))
    replay_files = list(replay_dir.glob("*_replay_result.json")) + list(replay_dir.glob("*_result.json"))

    gt_by_stem = {gt_file.stem.replace("_result", ""): gt_file for gt_file in gt_files}
    matched_pairs = []
    seen_task_names: set[str] = set()
    for replay_file in sorted(replay_files):
        replay_stem = (
            replay_file.stem.replace("_replay_result", "").replace("_result", "")
        )
        if replay_stem in seen_task_names:
            continue
        gt_file = gt_by_stem.get(replay_stem)
        if gt_file is None:
            continue
        matched_pairs.append((gt_file, replay_file, replay_stem))
        seen_task_names.add(replay_stem)

    print(f"找到 {len(matched_pairs)} 对样本")
    if not matched_pairs:
        print(f"警告: 未找到 {mask_name} 的匹配样本")
        return None

    all_results = []
    for index, (gt_file, replay_file, task_name) in enumerate(matched_pairs, start=1):
        print(f"  [{index}/{len(matched_pairs)}] {task_name}...", end=" ", flush=True)
        result = evaluate_sample(
            gt_file=gt_file,
            replay_file=replay_file,
            grader=grader,
            task_name=task_name,
        )
        all_results.append(result)
        print(f"✓ {result['avg_score']:.2f}/4 ({result['consistency_rate']:.1%})")

    summary = {
        "mask_method": mask_name,
        "total_samples": len(all_results),
        "results": all_results,
        "summary": {
            "total_score": sum(item["total_score"] for item in all_results),
            "max_score": sum(item["max_score"] for item in all_results),
            "avg_score": (
                sum(item["avg_score"] for item in all_results) / len(all_results)
                if all_results
                else 0.0
            ),
            "avg_consistency_rate": (
                sum(item["consistency_rate"] for item in all_results) / len(all_results)
                if all_results
                else 0.0
            ),
            "total_steps": sum(item["evaluated_steps"] for item in all_results),
        },
    }

    output_file = output_dir / f"{mask_name}_evaluation.json"
    output_file.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("")
    print(f"{mask_name} 汇总:")
    print(
        f"  总分: {summary['summary']['total_score']}/{summary['summary']['max_score']}"
    )
    print(f"  平均分: {summary['summary']['avg_score']:.2f}/4")
    print(
        "  平均一致性率: "
        f"{summary['summary']['avg_consistency_rate']:.2%}"
    )
    print(f"  结果已保存: {output_file}")
    return summary


def run_batch_evaluation(
    *,
    gt_dir: Path,
    replay_base_dir: Path,
    output_dir: Path,
    model_name: str,
    judge_model: str,
    judge_api_key: str,
    judge_base_url: str,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    api_log_path = output_dir / f"{model_name}_judge_api.log"
    grader = CustomSemanticConsistencyGrader(
        model=judge_model,
        api_key=judge_api_key,
        base_url=judge_base_url,
        api_log_path=str(api_log_path),
    )

    mask_methods = [path.name for path in replay_base_dir.iterdir() if path.is_dir()]
    mask_methods.sort()

    print("")
    print(f"找到 {len(mask_methods)} 个 mask 方法: {', '.join(mask_methods)}")

    all_mask_results = []
    start_time = datetime.now()
    for index, mask_name in enumerate(mask_methods, start=1):
        print("")
        print(f"[{index}/{len(mask_methods)}] 开始评估 {mask_name}...")
        mask_result = evaluate_mask_method(
            mask_name=mask_name,
            gt_dir=gt_dir,
            replay_dir=replay_base_dir / mask_name,
            grader=grader,
            output_dir=output_dir,
        )
        if mask_result:
            all_mask_results.append(mask_result)

    duration = (datetime.now() - start_time).total_seconds()
    overall_summary = {
        "model_name": model_name,
        "evaluation_time": start_time.isoformat(),
        "duration_seconds": duration,
        "evaluator_model": judge_model,
        "gt_dir": str(gt_dir),
        "replay_dir": str(replay_base_dir),
        "total_mask_methods": len(all_mask_results),
        "mask_results": [
            {
                "mask_method": item["mask_method"],
                "total_samples": item["total_samples"],
                "total_score": item["summary"]["total_score"],
                "max_score": item["summary"]["max_score"],
                "avg_score": item["summary"]["avg_score"],
                "avg_consistency_rate": item["summary"]["avg_consistency_rate"],
                "total_steps": item["summary"]["total_steps"],
            }
            for item in all_mask_results
        ],
        "overall_statistics": {
            "total_samples": sum(item["total_samples"] for item in all_mask_results),
            "total_score": sum(item["summary"]["total_score"] for item in all_mask_results),
            "total_max_score": sum(item["summary"]["max_score"] for item in all_mask_results),
            "avg_score_across_all": (
                sum(item["summary"]["avg_score"] for item in all_mask_results)
                / len(all_mask_results)
                if all_mask_results
                else 0.0
            ),
            "avg_consistency_rate_across_all": (
                sum(item["summary"]["avg_consistency_rate"] for item in all_mask_results)
                / len(all_mask_results)
                if all_mask_results
                else 0.0
            ),
            "total_steps": sum(item["summary"]["total_steps"] for item in all_mask_results),
        },
    }

    summary_file = output_dir / "overall_summary.json"
    summary_file.write_text(
        json.dumps(overall_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("")
    print("=" * 80)
    print(f"评估完成！{model_name} 总体汇总：")
    print("=" * 80)
    print(f"评估时长: {duration:.1f} 秒")
    print("")
    print("Mask 方法排名（按平均分）：")
    print("-" * 80)
    for rank, item in enumerate(
        sorted(
            all_mask_results,
            key=lambda result: result["summary"]["avg_score"],
            reverse=True,
        ),
        start=1,
    ):
        print(
            f"{rank:>2}. {item['mask_method']} - 平均: "
            f"{item['summary']['avg_score']:.2f}/4 "
            f"({item['summary']['avg_consistency_rate']:.1%}) - "
            f"总分: {item['summary']['total_score']}/{item['summary']['max_score']}"
        )
    print("")
    print("总体统计:")
    print(
        f"  总样本数: {overall_summary['overall_statistics']['total_samples']}"
    )
    print(
        "  总分: "
        f"{overall_summary['overall_statistics']['total_score']}/"
        f"{overall_summary['overall_statistics']['total_max_score']}"
    )
    print(
        "  平均分: "
        f"{overall_summary['overall_statistics']['avg_score_across_all']:.2f}/4"
    )
    print(
        "  平均一致性率: "
        f"{overall_summary['overall_statistics']['avg_consistency_rate_across_all']:.2%}"
    )
    print("")
    print(f"所有结果已保存到: {output_dir}")
    print(f"总汇总文件: {summary_file}")
    print("=" * 80)
    return overall_summary


def main() -> None:
    load_local_env()
    parser = argparse.ArgumentParser(description="Batch semantic consistency evaluation")
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--replay-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default="unknown")
    parser.add_argument("--judge-model", default=os.getenv("JUDGE_MODEL", "openai/gpt-5"))
    parser.add_argument("--judge-api-key", default=os.getenv("JUDGE_API_KEY", ""))
    parser.add_argument("--judge-base-url", default=os.getenv("JUDGE_BASE_URL", ""))
    args = parser.parse_args()

    if not args.judge_api_key:
        raise SystemExit("Missing judge API key. Set JUDGE_API_KEY or pass --judge-api-key.")

    run_batch_evaluation(
        gt_dir=Path(args.gt_dir),
        replay_base_dir=Path(args.replay_dir),
        output_dir=Path(args.output_dir),
        model_name=args.model_name,
        judge_model=args.judge_model,
        judge_api_key=args.judge_api_key,
        judge_base_url=args.judge_base_url,
    )


if __name__ == "__main__":
    main()
