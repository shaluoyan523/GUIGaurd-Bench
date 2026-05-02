import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import evaluate_experiment_a as eva


def main():
    parser = argparse.ArgumentParser(
        description="Export category 5/6 failure cases from privacy-recognition predictions"
    )
    parser.add_argument("--gt", required=True, help="Path to the ground-truth JSON file")
    parser.add_argument(
        "--android-root", required=True, help="Raw Android dataset root"
    )
    parser.add_argument("--pc-root", required=True, help="Raw PC dataset root")
    parser.add_argument(
        "--pred-root",
        default="outputs/predictions",
        help="Root directory containing model predictions",
    )
    parser.add_argument(
        "--output",
        default="outputs/analysis/experiment_a/category56_failure_cases.json",
        help="Output JSON path",
    )
    args = parser.parse_args()

    gt = eva.load_gt(args.gt, args.android_root, args.pc_root)
    pred = eva.load_predictions(args.pred_root)

    matched_but_wrong = []
    unmatched = []

    for key, gt_labels in gt.items():
        platform, task, image = key
        pred_labels = pred.get(key, [])
        gt_private = [x for x in gt_labels if x["risk"] != "none"]
        pred_private = [x for x in pred_labels if x["risk"] != "none"]
        matches = eva.greedy_match(gt_private, pred_private, 0.6)
        matched_gt = {x[0] for x in matches}

        for gt_idx, pred_idx, iou in matches:
            gt_box = gt_private[gt_idx]
            pred_box = pred_private[pred_idx]
            if gt_box["category"] not in {"5", "6"}:
                continue
            strict_ok = (
                pred_box["risk"] == gt_box["risk"]
                and pred_box["category"] == gt_box["category"]
                and pred_box["necessary"] == gt_box["necessary"]
            )
            if strict_ok:
                continue
            matched_but_wrong.append(
                {
                    "platform": platform,
                    "task": task,
                    "image": image,
                    "iou": iou,
                    "gt": gt_box,
                    "pred": pred_box,
                }
            )

        for gt_idx, gt_box in enumerate(gt_private):
            if gt_box["category"] not in {"5", "6"}:
                continue
            if gt_idx in matched_gt:
                continue
            unmatched.append(
                {
                    "platform": platform,
                    "task": task,
                    "image": image,
                    "gt": gt_box,
                }
            )

    matched_but_wrong.sort(
        key=lambda x: (
            x["gt"]["category"],
            x["platform"],
            -x["iou"],
        )
    )
    unmatched.sort(
        key=lambda x: (
            x["gt"]["category"],
            x["platform"],
            x["task"],
            x["image"],
        )
    )

    summary = {
        "matched_but_wrong_total": len(matched_but_wrong),
        "unmatched_total": len(unmatched),
        "matched_but_wrong_examples": matched_but_wrong[:60],
        "unmatched_examples": unmatched[:60],
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Saved failure cases to: {out_path}")
    print(f"matched_but_wrong_total={len(matched_but_wrong)}")
    print(f"unmatched_total={len(unmatched)}")


if __name__ == "__main__":
    main()
