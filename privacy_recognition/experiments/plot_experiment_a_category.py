import argparse
import json
from collections import defaultdict
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import evaluate_experiment_a as eva


CATEGORY_NAMES = {
    "1": "Identity",
    "2": "Contact",
    "3": "Technical",
    "4": "Behavior",
    "5": "Sensitive",
    "6": "Inference",
}


def safe_div(a, b):
    return 0.0 if b == 0 else a / b


def compute_category_metrics(gt, predictions, iou_threshold):
    stats = defaultdict(
        lambda: {
            "total": 0,
            "matched": 0,
            "strict": 0,
        }
    )

    for key, gt_labels in gt.items():
        platform = key[0]
        pred_labels = predictions.get(key, [])
        gt_private = [x for x in gt_labels if x["risk"] != "none"]
        pred_private = [x for x in pred_labels if x["risk"] != "none"]
        matches = eva.greedy_match(gt_private, pred_private, iou_threshold)

        for gt_box in gt_private:
            category = gt_box["category"]
            stats[("overall", category)]["total"] += 1
            stats[(platform, category)]["total"] += 1

        for gt_idx, pred_idx, _ in matches:
            gt_box = gt_private[gt_idx]
            pred_box = pred_private[pred_idx]
            category = gt_box["category"]
            strict_ok = int(
                pred_box["risk"] == gt_box["risk"]
                and pred_box["category"] == gt_box["category"]
                and pred_box["necessary"] == gt_box["necessary"]
            )
            stats[("overall", category)]["matched"] += 1
            stats[("overall", category)]["strict"] += strict_ok
            stats[(platform, category)]["matched"] += 1
            stats[(platform, category)]["strict"] += strict_ok

    result = {}
    for scope in ("overall", "Android", "PC"):
        result[scope] = {}
        for category in CATEGORY_NAMES:
            item = stats[(scope, category)]
            total = item["total"]
            matched = item["matched"]
            strict = item["strict"]
            result[scope][category] = {
                "count": total,
                "region_recall": safe_div(matched, total),
                "strict_accuracy": safe_div(strict, total),
            }
    return result


def plot_metrics(metrics, output_png):
    categories = list(CATEGORY_NAMES.keys())
    labels = [CATEGORY_NAMES[x] for x in categories]
    x = np.arange(len(categories))
    width = 0.24

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)

    for ax, metric_name, title in [
        (axes[0], "region_recall", "Category-wise Region Recall"),
        (axes[1], "strict_accuracy", "Category-wise Strict End-to-End Accuracy"),
    ]:
        overall_vals = [metrics["overall"][c][metric_name] * 100 for c in categories]
        android_vals = [metrics["Android"][c][metric_name] * 100 for c in categories]
        pc_vals = [metrics["PC"][c][metric_name] * 100 for c in categories]

        ax.bar(x - width, overall_vals, width, label="Overall", color="#4C78A8")
        ax.bar(x, android_vals, width, label="Android", color="#F58518")
        ax.bar(x + width, pc_vals, width, label="PC", color="#54A24B")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylabel("Percentage (%)")
        ax.set_title(title)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    axes[0].legend(frameon=False, ncol=3, loc="upper right")
    fig.suptitle("Experiment A on 1,015 Predicted Images", fontsize=13)
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot category-wise privacy-recognition metrics"
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
        "--iou",
        type=float,
        default=0.6,
        help="IoU threshold for region matching",
    )
    parser.add_argument(
        "--metrics-dir",
        default="outputs/metrics/experiment_a",
        help="Directory for metrics JSON",
    )
    parser.add_argument(
        "--figure-dir",
        default="outputs/figures/experiment_a",
        help="Directory for generated figures",
    )
    args = parser.parse_args()

    gt = eva.load_gt(args.gt, args.android_root, args.pc_root)
    predictions = eva.load_predictions(args.pred_root)
    metrics = compute_category_metrics(gt, predictions, args.iou)

    metrics_dir = Path(args.metrics_dir)
    figure_dir = Path(args.figure_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    json_path = metrics_dir / "category_metrics_engineering.json"
    png_path = figure_dir / "category_metrics_engineering.png"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    plot_metrics(metrics, png_path)
    print(f"Saved metrics to: {json_path}")
    print(f"Saved plot to: {png_path}")


if __name__ == "__main__":
    main()
