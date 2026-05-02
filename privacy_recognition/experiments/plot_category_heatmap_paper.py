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

import evaluate_experiment_a_paper as eva


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


def compute_category_metrics(gt, predictions, iou_threshold, text_threshold):
    stats = defaultdict(lambda: {"total": 0, "matched": 0, "strict": 0})

    for key, gt_labels in gt.items():
        platform = key[0]
        pred_labels = predictions.get(key)
        if pred_labels is None:
            continue
        gt_private = [x for x in gt_labels if x["risk"] != "none"]
        pred_private = [x for x in pred_labels if x["risk"] != "none"]
        matches = eva.greedy_match(gt_private, pred_private, iou_threshold, text_threshold)

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

    metrics = {}
    for scope in ("overall", "Android", "PC"):
        metrics[scope] = {}
        for category in CATEGORY_NAMES:
            item = stats[(scope, category)]
            metrics[scope][category] = {
                "count": item["total"],
                "recall": safe_div(item["matched"], item["total"]),
                "strict": safe_div(item["strict"], item["total"]),
            }
    return metrics


def plot_heatmap(metrics, out_png, out_pdf):
    categories = list(CATEGORY_NAMES.keys())
    rows = [CATEGORY_NAMES[c] for c in categories]
    cols = [
        "Overall Recall",
        "Android Recall",
        "PC Recall",
        "Overall Strict",
        "Android Strict",
        "PC Strict",
    ]

    matrix = []
    for category in categories:
        matrix.append(
            [
                metrics["overall"][category]["recall"] * 100,
                metrics["Android"][category]["recall"] * 100,
                metrics["PC"][category]["recall"] * 100,
                metrics["overall"][category]["strict"] * 100,
                metrics["Android"][category]["strict"] * 100,
                metrics["PC"][category]["strict"] * 100,
            ]
        )
    matrix = np.array(matrix)

    fig, ax = plt.subplots(figsize=(9.6, 4.8), constrained_layout=True)
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(rows)
    ax.set_title("Category-wise Privacy Recognition Difficulty")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            ax.text(j, i, f"{value:.1f}", ha="center", va="center", color="black", fontsize=9)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Percentage (%)")
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot paper-style category heatmap")
    parser.add_argument(
        "--gt",
        type=str,
        required=True,
        help="Path to the ground-truth JSON file",
    )
    parser.add_argument(
        "--pred-root",
        type=str,
        default="outputs/predictions",
        help="Root directory containing model predictions",
    )
    parser.add_argument(
        "--android-root",
        type=str,
        required=True,
        help="Raw Android dataset root",
    )
    parser.add_argument(
        "--pc-root",
        type=str,
        required=True,
        help="Raw PC dataset root",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.6,
    )
    parser.add_argument(
        "--text-threshold",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs/figures/experiment_a",
        help="Directory for generated figures",
    )
    args = parser.parse_args()

    gt = eva.load_gt(args.gt, args.android_root, args.pc_root)
    predictions = eva.load_predictions(args.pred_root)
    metrics = compute_category_metrics(gt, predictions, args.iou, args.text_threshold)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = out_dir.parent.parent / "metrics" / "experiment_a"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    json_path = metrics_dir / "category_metrics_paper.json"
    out_png = out_dir / "figure2_category_heatmap.png"
    out_pdf = out_dir / "figure2_category_heatmap.pdf"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    plot_heatmap(metrics, out_png, out_pdf)
    print(f"Saved: {json_path}")
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    main()
