import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_result(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_cascade(items, out_png, out_pdf):
    stages = ["Binary", "Recall", "Strict"]
    stage_keys = {
        "Binary": "binary_accuracy",
        "Recall": "privacy_recall",
        "Strict": "overall_end_to_end",
    }

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    platform_names = ["Android", "PC"]
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]
    markers = ["o", "s", "^", "D", "P"]

    for ax, platform in zip(axes, platform_names):
        for idx, item in enumerate(items):
            result = item["result"][platform]
            ys = [result[stage_keys[stage]] * 100 for stage in stages]
            ax.plot(
                stages,
                ys,
                marker=markers[idx % len(markers)],
                linewidth=2.2,
                markersize=7,
                color=colors[idx % len(colors)],
                label=item["label"],
            )
        ax.set_title(platform)
        ax.set_ylabel("Percentage (%)")
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    axes[0].legend(frameon=False, loc="upper right")
    fig.suptitle("Recognition Difficulty Cascade", fontsize=13)
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot paper-style recognition cascade figure")
    parser.add_argument(
        "--model-result",
        action="append",
        required=True,
        help="label=path pairs for paper-style result json files",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs/figures/experiment_a",
    )
    args = parser.parse_args()

    items = []
    for pair in args.model_result:
        label, path = pair.split("=", 1)
        items.append({"label": label, "result": load_result(path)})

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "figure1_recognition_cascade.png"
    out_pdf = out_dir / "figure1_recognition_cascade.pdf"
    plot_cascade(items, out_png, out_pdf)
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    main()
