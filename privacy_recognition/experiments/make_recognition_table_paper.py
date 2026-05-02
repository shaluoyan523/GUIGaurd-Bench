import argparse
import json
from pathlib import Path


def load_result(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def fmt_percent(value):
    return f"{value * 100:.1f}"


def build_rows(items):
    rows = []
    for item in items:
        result = item["result"]["overall"]
        rows.append(
            {
                "model": item["label"],
                "binary": fmt_percent(result["binary_accuracy"]),
                "recall": fmt_percent(result["privacy_recall"]),
                "strict": fmt_percent(result["overall_end_to_end"]),
            }
        )
    return rows


def write_markdown(rows, path):
    lines = [
        "| Model | Binary (%) | Recall (%) | Strict Overall (%) |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['binary']} | {row['recall']} | {row['strict']} |"
        )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex(rows, path):
    lines = [
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Model & Binary (\\%) & Recall (\\%) & Strict Overall (\\%) \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['model']} & {row['binary']} & {row['recall']} & {row['strict']} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Generate a compact paper table for recognition results")
    parser.add_argument(
        "--model-result",
        action="append",
        required=True,
        help="label=path pairs for paper-style result json files",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs/tables/experiment_a",
    )
    args = parser.parse_args()

    items = []
    for pair in args.model_result:
        label, path = pair.split("=", 1)
        items.append({"label": label, "result": load_result(path)})

    rows = build_rows(items)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / "recognition_summary_table.md"
    tex_path = out_dir / "recognition_summary_table.tex"
    json_path = out_dir / "recognition_summary_table.json"

    write_markdown(rows, md_path)
    write_latex(rows, tex_path)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    print(f"Saved: {md_path}")
    print(f"Saved: {tex_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
