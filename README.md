# GUIGuard-Bench

This repository provides evaluation code for **GUIGuard-Bench**, a cross-platform benchmark for studying privacy risks and privacy-preserving execution in GUI agents.

GUI agents operate directly on screenshots and interface trajectories. This creates a privacy challenge: screenshots may contain sensitive personal information, and the privacy risk of a GUI workflow often depends on sequential context rather than a single isolated image. GUIGuard-Bench supports evaluation of privacy recognition, protected-screen task execution, and click grounding under visual protection.

## Anonymous Review Notice

This repository is prepared for double-blind review. It intentionally avoids author names, affiliations, organization names, internal paths, private service endpoints, and API keys. Please do not add identifying information to issues, logs, examples, configs, or output files during review.

## Benchmark Overview

GUIGuard-Bench contains cross-platform GUI trajectories and screenshot annotations for privacy-aware agent evaluation.

The benchmark includes:

- Android and PC GUI trajectories.
- Region-level privacy grounding annotations.
- Fine-grained labels for privacy risk level, privacy category, and task necessity.
- Original screenshots and protected screenshot variants.
- Evaluation settings for privacy recognition, semantic trajectory consistency, and ScreenSpot-style grounding.

The dataset is hosted at:

```text
https://huggingface.co/datasets/ShaofantuoshuzhengzhiSha/GUIGuard-Bench
```

The dataset page lists the dataset license as MIT.

## Repository Layout

```text
.
├── planner_eval/
│   ├── planner_eval/
│   ├── configs/
│   ├── scripts/
│   ├── tests/
│   ├── README.md
│   └── pyproject.toml
└── grounding_eval/
    ├── grounding_eval/
    ├── configs/
    ├── examples/
    ├── tests/
    ├── README.md
    └── pyproject.toml
```

### `planner_eval`

`planner_eval` is a trajectory-level evaluation package. It runs a multimodal planning model on prerecorded GUI screenshot trajectories, compares original and protected trajectories, and scores step-level semantic consistency with an LLM-as-judge.

Use it for:

- Planning on original screenshots.
- Planning on protected screenshots.
- Comparing masked or transformed trajectories to ground-truth trajectories.
- Producing semantic-consistency scores and per-mask summaries.

See [`planner_eval/README.md`](planner_eval/README.md) for package-specific details.

### `grounding_eval`

`grounding_eval` is a ScreenSpot-style click grounding evaluation package. It sends a screenshot and the next-action plan to a vision-language model, asks for a click point, normalizes the returned coordinate frame, and checks whether the predicted point falls inside the ground-truth UI bounding box.

Use it for:

- Original-screen grounding accuracy.
- Grounding accuracy under black-mask or LLM-replacement protection.
- Multi-model and multi-mask concurrent evaluation.
- Model-specific coordinate normalization, including normalized coordinates, pixel coordinates, and Claude-style Android coordinates.

See [`grounding_eval/README.md`](grounding_eval/README.md) for package-specific details.

## Installation

Clone the repository and install the evaluation packages in editable mode:

```bash
git clone <repository-url>
cd GUIGaurd-Bench

python -m pip install -e planner_eval
python -m pip install -e grounding_eval
```

Optional test dependencies:

```bash
python -m pip install -e 'grounding_eval.[dev]'
```

Run tests:

```bash
pytest grounding_eval/tests
pytest planner_eval/tests
```

## Dataset Download

Install the Hugging Face CLI if needed:

```bash
python -m pip install -U huggingface_hub
```

Download the dataset:

```bash
huggingface-cli download \
  ShaofantuoshuzhengzhiSha/GUIGuard-Bench \
  --repo-type dataset \
  --local-dir data/GUIGuard-Bench
```

The exact directory names may depend on the dataset release. The evaluation scripts accept explicit paths, so keep the dataset outside the repository if desired and point the CLI arguments to your local copy.

## Environment Variables

Both evaluation packages use OpenAI-compatible APIs. Do not commit real credentials.

```bash
export OPENAI_API_KEY="..."
export OPENAI_BASE_URL="https://api.openai.com/v1"
```

For judge models, `planner_eval` can use separate judge credentials:

```bash
export JUDGE_API_KEY="..."
export JUDGE_BASE_URL="https://api.openai.com/v1"
```

Use `.env.example` files as templates only.

## Quick Start: Planner Evaluation

Minimal example:

```bash
planner-eval-pipeline \
  --model <planner-model> \
  --api-key "$OPENAI_API_KEY" \
  --mask-dataset-root /path/to/masked_dataset \
  --original-android-base /path/to/Android \
  --original-pc-base /path/to/PC \
  --skip-evaluation
```

With judge scoring:

```bash
planner-eval-pipeline \
  --model <planner-model> \
  --api-key "$OPENAI_API_KEY" \
  --mask-dataset-root /path/to/masked_dataset \
  --original-android-base /path/to/Android \
  --original-pc-base /path/to/PC \
  --judge-model <judge-model> \
  --judge-api-key "$JUDGE_API_KEY" \
  --judge-base-url "$JUDGE_BASE_URL"
```

Typical outputs are written under a run directory and include trajectory logs, per-mask evaluation JSON files, platform-level summaries, and an overall `summary.json`.

## Quick Start: Grounding Evaluation

The grounding evaluator expects:

- A manifest JSONL with sample IDs, image names, plans, and target actions.
- One or more box JSONL files with `bbox_xyxy` annotations.
- One image root per evaluated image variant.

Example:

```bash
grounding-eval \
  --manifest /path/to/manifest.jsonl \
  --boxes /path/to/manual_click_boxes.jsonl \
  --image-root original=/path/to/images \
  --image-root black=/path/to/black \
  --image-root replace_llm=/path/to/replace_llm \
  --models <model-a> <model-b> \
  --masks original black replace_llm \
  --out-dir outputs/grounding_eval \
  --workers 4
```

The output directory contains:

```text
outputs/grounding_eval/
├── samples.json
├── summary.json
└── <model>/
    ├── original.jsonl
    ├── black.jsonl
    └── replace_llm.jsonl
```

`summary.json` reports point accuracy, parse rate, and prediction format statistics per model and image variant.

## Evaluation Protocols

### Privacy Recognition

Privacy recognition evaluates whether models can identify sensitive regions and privacy labels in GUI screenshots. The benchmark annotations provide region-level grounding and fine-grained privacy metadata.

### Protected-Screen Planning

Protected-screen planning evaluates whether agents can preserve task semantics when screenshots are transformed by privacy protection methods such as masking or text replacement.

The trajectory-level score compares a replay plan against a ground-truth plan using a five-point semantic consistency scale:

```text
0: Completely inconsistent
1: Minimally consistent
2: Partially consistent
3: Mostly consistent
4: Fully consistent
```

### ScreenSpot-Style Grounding

Grounding evaluation asks a model to return one click point for the target UI element. A prediction is correct when the point lies inside the ground-truth bounding box.

Supported coordinate modes:

| Mode | Meaning |
|---|---|
| `normal_1000` | Model returns coordinates in a normalized `1000 x 1000` frame. |
| `pixel` | Model returns actual image pixel coordinates. |
| `claude_android` | Model returns Android coordinates in a `705 x 1567` frame, then scaled to image pixels. |

## Reproducibility Notes

- Keep API keys in environment variables or local `.env` files.
- Do not commit raw model responses if they contain sensitive data.
- Do not commit private paths or service endpoints in config files.
- Keep generated runs and large dataset files outside the source tree.
- Use the example config files as templates and replace paths locally.

## Citation

Citation metadata is intentionally omitted in this anonymous-review version. If this work is accepted or the review policy allows de-anonymization, replace this section with the final citation.

## License

The dataset page lists the dataset license as MIT. Check the dataset repository and individual files for the latest license and usage terms.
