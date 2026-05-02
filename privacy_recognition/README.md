# Privacy Recognition

This folder contains the GUIGuard-Bench privacy-recognition code used to evaluate whether vision-language models can identify, localize, and label privacy-sensitive GUI regions.

It corresponds to the `Privacy Recognition` section of the GUIGuard paper. The release is intentionally code-only: raw screenshots, private trajectories, generated annotations, API keys, and local cluster paths are not included.

## Scope

Included:

- Multimodal privacy-recognition prompting for GUI screenshots.
- Batch inference over Android-style and PC-style trajectory folders.
- Parsing model outputs into region-level JSON labels.
- IoU-only and paper-protocol privacy-recognition evaluation scripts.

Excluded on purpose:

- Real screenshots and trajectories.
- Generated annotations or prediction outputs.
- API credentials and provider-specific private endpoints.
- Local machine, cluster, or user-specific paths.

## Install

```bash
cd privacy_recognition
python -m pip install -e .
```

## API Configuration

The inference script uses the OpenAI Python SDK against an OpenAI-compatible chat-completions endpoint.

Set credentials through environment variables only:

```bash
export OPENAI_API_KEY="<your_api_key>"
export OPENAI_BASE_URL="<optional_openai_compatible_base_url>"
```

If `OPENAI_BASE_URL` is omitted, the SDK default endpoint is used.

## Input Data Layout

Android-style task:

```text
<task_folder>/
  task_result.json
  traj.jsonl
  images/
    *.png
```

PC-style task:

```text
<task_folder>/
  instruction.txt
  traj.jsonl
  step_0.png
  step_1.png
```

The batch runner accepts either a single task folder or a dataset root containing multiple task folders.

## Run Recognition

```bash
python run_dataset.py <task_or_dataset_root> \
  --model "<model_name>" \
  --output-root outputs/predictions
```

For a single Android-style task, the lightweight runner is also available:

```bash
python privacy.py <task_folder> --model "<model_name>"
```

Outputs are written under `outputs/` by default and are ignored by git.

## Evaluation

IoU-only evaluation:

```bash
python experiments/evaluate_privacy_recognition_iou.py \
  --gt <ground_truth_json> \
  --android-root <android_dataset_root> \
  --pc-root <pc_dataset_root> \
  --pred-root outputs/predictions \
  --output outputs/metrics/privacy_recognition_iou.json
```

This script matches privacy regions by bounding-box IoU only. It is useful for quick region-level debugging.

Paper-style evaluation:

```bash
python experiments/evaluate_privacy_recognition_paper.py \
  --gt <ground_truth_json> \
  --android-root <android_dataset_root> \
  --pc-root <pc_dataset_root> \
  --pred-root outputs/predictions \
  --output outputs/metrics/privacy_recognition_paper.json
```

This script follows the paper protocol: a predicted privacy element must match both text and location before label accuracy is counted.

## Output Format

Prediction JSON files contain one item per screenshot:

```json
{
  "batchId": "task_name",
  "index": 0,
  "file": "screenshot.png",
  "width": 1080,
  "height": 2400,
  "labels": [
    {
      "id": 1,
      "risk": "High risk",
      "category": "2",
      "points": [80, 250, 740, 350],
      "text": "visible private text",
      "necessary": true
    }
  ]
}
```

## Notes For Publishing

- Do not commit `.env`, API keys, raw screenshots, trajectories, predictions, or annotations.
- Use command-line arguments for local dataset paths.
- Keep `outputs/` and `annotations/` out of source control unless you intentionally release a public, sanitized artifact.
