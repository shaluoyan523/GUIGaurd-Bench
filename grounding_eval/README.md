# Grounding Eval

Open-source friendly ScreenSpot-style GUI grounding evaluation for click-point prediction.

The evaluator sends a screenshot and the next-action plan to a vision-language model, asks for a click point, normalizes model-specific coordinate frames, and scores whether the predicted point falls inside the ground-truth bounding box.

## Features

- ScreenSpot-style point accuracy: `predicted point inside GT bbox`.
- Multiple image variants per sample, such as `original`, `black`, and `replace_llm`.
- OpenAI-compatible chat completions API.
- Concurrent evaluation with resume support.
- Model-specific coordinate normalization:
  - `normal_1000`: normalized `1000 x 1000` coordinate frame.
  - `claude_android`: Claude Android frame `705 x 1567`, mapped to real image pixels.
  - `pixel`: model returns actual pixel coordinates.
- Robust output parsing for JSON points, bbox center fallbacks, loose coordinate text, and GUI-Owl-style responses.

## Install

```bash
cd grounding_eval
python -m pip install -e .
```

For tests:

```bash
python -m pip install -e '.[dev]'
pytest
```

## Data Format

The evaluator expects one manifest JSONL and one or more box JSONL files.

Manifest rows should include:

```json
{
  "id": 1,
  "copied_image_name": "example.png",
  "plan": "(Previous action verification)...",
  "actions": ["# Click: Settings button"]
}
```

Box rows should include:

```json
{
  "id": 1,
  "action": "# Click: Settings button",
  "bbox_xyxy": [100, 200, 300, 260],
  "target_visible": true
}
```

Images are resolved from per-mask image roots:

```text
original/example.png
black/example.png
replace_llm/example.png
```

## Run

Create an environment file or export credentials:

```bash
export OPENAI_API_KEY='...'
export OPENAI_BASE_URL='https://api.openai.com/v1'
```

Run with a config:

```bash
grounding-eval --config configs/example.json
```

Or run directly:

```bash
grounding-eval \
  --manifest /path/to/manifest.jsonl \
  --boxes /path/to/manual_click_boxes.jsonl \
  --image-root original=/path/to/images \
  --image-root black=/path/to/black \
  --image-root replace_llm=/path/to/replace_llm \
  --models qwen3.5-397b-a17b claude-sonnet-4-5-20250929 gemini-3.1-pro-preview \
  --masks original black replace_llm \
  --out-dir outputs/grounding_eval \
  --workers 4
```

## Outputs

The output directory contains:

```text
outputs/grounding_eval/
  samples.json
  summary.json
  <model>/
    original.jsonl
    black.jsonl
    replace_llm.jsonl
```

Each per-sample JSONL row includes:

- `raw_response`
- `parse_ok`
- `prediction_kind`
- `raw_point`
- `pred_point_xy`
- `gt_bbox_xyxy`
- `point_in_gt_bbox`
- `iou`
- `latency_s`

`summary.json` reports count, evaluated count, correct count, point accuracy, parse rate, and prediction-kind distribution per model and mask.

## Coordinate Normalization

By default, normalization is inferred from model names:

| Model name contains | Mode | Meaning |
|---|---|---|
| `claude` | `claude_android` | Raw coordinates are in `705 x 1567`, then scaled to actual image size. |
| `gui-owl` or `ui-tars` | `pixel` | Raw coordinates are actual image pixels. |
| anything else | `normal_1000` | Raw coordinates are in normalized `1000 x 1000`. |

Override per model in config:

```json
{
  "norm_modes": {
    "my-model": "pixel"
  },
  "parse_modes": {
    "my-model": "default"
  }
}
```

Allowed normalization modes are `normal_1000`, `claude_android`, and `pixel`.

## Prompt

The default prompt is:

```text
You are evaluating GUI grounding like ScreenSpot. Given the screenshot and the plan for the next action, locate the exact point that should be clicked.

{coordinate_frame_instruction}
Return strict JSON only in this schema: {"x": number, "y": number}. The point must be inside the target UI element. No markdown, no commentary.

Target action: {sample.action}
Target element: {sample.target}

Plan:
{sample.plan}
```

## Notes For Publishing

- Do not commit `.env` or any real API key.
- Do not commit raw screenshots if the dataset license or privacy policy does not allow it.
- Prefer publishing config templates with placeholder paths.
- Keep result JSONL files out of the source distribution unless they are explicitly intended as benchmark outputs.
