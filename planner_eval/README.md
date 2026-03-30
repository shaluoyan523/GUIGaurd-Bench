# planner_eval

`planner_eval` is a cleaned, self-contained extraction of the trajectory-evaluation workflow from the original `s3_planner` codebase.

It is designed for GitHub release and focuses on one reproducible pipeline:

1. Run a planning model on prerecorded screenshot trajectories.
2. Compare original trajectories against multiple masked variants.
3. Score step-level semantic consistency with an LLM-as-Judge.

## Scope

This package only keeps the pieces needed for trajectory-based evaluation.

Included:

- OpenAI-compatible multimodal planner inference
- Symbolic trajectory execution (`agent.click(...)`, `agent.done()`, etc.)
- Batch evaluation across original and masked datasets
- Semantic-consistency judge scoring
- Result summaries and per-mask reports

Excluded on purpose:

- Full desktop-control runtime
- OCR / grounding server code
- Local model-serving scripts
- Experiment-specific logs, checkpoints, and private paths

## Repository Layout

```text
planner_eval/
├── configs/
│   └── model_presets.example.json
├── planner_eval/
│   ├── agents/
│   ├── core/
│   ├── judges/
│   ├── memory/
│   ├── utils/
│   ├── model_presets.py
│   ├── pipeline.py
│   ├── simple_grounding.py
│   └── trajectory.py
├── scripts/
│   ├── run_full_pipeline.sh
│   └── run_trajectory.sh
├── .env.example
└── pyproject.toml
```

## Installation

```bash
cd planner_eval
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Dataset Assumptions

Original Android dataset, flat layout:

```text
Android/
└── <task_name>/
    ├── images/
    │   ├── screenshot_001.png
    │   └── ...
    └── instruction.txt
```

Original PC dataset, flat layout:

```text
PC/
└── <task_name>/
    ├── screenshot_001.png
    ├── screenshot_002.png
    └── instruction.txt
```

Nested public-split layout is also supported:

```text
Android_public/
└── Android_EN/
    └── AI/
        └── task_0054/
            ├── images/
            └── task_result.json

PC_public/
└── AI/
    └── task_0001/
        ├── instruction.txt
        ├── step_0.png
        └── ...
```

Masked dataset root:

```text
mask_root/
├── output_black_mask/
│   ├── Android/<task_name>/images/...
│   └── PC/<task_name>/...
├── output_mosaic_mask/
├── output_randblocks_mask/
└── output_replace_llm_mask/
```

## Planner Model Configuration

You can either pass a model directly:

```bash
planner-eval-pipeline \
  --model gpt-5 \
  --api-key "$OPENAI_API_KEY" \
  --mask-dataset-root /data/mask_4 \
  --original-android-base /data/Android \
  --original-pc-base /data/PC
```

Or use a preset file:

```bash
planner-eval-pipeline \
  --model-preset openai_gpt5 \
  --model-presets-file configs/model_presets.example.json \
  --mask-dataset-root /data/mask_4 \
  --original-android-base /data/Android \
  --original-pc-base /data/PC
```

## Judge Model Configuration

Judge scoring uses a separate OpenAI-compatible endpoint.

```bash
planner-eval-pipeline \
  --model gpt-5 \
  --api-key "$OPENAI_API_KEY" \
  --judge-model gpt-5 \
  --judge-api-key "$JUDGE_API_KEY" \
  --judge-base-url "$JUDGE_BASE_URL" \
  --mask-dataset-root /data/mask_4 \
  --original-android-base /data/Android \
  --original-pc-base /data/PC
```

If `--judge-api-key` is omitted, the evaluation stage is skipped unless you pass `--skip-evaluation`.

For nested datasets, `--android-task` and `--pc-task` accept either:

- the full relative task path, for example `Android_EN/AI/task_0054`
- a unique leaf task name, for example `task_0054`

## Running A Small Slice

```bash
planner-eval-pipeline \
  --model gpt-5 \
  --api-key "$OPENAI_API_KEY" \
  --judge-model gpt-5 \
  --judge-api-key "$JUDGE_API_KEY" \
  --mask-dataset-root /data/mask_4 \
  --original-android-base /data/Android \
  --android-task "20251031_120453_Check logged-in Google account" \
  --android-task "20251031_122836_Google Docs_ Open any document"
```

## Outputs

Each run creates a timestamped directory under `runs/`:

```text
runs/<run_name>/
├── combined.log
├── original_android/
├── original_pc/
├── masked_output_black_mask_android/
├── masked_output_black_mask_pc/
├── ...
├── evaluation/
│   ├── android/
│   └── pc/
├── summary.json
└── test_summary.txt
```

The judge stage writes:

- per-mask `*_evaluation.json`
- platform-level `overall_summary.json`
- optional `*_judge_api.log`

## Notes

- The pipeline is trajectory-based, not live GUI control.
- Grounded actions are symbolic no-ops used only for evaluation.
- Task names are passed as repeated CLI flags, so names containing commas are safe.
