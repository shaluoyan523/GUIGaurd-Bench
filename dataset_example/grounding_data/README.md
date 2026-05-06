# Grounding Data Example

This folder contains 290 ScreenSpot-style click-grounding samples derived from the dataset example.

Images are renamed in zero-based order from `000.jpg` to `289.jpg`. Each sample keeps `original_id` for traceability to the source ordering `1-290`. The JPEG files preserve the original pixel dimensions, so bbox coordinates remain valid.

Layout:

```text
grounding_data/
  images/original/*.jpg
  images/black/*.jpg
  images/replace_llm/*.jpg
  manifest.jsonl
  boxes.jsonl
  samples.json
```

`manifest.jsonl` stores the next-action plan and action text. `boxes.jsonl` stores the target bbox in `[x1, y1, x2, y2]` pixel coordinates. `samples.json` is a compact combined view for inspection.
