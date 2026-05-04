from __future__ import annotations

import json

from PIL import Image

from grounding_eval.data import build_samples_from_example
from grounding_eval.metrics import normalize_point, point_in_bbox
from grounding_eval.parsing import parse_prediction


def test_parse_xy_json() -> None:
    parsed = parse_prediction('{"x": 500, "y": 250}')
    assert parsed["ok"] is True
    assert parsed["raw_point"] == [500.0, 250.0]
    assert parsed["prediction_kind"] == "xy_point"


def test_parse_bbox_center() -> None:
    parsed = parse_prediction('{"bbox": [10, 20, 30, 40]}')
    assert parsed["ok"] is True
    assert parsed["raw_point"] == [20.0, 30.0]
    assert parsed["raw_bbox"] == [10, 20, 30, 40]


def test_parse_gui_owl_x_list() -> None:
    parsed = parse_prediction('{"x": [123, 456], "y": [123, 456]}', mode="x_list_point")
    assert parsed["ok"] is True
    assert parsed["raw_point"] == [123.0, 456.0]


def test_normal_1000_coordinates() -> None:
    point, _ = normalize_point([500, 250], None, width=1080, height=2400, mode="normal_1000")
    assert point == [540.0, 600.0]


def test_claude_android_coordinates() -> None:
    point, _ = normalize_point([352.5, 783.5], None, width=1080, height=2400, mode="claude_android")
    assert point == [540.0, 1200.0]


def test_pixel_coordinates_and_hit() -> None:
    point, _ = normalize_point([100, 200], None, width=1080, height=2400, mode="pixel")
    assert point == [100, 200]
    assert point_in_bbox(point, (50, 100, 150, 250))


def test_build_samples_from_dataset_example(tmp_path) -> None:
    example_root = tmp_path / "dataset_example"
    image_dir = example_root / "Android" / "2" / "images"
    image_dir.mkdir(parents=True)
    image_path = image_dir / "screen.png"
    Image.new("RGB", (100, 200), "white").save(image_path)
    (example_root / "Android" / "2" / "task_result.json").write_text(
        json.dumps({"goal": "Open the settings screen"}),
        encoding="utf-8",
    )
    (example_root / "image_privacy_labels_example.json").write_text(
        json.dumps(
            [
                {
                    "info": "dataset_example/Android/2/images/screen.png",
                    "labels": [
                        {
                            "label": "low risk",
                            "points": [10, 20, 30, 50],
                            "attr": {"ocrResult": "Settings"},
                        }
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )

    samples = build_samples_from_example(example_root=example_root)

    assert len(samples) == 1
    assert samples[0].image_name == "Android/2/images/screen.png"
    assert samples[0].plan == "Open the settings screen"
    assert samples[0].bbox_xyxy == (10.0, 20.0, 30.0, 50.0)
    assert samples[0].target == "low risk | Settings"
    assert samples[0].image_paths["original"] == str(image_path.resolve())
