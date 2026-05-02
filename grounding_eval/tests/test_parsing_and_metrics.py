from __future__ import annotations

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
