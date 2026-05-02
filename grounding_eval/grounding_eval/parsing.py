from __future__ import annotations

import json
import re
from typing import Any


def extract_jsonish(text: str) -> Any:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        return json.loads(stripped)
    except Exception:
        pass
    match = re.search(r"(\{.*\}|\[.*\])", stripped, re.S)
    if match:
        return json.loads(match.group(1))
    raise ValueError("No JSON object found")


def numeric_list(value: Any) -> list[float] | None:
    if isinstance(value, list) and len(value) >= 2:
        nums: list[float] = []
        for item in value:
            if not isinstance(item, (int, float)):
                return None
            nums.append(float(item))
        return nums
    return None


def numeric_scalar(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], (int, float)):
        return float(value[0])
    return None


def xy_fields_to_point(x_value: Any, y_value: Any) -> list[float] | None:
    x = numeric_scalar(x_value)
    y = numeric_scalar(y_value)
    if x is not None and y is not None:
        return [x, y]

    x_nums = numeric_list(x_value)
    y_nums = numeric_list(y_value)
    if x_nums and y_nums:
        if x_nums == y_nums and len(x_nums) >= 2:
            return x_nums[:2]
        return [x_nums[0], y_nums[0]]
    return None


def find_prediction(value: Any) -> dict[str, Any] | None:
    """Prefer point predictions; accept bbox outputs as center-point fallback."""
    if isinstance(value, dict):
        for key in ["point", "click_point", "target_point", "position", "center", "click"]:
            if key in value:
                nums = numeric_list(value[key])
                if nums:
                    return {"raw_point": nums[:2], "raw_bbox": None, "prediction_kind": f"{key}_point"}
                nested = find_prediction(value[key])
                if nested:
                    return nested

        if all(k in value for k in ["x", "y"]):
            point = xy_fields_to_point(value["x"], value["y"])
            if not point:
                raise ValueError("x/y fields are present but not numeric")
            return {"raw_point": point, "raw_bbox": None, "prediction_kind": "xy_point"}

        for key in ["bbox_2d", "bbox", "box", "bounding_box"]:
            if key in value:
                nums = numeric_list(value[key])
                if nums and len(nums) >= 4:
                    x1, y1, x2, y2 = nums[:4]
                    return {
                        "raw_point": [(x1 + x2) / 2, (y1 + y2) / 2],
                        "raw_bbox": [x1, y1, x2, y2],
                        "prediction_kind": f"{key}_center_fallback",
                    }
                nested = find_prediction(value[key])
                if nested:
                    return nested

        if all(k in value for k in ["x1", "y1", "x2", "y2"]):
            x1, y1, x2, y2 = [float(value[k]) for k in ["x1", "y1", "x2", "y2"]]
            return {
                "raw_point": [(x1 + x2) / 2, (y1 + y2) / 2],
                "raw_bbox": [x1, y1, x2, y2],
                "prediction_kind": "xyxy_center_fallback",
            }

        for key in ["coordinates", "coord"]:
            if key in value:
                nums = numeric_list(value[key])
                if nums:
                    if len(nums) == 2:
                        return {"raw_point": nums[:2], "raw_bbox": None, "prediction_kind": f"{key}_point"}
                    x1, y1, x2, y2 = nums[:4]
                    return {
                        "raw_point": [(x1 + x2) / 2, (y1 + y2) / 2],
                        "raw_bbox": [x1, y1, x2, y2],
                        "prediction_kind": f"{key}_center_fallback",
                    }
                nested = find_prediction(value[key])
                if nested:
                    return nested

        for child in value.values():
            result = find_prediction(child)
            if result:
                return result

    if isinstance(value, list):
        nums = numeric_list(value)
        if nums:
            if len(nums) == 2:
                return {"raw_point": nums[:2], "raw_bbox": None, "prediction_kind": "list_point"}
            x1, y1, x2, y2 = nums[:4]
            return {
                "raw_point": [(x1 + x2) / 2, (y1 + y2) / 2],
                "raw_bbox": [x1, y1, x2, y2],
                "prediction_kind": "list_center_fallback",
            }
        for child in value:
            result = find_prediction(child)
            if result:
                return result
    return None


def parse_x_list_point(text: str) -> list[float] | None:
    match = re.search(
        r'"x"\s*:\s*\[\s*([-+]?(?:\d+\.\d+|\d+))\s*,\s*([-+]?(?:\d+\.\d+|\d+))',
        text,
    )
    if match:
        return [float(match.group(1)), float(match.group(2))]
    return None


def parse_malformed_xy(text: str) -> list[float] | None:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    if '"x"' not in stripped:
        return None
    numbers = [float(x) for x in re.findall(r"[-+]?(?:\d+\.\d+|\d+)", stripped)]
    if len(numbers) == 2:
        return numbers
    if len(numbers) >= 3 and '"y"' in stripped:
        return [numbers[0], numbers[-1]]
    return None


def parse_loose_point(text: str) -> list[float] | None:
    patterns = [
        r"start_box\s*=\s*['\"]?\(?\s*([-+]?(?:\d+\.\d+|\d+))\s*,\s*([-+]?(?:\d+\.\d+|\d+))",
        r"click\s*\([^)]*?\(?\s*([-+]?(?:\d+\.\d+|\d+))\s*,\s*([-+]?(?:\d+\.\d+|\d+))",
        r'"bbox_2d"\s*:\s*\[\s*([-+]?(?:\d+\.\d+|\d+))\s*,\s*([-+]?(?:\d+\.\d+|\d+))',
        r'"x"\s*:\s*([-+]?(?:\d+\.\d+|\d+))\s*,\s*([-+]?(?:\d+\.\d+|\d+))',
        r"\[\s*([-+]?(?:\d+\.\d+|\d+))\s*,\s*([-+]?(?:\d+\.\d+|\d+))\s*\]",
        r"\(\s*([-+]?(?:\d+\.\d+|\d+))\s*,\s*([-+]?(?:\d+\.\d+|\d+))\s*\)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I)
        if match:
            return [float(match.group(1)), float(match.group(2))]
    return None


def parse_prediction(text: str, mode: str = "default") -> dict[str, Any]:
    try:
        if mode == "x_list_point":
            x_list_point = parse_x_list_point(text)
            if x_list_point:
                return {
                    "ok": True,
                    "raw_bbox": None,
                    "raw_point": x_list_point,
                    "prediction_kind": "x_list_point",
                }
        data = extract_jsonish(text)
        prediction = find_prediction(data)
        if not prediction:
            raise ValueError("No numeric point coordinate found")
        return {"ok": True, **prediction}
    except Exception as exc:
        malformed_point = parse_malformed_xy(text)
        if malformed_point:
            return {
                "ok": True,
                "raw_bbox": None,
                "raw_point": malformed_point,
                "prediction_kind": "malformed_xy_point",
            }
        loose_point = parse_loose_point(text)
        if loose_point:
            return {
                "ok": True,
                "raw_bbox": None,
                "raw_point": loose_point,
                "prediction_kind": "loose_point",
            }
        return {
            "ok": False,
            "error": str(exc),
            "raw_bbox": None,
            "raw_point": None,
            "prediction_kind": None,
        }
