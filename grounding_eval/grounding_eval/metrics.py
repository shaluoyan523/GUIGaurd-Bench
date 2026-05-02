from __future__ import annotations

from .types import BBox


def normalize_point(
    raw_point: list[float] | None,
    raw_bbox: list[float] | None,
    *,
    width: int,
    height: int,
    mode: str,
) -> tuple[list[float] | None, list[float] | None]:
    if raw_point is None:
        return None, None

    if mode == "claude_android":
        denom_x, denom_y = 705.0, 1567.0
    else:
        denom_x, denom_y = 1000.0, 1000.0

    def convert_pair(x: float, y: float) -> list[float]:
        if mode == "claude_android" and (width, height) != (1080, 2400):
            if 0 <= x <= width and 0 <= y <= height:
                return [x, y]
        if mode == "pixel":
            if 0 <= x <= 1 and 0 <= y <= 1:
                return [x * width, y * height]
            return [x, y]
        if 0 <= x <= 1 and 0 <= y <= 1:
            return [x * width, y * height]
        if 0 <= x <= width and 0 <= y <= height and (x > denom_x or y > denom_y):
            return [x, y]
        return [x * width / denom_x, y * height / denom_y]

    point = convert_pair(raw_point[0], raw_point[1])
    bbox = None
    if raw_bbox:
        x1, y1 = convert_pair(raw_bbox[0], raw_bbox[1])
        x2, y2 = convert_pair(raw_bbox[2], raw_bbox[3])
        bbox = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
    return point, bbox


def point_in_bbox(point: list[float] | None, bbox: BBox) -> bool:
    if not point:
        return False
    x, y = point
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2


def iou(a: list[float] | None, b: BBox) -> float:
    if not a:
        return 0.0
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0
