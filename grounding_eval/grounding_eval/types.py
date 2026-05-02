from __future__ import annotations

from dataclasses import dataclass
from typing import Any


BBox = tuple[float, float, float, float]


@dataclass(frozen=True)
class Sample:
    """One ScreenSpot-style grounding sample."""

    id: int
    image_name: str
    plan: str
    action: str
    target: str
    bbox_xyxy: BBox
    target_visible: bool
    image_paths: dict[str, str]

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "image_name": self.image_name,
            "plan": self.plan,
            "action": self.action,
            "target": self.target,
            "bbox_xyxy": list(self.bbox_xyxy),
            "target_visible": self.target_visible,
            "image_paths": self.image_paths,
        }


@dataclass(frozen=True)
class EvalResult:
    """Serializable per-sample evaluation output."""

    row: dict[str, Any]
