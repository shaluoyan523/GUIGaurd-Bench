from __future__ import annotations

from .types import Sample


def infer_norm_mode(model: str) -> str:
    model_l = model.lower()
    if "claude" in model_l:
        return "claude_android"
    if "gui-owl" in model_l or "ui-tars" in model_l:
        return "pixel"
    return "normal_1000"


def infer_parse_mode(model: str) -> str:
    model_l = model.lower()
    if "gui-owl" in model_l:
        return "x_list_point"
    return "default"


def prompt_for(
    model: str,
    sample: Sample,
    *,
    width: int | None = None,
    height: int | None = None,
    norm_mode: str | None = None,
) -> str:
    mode = norm_mode or infer_norm_mode(model)
    if mode == "claude_android":
        frame = "Use a coordinate frame of width 705 and height 1567."
    elif mode == "pixel" and width and height:
        frame = f"Use the screenshot's actual pixel coordinate frame of width {width} and height {height}."
    else:
        frame = "Use a normalized coordinate frame of width 1000 and height 1000."

    return (
        "You are evaluating GUI grounding like ScreenSpot. Given the screenshot "
        "and the plan for the next action, locate the exact point that should be clicked.\n\n"
        f"{frame}\n"
        "Return strict JSON only in this schema: {\"x\": number, \"y\": number}. "
        "The point must be inside the target UI element. No markdown, no commentary.\n\n"
        f"Target action: {sample.action}\n"
        f"Target element: {sample.target}\n\n"
        f"Plan:\n{sample.plan}"
    )
