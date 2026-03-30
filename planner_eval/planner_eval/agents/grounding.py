from __future__ import annotations

from typing import List


class ACI:
    """Minimal interface used by the trajectory evaluator.

    The full desktop-control implementation is intentionally excluded from the
    open-source package. For trajectory evaluation we only need a symbolic
    action interface and a simple note buffer.
    """

    def __init__(self) -> None:
        self.notes: List[str] = []


def agent_action(func):
    func.is_agent_action = True
    return func
