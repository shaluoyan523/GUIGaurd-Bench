from __future__ import annotations

from typing import Any, Dict, List, Optional

from planner_eval.agents.grounding import ACI, agent_action


class SimpleACI(ACI):
    """Symbolic action space for trajectory-only evaluation."""

    def __init__(self) -> None:
        super().__init__()
        self.obs = None
        self.current_task_instruction = None
        self.last_code_agent_result = None
        self.platform = "linux"

    def assign_screenshot(self, obs):
        self.obs = obs

    def set_task_instruction(self, task_instruction: str):
        self.current_task_instruction = task_instruction

    @agent_action
    def click(
        self,
        element_description: str,
        num_clicks: int = 1,
        button_type: str = "left",
        hold_keys: List = [],
    ):
        """Click on a UI element."""
        return f"# Click: {element_description}"

    @agent_action
    def type(
        self,
        element_description: Optional[str] = None,
        text: str = "",
        overwrite: bool = False,
        enter: bool = False,
    ):
        """Type text into a UI element."""
        return f"# Type: {text}"

    @agent_action
    def scroll(self, element_description: str, clicks: int, shift: bool = False):
        """Scroll within a UI element."""
        return f"# Scroll: {clicks} clicks"

    @agent_action
    def drag_and_drop(
        self,
        starting_description: str,
        ending_description: str,
        hold_keys: List = [],
    ):
        """Drag from one location to another."""
        return f"# Drag from {starting_description} to {ending_description}"

    @agent_action
    def highlight_text_span(
        self,
        starting_phrase: str,
        ending_phrase: str,
        button: str = "left",
    ):
        """Highlight a span of text."""
        return f"# Highlight: {starting_phrase} to {ending_phrase}"

    @agent_action
    def hotkey(self, keys: List):
        """Press a key combination."""
        return f"# Hotkey: {'+'.join(keys)}"

    @agent_action
    def hold_and_press(self, hold_keys: List, press_keys: List):
        """Hold keys and press another key sequence."""
        return f"# Hold {hold_keys} and press {press_keys}"

    @agent_action
    def open(self, app_or_filename: str):
        """Open an application or file."""
        return f"# Open: {app_or_filename}"

    @agent_action
    def switch_applications(self, app_code: str):
        """Switch to another application."""
        return f"# Switch to: {app_code}"

    @agent_action
    def set_cell_values(self, cell_values: Dict[str, Any], app_name: str, sheet_name: str):
        """Set cell values in a spreadsheet."""
        return f"# Set cells in {app_name}/{sheet_name}"

    @agent_action
    def call_code_agent(self, task: str = None):
        """Request code-agent execution for a task."""
        return f"# Code agent: {task or 'full task'}"

    @agent_action
    def wait(self, time: float):
        """Wait for a period of time."""
        return f"import time; time.sleep({time})"

    @agent_action
    def done(self):
        """Mark the task as completed."""
        return "DONE"

    @agent_action
    def fail(self):
        """Mark the task as failed."""
        return "FAIL"

    @agent_action
    def next(self):
        """Advance without taking an action."""
        return "NEXT"

    @agent_action
    def save_to_knowledge(self, text: List[str]):
        """Save notes into the agent memory buffer."""
        self.notes.extend(text)
        return "WAIT"
