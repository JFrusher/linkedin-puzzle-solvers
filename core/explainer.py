"""Human-readable deduction logs (explainable AI) for solver runs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class DeductionStep:
    """One reasoning step, e.g. ``Step 2 [Single Choice]: Cell (3,4) must be a Queen ...``."""

    index: int
    rule: str
    message: str
    cell: tuple[int, int] | None = None

    def __str__(self) -> str:
        return f"Step {self.index} [{self.rule}]: {self.message}"


class Explainer:
    """Collects deduction steps and serialises them as JSON or Markdown."""

    def __init__(self) -> None:
        self.steps: list[DeductionStep] = []

    def add(self, rule: str, message: str, cell: tuple[int, int] | None = None) -> DeductionStep:
        """Append a step and return it."""
        step = DeductionStep(index=len(self.steps) + 1, rule=rule, message=message, cell=cell)
        self.steps.append(step)
        return step

    def __len__(self) -> int:
        return len(self.steps)

    def to_json(self) -> str:
        """Serialise the log as a JSON array of step objects."""
        return json.dumps([asdict(step) for step in self.steps], indent=2)

    def to_markdown(self) -> str:
        """Render the log as a Markdown table."""
        lines = ["| Step | Rule | Cell | Deduction |", "|---:|---|---|---|"]
        for step in self.steps:
            cell = "" if step.cell is None else f"({step.cell[0]},{step.cell[1]})"
            message = step.message.replace("|", "\\|")
            lines.append(f"| {step.index} | {step.rule} | {cell} | {message} |")
        return "\n".join(lines)

    def to_text(self) -> str:
        """Render the log as one line per step."""
        return "\n".join(str(step) for step in self.steps)
