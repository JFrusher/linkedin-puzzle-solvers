"""Difficulty from the deduction log: the hardest rule a solve needs, as Sudoku graders rate puzzles.

A board is as hard as the hardest step a person must take. Within a band, how often that rule fires orders boards:
a Queens board that needs Shared Attack seven times is harder than one that needs it once.
"""

from __future__ import annotations

from collections import Counter

from core.explainer import DeductionStep

TIERS: dict[str, int] = {
    "Given": 0,
    # Look at one cell, line or clue.
    "Single Choice": 1,
    "No Three": 1,
    "Relation": 1,
    "Balance": 1,
    "Naked Single": 1,
    "Only Fit": 1,
    "Only Word": 1,
    "Nook": 1,
    # Combine several lines, regions or candidates.
    "Shared Attack": 2,
    "Confinement": 2,
    "Sign Balance": 2,
    "Equal Pair Neighbour": 2,
    "Hidden Single": 2,
    "Only Cover": 2,
    # Assume a move and follow it to a contradiction.
    "Try and Fail": 3,
    # No rule applies; the solver branches.
    "Search": 4,
    "SAT": 4,
}
LEVELS = ("Trivial", "Easy", "Medium", "Hard", "Expert")


def rate(steps: list[DeductionStep]) -> str:
    """``'Medium: Shared Attack ×7'``: the band of the hardest rule used, and how often each rule in it fired."""
    if not steps:
        raise ValueError("an empty deduction log has no difficulty")
    unknown = {step.rule for step in steps} - TIERS.keys()
    if unknown:
        raise KeyError(f"no difficulty tier for rules {sorted(unknown)}")
    hardest = max(TIERS[step.rule] for step in steps)
    fired = Counter(step.rule for step in steps if TIERS[step.rule] == hardest)
    return f"{LEVELS[hardest]}: " + ", ".join(f"{rule} ×{count}" for rule, count in fired.most_common())
