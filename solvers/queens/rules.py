"""Human-style Queens deductions, applied one at a time until none fires.

Each rule inspects the remaining candidate cells and returns the first deduction it can prove:

* **Single Choice** — a row, column or region has one candidate left: it holds the queen.
* **Shared Attack** — every candidate of a unit attacks some outside cell (same row, column, region or
  touching), so that cell can never hold a queen. Covers the 2x2, L and T shapes.
* **Confinement** — k regions whose candidates lie inside k rows (or columns) own those lines, so every
  other cell in them is out; likewise k lines whose candidates lie inside k regions own those regions.
  The 2x1 domino confined to one row is the k = 1 case.
* **Try and Fail** — placing a queen on a candidate and following only Single Choice leaves some unit
  with no candidate, so that candidate is out.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import combinations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from solvers.queens.solver import QueensBoard

Cell = tuple[int, int]


@dataclass(frozen=True)
class Deduction:
    rule: str
    reason: str
    place: Cell | None = None
    eliminate: frozenset[Cell] = frozenset()


@dataclass
class QueensState:
    board: QueensBoard
    candidates: set[Cell] = field(default_factory=set)
    placed: set[Cell] = field(default_factory=set)

    @classmethod
    def start(cls, board: QueensBoard) -> QueensState:
        n = board.size
        return cls(board, {(r, c) for r in range(n) for c in range(n)})

    def region(self, cell: Cell) -> int:
        return self.board.regions[cell[0]][cell[1]]

    def attacks(self, a: Cell, b: Cell) -> bool:
        """True when queens on ``a`` and ``b`` would break a rule (``a != b``)."""
        return (
            a[0] == b[0]
            or a[1] == b[1]
            or self.region(a) == self.region(b)
            or (abs(a[0] - b[0]) <= 1 and abs(a[1] - b[1]) <= 1)
        )

    def units(self) -> list[tuple[str, set[Cell]]]:
        """Unfinished rows, columns and regions with their remaining candidates."""
        n = self.board.size
        done_rows = {r for r, _ in self.placed}
        done_cols = {c for _, c in self.placed}
        done_regions = {self.region(cell) for cell in self.placed}
        units = [(f"Row {r}", {x for x in self.candidates if x[0] == r}) for r in range(n) if r not in done_rows]
        units += [(f"Column {c}", {x for x in self.candidates if x[1] == c}) for c in range(n) if c not in done_cols]
        units += [
            (f"Region {g}", {x for x in self.candidates if self.region(x) == g})
            for g in range(n)
            if g not in done_regions
        ]
        return units

    def apply(self, deduction: Deduction) -> None:
        if deduction.place is not None:
            cell = deduction.place
            self.placed.add(cell)
            self.candidates = {x for x in self.candidates if x != cell and not self.attacks(cell, x)}
        self.candidates -= deduction.eliminate

    def solved(self) -> bool:
        return len(self.placed) == self.board.size

    def stuck(self) -> bool:
        """True when some unfinished unit has no candidate left (a contradiction)."""
        return any(not cells for _, cells in self.units())


def single_choice(state: QueensState) -> Deduction | None:
    for name, cells in state.units():
        if len(cells) == 1:
            (cell,) = cells
            return Deduction(
                "Single Choice", f"Cell {cell} must be a Queen because {name} has no other valid slot", cell
            )
    return None


def shared_attack(state: QueensState) -> Deduction | None:
    for name, cells in state.units():
        if len(cells) < 2:
            continue
        hit = {x for x in state.candidates - cells if all(state.attacks(cell, x) for cell in cells)}
        if hit:
            return Deduction(
                "Shared Attack",
                f"every slot left in {name} attacks {sorted(hit)}, so none of them can be a Queen",
                eliminate=frozenset(hit),
            )
    return None


def confinement(state: QueensState) -> Deduction | None:
    n = state.board.size
    groupings: list[tuple[str, str, Callable[[Cell], int], Callable[[Cell], int]]] = [
        ("Region", "row", state.region, lambda x: x[0]),
        ("Region", "column", state.region, lambda x: x[1]),
        ("Row", "region", lambda x: x[0], state.region),
        ("Column", "region", lambda x: x[1], state.region),
    ]
    for owner_name, line_name, owner_of, line_of in groupings:
        owners = sorted({owner_of(x) for x in state.candidates})
        for k in range(1, max(1, n // 2) + 1):
            for group in combinations(owners, k):
                members = {x for x in state.candidates if owner_of(x) in group}
                lines = {line_of(x) for x in members}
                if len(lines) != k:
                    continue
                outside = {x for x in state.candidates if line_of(x) in lines and owner_of(x) not in group}
                if outside:
                    owners_text = ", ".join(f"{owner_name} {o}" for o in group)
                    lines_text = ", ".join(f"{line_name} {line}" for line in sorted(lines))
                    return Deduction(
                        "Confinement",
                        f"{owners_text} fit entirely inside {lines_text}, so {sorted(outside)} cannot be Queens",
                        eliminate=frozenset(outside),
                    )
    return None


def try_and_fail(state: QueensState) -> Deduction | None:
    for cell in sorted(state.candidates):
        trial = QueensState(state.board, set(state.candidates), set(state.placed))
        trial.apply(Deduction("Trial", "", cell))
        while not trial.stuck() and not trial.solved():
            forced = single_choice(trial)
            if forced is None:
                break
            trial.apply(forced)
        if trial.stuck():
            return Deduction(
                "Try and Fail",
                f"a Queen on {cell} leaves some row, column or region with no slot, so {cell} is not a Queen",
                eliminate=frozenset({cell}),
            )
    return None


RULES: tuple[Callable[[QueensState], Deduction | None], ...] = (single_choice, shared_attack, confinement, try_and_fail)


def next_deduction(state: QueensState) -> Deduction | None:
    """The first deduction any rule can prove, simplest rule first."""
    for rule in RULES:
        deduction = rule(state)
        if deduction is not None:
            return deduction
    return None
