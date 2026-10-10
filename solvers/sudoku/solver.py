"""Mini Sudoku (6x6, 2x3 boxes) solved as SAT, explained with naked and hidden singles.

Variable ``x(r, c, d)`` is true iff cell (r, c) holds digit d. Exactly-one constraints cover every cell,
and every digit in every row, column and box; givens are unit clauses.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from itertools import combinations

from pysat.solvers import Minisat22

from core.solver_base import BaseSolver, UnsolvableError

Cell = tuple[int, int]
Grid = list[list[int]]
SIZE = 6
BOX_ROWS, BOX_COLS = 2, 3
DIGITS = range(1, SIZE + 1)


@dataclass(frozen=True)
class SudokuBoard:
    givens: Mapping[Cell, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for (r, c), digit in self.givens.items():
            if not (0 <= r < SIZE and 0 <= c < SIZE and digit in DIGITS):
                raise ValueError(f"given {digit} at {(r, c)} is outside a {SIZE}x{SIZE} board of digits 1-{SIZE}")


def units() -> list[tuple[str, list[Cell]]]:
    """Every row, column and box with its cells."""
    rows = [(f"Row {r}", [(r, c) for c in range(SIZE)]) for r in range(SIZE)]
    cols = [(f"Column {c}", [(r, c) for r in range(SIZE)]) for c in range(SIZE)]
    boxes = [
        (f"Box {b}", [(r, c) for r in range(br, br + BOX_ROWS) for c in range(bc, bc + BOX_COLS)])
        for b, (br, bc) in enumerate((br, bc) for br in range(0, SIZE, BOX_ROWS) for bc in range(0, SIZE, BOX_COLS))
    ]
    return rows + cols + boxes


def _var(cell: Cell, digit: int) -> int:
    return (cell[0] * SIZE + cell[1]) * SIZE + digit


def _exactly_one(literals: list[int]) -> list[list[int]]:
    return [literals] + [[-a, -b] for a, b in combinations(literals, 2)]


class SudokuSolver(BaseSolver[SudokuBoard, Grid]):
    name = "mini-sudoku"

    def solve(self, board_state: SudokuBoard) -> Grid:
        grid = next(self.solutions(board_state), None)
        if grid is None:
            raise UnsolvableError("no grid satisfies the givens and the row, column and box rules")
        return grid

    def solutions(self, board_state: SudokuBoard) -> Iterator[Grid]:
        """Yield every valid grid, blocking each model once found; take two to test uniqueness."""
        cells = [(r, c) for r in range(SIZE) for c in range(SIZE)]
        clauses = [[_var(cell, digit)] for cell, digit in board_state.givens.items()]
        for cell in cells:
            clauses += _exactly_one([_var(cell, d) for d in DIGITS])
        for _, unit in units():
            for d in DIGITS:
                clauses += _exactly_one([_var(cell, d) for cell in unit])
        with Minisat22(bootstrap_with=clauses) as sat:
            while sat.solve():
                model = {lit for lit in sat.get_model() if lit > 0}
                self.nodes_expanded = int(sat.accum_stats()["decisions"])
                grid = [[next(d for d in DIGITS if _var((r, c), d) in model) for c in range(SIZE)] for r in range(SIZE)]
                yield grid
                sat.add_clause([-_var((r, c), grid[r][c]) for r, c in cells])

    def explain(self, board_state: SudokuBoard, solution: Grid) -> None:
        """Replay with naked and hidden singles; take the SAT answer only where neither applies."""
        known: dict[Cell, int] = dict(board_state.givens)
        while len(known) < SIZE * SIZE:
            step = _next_single(known)
            if step is None:
                cell = min((r, c) for r in range(SIZE) for c in range(SIZE) if (r, c) not in known)
                step = (cell, solution[cell[0]][cell[1]], "SAT", "no single applies; the SAT model fixes it")
            cell, digit, rule, reason = step
            known[cell] = digit
            self.explainer.add(rule, f"Cell {cell} must be {digit} because {reason}", cell)


def candidates(known: dict[Cell, int], cell: Cell) -> set[int]:
    """Digits not yet used in any unit containing ``cell``."""
    seen = {known[other] for _, unit in units() if cell in unit for other in unit if other in known}
    return set(DIGITS) - seen


def _next_single(known: dict[Cell, int]) -> tuple[Cell, int, str, str] | None:
    open_cells = [(r, c) for r in range(SIZE) for c in range(SIZE) if (r, c) not in known]
    for cell in open_cells:
        options = candidates(known, cell)
        if len(options) == 1:
            (digit,) = options
            return cell, digit, "Naked Single", "every other digit already appears in its row, column or box"
    for name, unit in units():
        for digit in DIGITS:
            if any(known.get(cell) == digit for cell in unit):
                continue
            places = [cell for cell in unit if cell not in known and digit in candidates(known, cell)]
            if len(places) == 1:
                return places[0], digit, "Hidden Single", f"it is the only place for {digit} in {name}"
    return None


def is_valid_solution(board: SudokuBoard, grid: Grid) -> bool:
    if any(grid[r][c] != digit for (r, c), digit in board.givens.items()):
        return False
    return all(sorted(grid[r][c] for r, c in unit) == list(DIGITS) for _, unit in units())
