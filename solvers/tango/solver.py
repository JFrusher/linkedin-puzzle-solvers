"""Tango (Sun & Moon) reduced to Boolean satisfiability and solved with PySAT.

Variable ``x(r, c)`` is true iff cell (r, c) holds a Sun. CNF encoding:

* givens: unit clauses;
* balance: ``sum(row) = sum(col) = N/2`` via a sequential-counter cardinality encoding;
* no three in a row/column: ``(¬a ∨ ¬b ∨ ¬c) ∧ (a ∨ b ∨ c)`` for every window of three;
* ``=`` edges: ``(¬a ∨ b) ∧ (a ∨ ¬b)``; ``x`` edges: ``(a ∨ b) ∧ (¬a ∨ ¬b)``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum

from pysat.card import CardEnc, EncType
from pysat.solvers import Minisat22

from core.solver_base import BaseSolver, UnsolvableError

Cell = tuple[int, int]


class Symbol(StrEnum):
    SUN = "sun"
    MOON = "moon"

    @property
    def other(self) -> Symbol:
        return Symbol.MOON if self is Symbol.SUN else Symbol.SUN


class Relation(StrEnum):
    EQUAL = "="
    OPPOSITE = "x"


@dataclass(frozen=True)
class Constraint:
    """A relation sign between two orthogonally adjacent cells."""

    a: Cell
    b: Cell
    relation: Relation


@dataclass(frozen=True)
class TangoBoard:
    size: int
    givens: Mapping[Cell, Symbol] = field(default_factory=dict)
    constraints: tuple[Constraint, ...] = ()

    def __post_init__(self) -> None:
        if self.size % 2:
            raise ValueError(f"Tango grid size must be even, got {self.size}")
        for constraint in self.constraints:
            (r1, c1), (r2, c2) = constraint.a, constraint.b
            if abs(r1 - r2) + abs(c1 - c2) != 1:
                raise ValueError(f"constraint cells {constraint.a} and {constraint.b} are not adjacent")


Grid = list[list[Symbol]]


class TangoSolver(BaseSolver[TangoBoard, Grid]):
    """Encode the board as CNF, solve with MiniSat and explain with human rules."""

    name = "tango"

    def solve(self, board_state: TangoBoard) -> Grid:
        n = board_state.size

        def var(cell: Cell) -> int:
            return cell[0] * n + cell[1] + 1

        clauses: list[list[int]] = []
        for cell, symbol in board_state.givens.items():
            clauses.append([var(cell) if symbol is Symbol.SUN else -var(cell)])
        lines = [[(i, j) for j in range(n)] for i in range(n)] + [[(j, i) for j in range(n)] for i in range(n)]
        for line in lines:
            literals = [var(cell) for cell in line]
            for k in range(n - 2):
                window = literals[k : k + 3]
                clauses.append([-lit for lit in window])
                clauses.append(list(window))
        for constraint in board_state.constraints:
            a, b = var(constraint.a), var(constraint.b)
            if constraint.relation is Relation.EQUAL:
                clauses += [[-a, b], [a, -b]]
            else:
                clauses += [[a, b], [-a, -b]]
        top = n * n  # cardinality encodings allocate auxiliary variables above the cell variables
        for line in lines:
            encoding = CardEnc.equals(
                lits=[var(cell) for cell in line], bound=n // 2, top_id=top, encoding=EncType.seqcounter
            )
            top = max(top, encoding.nv)
            clauses += encoding.clauses

        with Minisat22(bootstrap_with=clauses) as sat:
            if not sat.solve():
                raise UnsolvableError("the Tango CNF is unsatisfiable")
            model = set(lit for lit in sat.get_model() if lit > 0)
            self.nodes_expanded = int(sat.accum_stats()["decisions"])
        return [[Symbol.SUN if var((r, c)) in model else Symbol.MOON for c in range(n)] for r in range(n)]

    def explain(self, board_state: TangoBoard, solution: Grid) -> None:
        """Replay the solution as human deductions; use the SAT model only where no rule applies."""
        board = board_state
        n = board.size
        known: dict[Cell, Symbol] = dict(board.givens)
        for cell, symbol in sorted(known.items()):
            self.explainer.add("Given", f"Cell {cell} starts as {symbol.value}", cell)
        while len(known) < n * n:
            deduction = _next_deduction(board, known)
            if deduction is None:
                cell = min((r, c) for r in range(n) for c in range(n) if (r, c) not in known)
                deduction = (cell, solution[cell[0]][cell[1]], "SAT", "the SAT model fixes it; no single rule applies")
            cell, symbol, rule, reason = deduction
            known[cell] = symbol
            self.explainer.add(rule, f"Cell {cell} must be {symbol.value} because {reason}", cell)


def _next_deduction(board: TangoBoard, known: dict[Cell, Symbol]) -> tuple[Cell, Symbol, str, str] | None:
    n = board.size
    for constraint in board.constraints:
        for src, dst in ((constraint.a, constraint.b), (constraint.b, constraint.a)):
            if src in known and dst not in known:
                symbol = known[src] if constraint.relation is Relation.EQUAL else known[src].other
                return dst, symbol, "Relation", f"of the '{constraint.relation.value}' sign with {src}"
    lines = [[(i, j) for j in range(n)] for i in range(n)] + [[(j, i) for j in range(n)] for i in range(n)]
    for line in lines:
        for k in range(n - 2):
            a, b, c = line[k : k + 3]
            for x, y, target in ((a, b, c), (b, c, a), (a, c, b)):
                if x in known and y in known and target not in known and known[x] == known[y]:
                    return target, known[x].other, "No Three", f"{x} and {y} are both {known[x].value}"
        for symbol in Symbol:
            if sum(known.get(cell) == symbol for cell in line) == n // 2:
                for cell in line:
                    if cell not in known:
                        return cell, symbol.other, "Balance", f"its line already holds {n // 2} {symbol.value}s"
    return None


def is_valid_solution(board: TangoBoard, grid: Grid) -> bool:
    """Check every Tango rule against a filled grid."""
    n = board.size
    if any(grid[r][c] != symbol for (r, c), symbol in board.givens.items()):
        return False
    lines = [[grid[i][j] for j in range(n)] for i in range(n)] + [[grid[j][i] for j in range(n)] for i in range(n)]
    for line in lines:
        if line.count(Symbol.SUN) != n // 2:
            return False
        if any(line[k] == line[k + 1] == line[k + 2] for k in range(n - 2)):
            return False
    for constraint in board.constraints:
        same = grid[constraint.a[0]][constraint.a[1]] == grid[constraint.b[0]][constraint.b[1]]
        if same != (constraint.relation is Relation.EQUAL):
            return False
    return True
