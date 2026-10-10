"""Queens (Crowns) as exact cover, solved with Knuth's Algorithm X on Dancing Links.

Primary items (covered exactly once): every row, column and colour region.
Secondary items (covered at most once): every 2x2 window of the board. Two
crowns touch (including diagonally) iff they share a 2x2 window, and rows and
columns are already exclusive, so the windows encode non-adjacency exactly.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from core.solver_base import BaseSolver, UnsolvableError
from solvers.queens import rules

Cell = tuple[int, int]


@dataclass(frozen=True)
class QueensBoard:
    """An N x N grid of region ids in ``0..N-1``."""

    regions: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        n = len(self.regions)
        if any(len(row) != n for row in self.regions):
            raise ValueError("Queens board must be square")
        ids = {region for row in self.regions for region in row}
        if ids != set(range(n)):
            raise ValueError(f"Queens board needs exactly regions 0..{n - 1}, got {sorted(ids)}")

    @property
    def size(self) -> int:
        return len(self.regions)


class Node:
    """A Dancing Links node; column headers additionally track ``size``."""

    __slots__ = ("left", "right", "up", "down", "column", "row_id", "size", "name")

    def __init__(self, column: Node | None = None, row_id: int = -1, name: str = "") -> None:
        self.left: Node = self
        self.right: Node = self
        self.up: Node = self
        self.down: Node = self
        self.column: Node = column if column is not None else self
        self.row_id = row_id
        self.size = 0
        self.name = name


class DancingLinks:
    """Toroidal doubly-linked sparse matrix for Algorithm X."""

    def __init__(self, primary: int, secondary: int) -> None:
        """Columns ``0..primary-1`` must be covered exactly once, the next ``secondary`` at most once."""
        self.root = Node(name="root")
        self.columns: list[Node] = []
        for index in range(primary):
            column = Node(name=str(index))
            column.left, column.right = self.root.left, self.root
            self.root.left.right = column
            self.root.left = column
            self.columns.append(column)
        # Secondary headers stay out of the root list, so search never requires them.
        self.columns.extend(Node(name=str(primary + index)) for index in range(secondary))
        self.nodes_expanded = 0

    def add_row(self, row_id: int, column_ids: list[int]) -> None:
        """Append a matrix row with 1s in ``column_ids``."""
        first: Node | None = None
        for column_id in column_ids:
            column = self.columns[column_id]
            node = Node(column=column, row_id=row_id)
            node.up, node.down = column.up, column
            column.up.down = node
            column.up = node
            column.size += 1
            if first is None:
                first = node
            else:
                node.left, node.right = first.left, first
                first.left.right = node
                first.left = node

    @staticmethod
    def cover(column: Node) -> None:
        column.right.left = column.left
        column.left.right = column.right
        row = column.down
        while row is not column:
            node = row.right
            while node is not row:
                node.down.up = node.up
                node.up.down = node.down
                node.column.size -= 1
                node = node.right
            row = row.down

    @staticmethod
    def uncover(column: Node) -> None:
        row = column.up
        while row is not column:
            node = row.left
            while node is not row:
                node.column.size += 1
                node.down.up = node
                node.up.down = node
                node = node.left
            row = row.up
        column.right.left = column
        column.left.right = column

    def search(self, partial: list[int] | None = None) -> Iterator[list[int]]:
        """Yield every exact cover as a list of row ids (Knuth's Algorithm X)."""
        partial = [] if partial is None else partial
        if self.root.right is self.root:
            yield list(partial)
            return
        # Minimum-remaining-values heuristic: branch on the tightest column.
        column = self.root.right
        best = column
        while column is not self.root:
            if column.size < best.size:
                best = column
            column = column.right
        if best.size == 0:
            return
        self.cover(best)
        row = best.down
        while row is not best:
            self.nodes_expanded += 1
            partial.append(row.row_id)
            node = row.right
            while node is not row:
                self.cover(node.column)
                node = node.right
            yield from self.search(partial)
            node = row.left
            while node is not row:
                self.uncover(node.column)
                node = node.left
            partial.pop()
            row = row.down
        self.uncover(best)


class QueensSolver(BaseSolver[QueensBoard, list[Cell]]):
    """Place one crown per row, column and region with no two crowns touching."""

    name = "queens"

    def solve(self, board_state: QueensBoard) -> list[Cell]:
        solution = next(self.solutions(board_state), None)
        if solution is None:
            raise UnsolvableError("no crown placement satisfies the row, column, region and adjacency rules")
        return solution

    def solutions(self, board_state: QueensBoard) -> Iterator[list[Cell]]:
        """Yield every valid crown placement; take the first two to test uniqueness."""
        n = board_state.size
        # Column ids: rows 0..n-1, columns n..2n-1, regions 2n..3n-1, then the (n-1)^2 secondary 2x2 windows.
        links = DancingLinks(primary=3 * n, secondary=(n - 1) ** 2)
        for r in range(n):
            for c in range(n):
                windows = [
                    3 * n + wr * (n - 1) + wc
                    for wr in (r - 1, r)
                    for wc in (c - 1, c)
                    if 0 <= wr < n - 1 and 0 <= wc < n - 1
                ]
                links.add_row(r * n + c, [r, n + c, 2 * n + board_state.regions[r][c], *windows])
        for solution in links.search():
            self.nodes_expanded = links.nodes_expanded
            yield sorted(divmod(row_id, n) for row_id in solution)
        self.nodes_expanded = links.nodes_expanded

    def explain(self, board_state: QueensBoard, solution: list[Cell]) -> None:
        """Replay the solution with human rules; branch on the known solution only when no rule fires."""
        state = rules.QueensState.start(board_state)
        while not state.solved():
            deduction = rules.next_deduction(state)
            if deduction is None:
                cell = min(set(solution) - state.placed)
                deduction = rules.Deduction(
                    "Search",
                    f"No rule applies; Algorithm X branches on Cell {cell}, which leads to a valid completion",
                    cell,
                )
            state.apply(deduction)
            cell = deduction.place if deduction.place is not None else min(deduction.eliminate)
            self.explainer.add(deduction.rule, deduction.reason, cell)


def is_valid_solution(board: QueensBoard, queens: list[Cell]) -> bool:
    """Check every Queens rule; used by tests and the benchmark accuracy column."""
    n = board.size
    if len(queens) != n:
        return False
    rows = {r for r, _ in queens}
    cols = {c for _, c in queens}
    regions = {board.regions[r][c] for r, c in queens}
    if len(rows) != n or len(cols) != n or len(regions) != n:
        return False
    return all(max(abs(a[0] - b[0]), abs(a[1] - b[1])) > 1 for i, a in enumerate(queens) for b in queens[i + 1 :])
